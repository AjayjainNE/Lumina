"""
LUMINA — Confidence-Gated Routing (CGR) Algorithm
Novel PPO-trained RL router that dispatches document chunks to expert agents.
Reward signal: LLM-as-Judge composite score. Zero human labels required.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

AGENT_NAMES = ["document_qa", "ner", "summariser", "vqa", "tts"]
N_AGENTS = len(AGENT_NAMES)
STATE_DIM = 768
HIDDEN_DIM = 256
ENTROPY_COEFF = 0.01
CLIP_EPS = 0.2
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
TEMPERATURE = 0.7


@dataclass
class Trajectory:
    state: torch.Tensor
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool = False


@dataclass
class RoutingDecision:
    selected_agents: List[str]
    confidence_scores: Dict[str, float]
    action_index: int
    log_prob: float
    ensemble: bool = False


class ConfidenceEstimator(nn.Module):
    """Temperature-scaled MLP: chunk embedding → per-agent confidence probs."""
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM,
                 n_agents=N_AGENTS, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_agents),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net[2](self.net[1](self.net[0](state)))   # GELU after LayerNorm
        h = self.net[3](h)                                  # Dropout
        h = self.net[5](self.net[4](h))                     # hidden//2
        logits = self.net[6](h)
        calibrated = logits / self.temperature.clamp(min=0.1, max=2.0)
        probs = torch.softmax(calibrated, dim=-1)
        return probs, probs


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
    def forward(self, s): return self.net(s).squeeze(-1)


class CGRRouter:
    """PPO-based RL router. Learns dispatch policy from LLM-judge rewards."""
    ENSEMBLE_THRESHOLD = 0.40

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.actor = ConfidenceEstimator().to(self.device)
        self.critic = ValueNetwork().to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.trajectories: List[Trajectory] = []
        self._step = 0

    @torch.no_grad()
    def route(self, state_vector: np.ndarray) -> RoutingDecision:
        s = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs, conf = self.actor(s)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        action_idx = action.item()
        max_conf = probs.max().item()
        ensemble = max_conf < self.ENSEMBLE_THRESHOLD
        selected = AGENT_NAMES if ensemble else [AGENT_NAMES[action_idx % N_AGENTS]]
        conf_dict = {n: round(conf[0, i].item(), 4) for i, n in enumerate(AGENT_NAMES)}
        return RoutingDecision(selected_agents=selected, confidence_scores=conf_dict,
                               action_index=action_idx, log_prob=log_prob, ensemble=ensemble)

    def record(self, state: np.ndarray, decision: RoutingDecision, reward: float):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(s).item()
        self.trajectories.append(Trajectory(
            state=torch.tensor(state, dtype=torch.float32),
            action=decision.action_index, log_prob=decision.log_prob,
            reward=reward, value=value))

    def update(self, n_epochs=4) -> Dict:
        if len(self.trajectories) < 2:
            return {}
        states   = torch.stack([t.state for t in self.trajectories]).to(self.device)
        actions  = torch.tensor([t.action for t in self.trajectories], dtype=torch.long).to(self.device)
        old_lp   = torch.tensor([t.log_prob for t in self.trajectories]).to(self.device)
        rewards  = torch.tensor([t.reward for t in self.trajectories]).to(self.device)
        values   = torch.tensor([t.value for t in self.trajectories]).to(self.device)
        advs = self._gae(rewards, values)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        a_losses, c_losses, entropies = [], [], []
        for _ in range(n_epochs):
            probs, _ = self.actor(states)
            dist = Categorical(probs)
            new_lp = dist.log_prob(actions)
            ent = dist.entropy().mean()
            ratio = torch.exp(new_lp - old_lp)
            a_loss = -torch.min(ratio * advs,
                                torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advs).mean() \
                     - ENTROPY_COEFF * ent
            c_loss = nn.MSELoss()(self.critic(states), returns.detach())
            self.opt_actor.zero_grad(); a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_actor.step()
            self.opt_critic.zero_grad(); c_loss.backward(); self.opt_critic.step()
            a_losses.append(a_loss.item()); c_losses.append(c_loss.item()); entropies.append(ent.item())
        self.trajectories.clear(); self._step += 1
        return {"actor_loss": float(np.mean(a_losses)), "critic_loss": float(np.mean(c_losses)),
                "policy_entropy": float(np.mean(entropies)), "ppo_step": self._step}

    def _gae(self, rewards, values, lam=0.95):
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = values[t+1] if t+1 < len(values) else 0.0
            delta = rewards[t] + GAMMA * nv - values[t]
            gae = delta + GAMMA * lam * gae
            adv[t] = gae
        return adv

    def save(self, path: str):
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "step": self._step}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self._step = ckpt.get("step", 0)
