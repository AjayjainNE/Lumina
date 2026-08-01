"""LUMINA — Curriculum Learning: difficulty estimation, 3 strategies, PER buffer, reward shaper."""
from __future__ import annotations
import logging, math, random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)


def estimate_difficulty(chunk: str) -> float:
    import re; words = chunk.split(); n = max(len(words), 1)
    if n < 5: return 0.9
    length = 1.0 - min(abs(n - 140) / 140, 1.0)
    numeric = min(len(re.findall(r'\d+\.?\d*', chunk)) / n * 8, 1.0)
    unique = 1.0 - min(len(set(w.lower() for w in words)) / n * 1.5, 1.0)
    sents = [len(s.split()) for s in re.split(r'[.!?]+', chunk) if s.strip()]
    sent_len = min(sum(sents)/max(len(sents),1)/40, 1.0)
    jargon = min(len(re.findall(
        r'\b(?:EBITDA|amortisation|depreciation|impairment|covenant|diluted|CAGR|YoY|QoQ)\b',
        chunk, re.IGNORECASE))/3, 1.0)
    d = 0.20*(1-length) + 0.25*numeric + 0.20*unique + 0.20*sent_len + 0.15*jargon
    return round(min(max(d, 0.0), 1.0), 4)


@dataclass
class ChunkDifficulty:
    chunk_id: str; text: str; embedding: np.ndarray
    difficulty: float = 0.5; quality_score: float = 1.0; agent_hint: str = ""


class EasyCurriculumScheduler:
    def order(self, chunks: List[ChunkDifficulty]) -> List[ChunkDifficulty]:
        return sorted(chunks, key=lambda c: c.difficulty)
    def step(self, step, mean_reward): pass


class MixedCurriculumScheduler:
    def __init__(self, anneal_steps=500, easy_fraction_start=0.80, easy_fraction_end=0.20):
        self.anneal_steps = anneal_steps
        self.easy_start = easy_fraction_start; self.easy_end = easy_fraction_end
        self._easy_fraction = easy_fraction_start

    def step(self, step, mean_reward):
        progress = min(step / self.anneal_steps, 1.0)
        cosine = (1 - math.cos(math.pi * progress)) / 2
        self._easy_fraction = self.easy_start + cosine * (self.easy_end - self.easy_start)

    def order(self, chunks: List[ChunkDifficulty]) -> List[ChunkDifficulty]:
        if not chunks: return chunks
        s = sorted(chunks, key=lambda c: c.difficulty)
        n_easy = max(1, int(len(s) * self._easy_fraction))
        easy, hard = s[:n_easy], s[n_easy:]
        result, ei, hi = [], 0, 0
        while ei < len(easy) or hi < len(hard):
            if ei < len(easy): result.append(easy[ei]); ei += 1
            if hi < len(hard): result.append(hard[hi]); hi += 1
        return result


class AdaptiveCurriculumScheduler:
    BANDS = [(0.0,0.25),(0.25,0.50),(0.50,0.75),(0.75,1.01)]
    def __init__(self, reward_threshold=0.70, window_size=20, start_band=0):
        self.threshold = reward_threshold; self.window = window_size
        self._band_idx = start_band
        self._rewards: Deque[float] = deque(maxlen=window_size)

    @property
    def current_band(self): return self.BANDS[self._band_idx]

    def step(self, step, mean_reward):
        self._rewards.append(mean_reward)
        if len(self._rewards) >= self.window:
            rolling = sum(self._rewards) / len(self._rewards)
            if rolling >= self.threshold and self._band_idx < len(self.BANDS)-1:
                self._band_idx += 1; self._rewards.clear()
                logger.info("[curriculum] Promoted → band %d", self._band_idx)

    def order(self, chunks: List[ChunkDifficulty]) -> List[ChunkDifficulty]:
        lo, hi = self.current_band
        band = [c for c in chunks if lo <= c.difficulty < hi]
        if not band:
            centre = (lo+hi)/2
            band = sorted(chunks, key=lambda c: abs(c.difficulty - centre))
        random.shuffle(band); return band


@dataclass
class Experience:
    state: np.ndarray; action: int; reward: float
    log_prob: float; value: float; priority: float = 1.0


class PrioritisedReplayBuffer:
    def __init__(self, capacity=2000, alpha=0.6, beta_start=0.4, beta_steps=1000):
        self.capacity = capacity; self.alpha = alpha
        self.beta_start = beta_start; self.beta_steps = beta_steps
        self._buf: List[Experience] = []; self._prios: List[float] = []; self._step = 0

    @property
    def beta(self):
        return self.beta_start + min(self._step/self.beta_steps,1.0) * (1.0 - self.beta_start)

    def add(self, exp: Experience):
        max_p = max(self._prios, default=1.0)
        if len(self._buf) >= self.capacity:
            idx = self._prios.index(min(self._prios))
            self._buf[idx] = exp; self._prios[idx] = max_p
        else:
            self._buf.append(exp); self._prios.append(max_p)

    def sample(self, batch_size: int):
        bs = min(batch_size, len(self._buf))
        probs = np.array(self._prios)**self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(len(self._buf), bs, replace=False, p=probs)
        exps = [self._buf[i] for i in idxs]
        weights = (len(self._buf) * probs[idxs]) ** (-self.beta)
        weights /= weights.max()
        self._step += 1
        return exps, list(weights), list(idxs)

    def update_priorities(self, idxs, rewards, baseline=0.0):
        for i, r in zip(idxs, rewards):
            self._prios[i] = (abs(r - baseline) + 1e-6) ** self.alpha

    def __len__(self): return len(self._buf)


class NormalisedRewardShaper:
    def __init__(self, epsilon=0.05, clip=1.0, alpha=0.05):
        self.epsilon = epsilon; self.clip = clip; self.alpha = alpha
        self._means: Dict[str,float] = {}; self._vars: Dict[str,float] = {}

    def shape(self, agent: str, raw: float) -> float:
        if abs(raw) < self.epsilon: return 0.0
        mu = self._means.get(agent, raw); var = self._vars.get(agent, 1.0)
        self._means[agent] = (1-self.alpha)*mu + self.alpha*raw
        self._vars[agent]  = max((1-self.alpha)*var + self.alpha*(raw-mu)**2, 1e-8)
        normalised = (raw - self._means[agent]) / math.sqrt(self._vars[agent])
        return float(np.clip(normalised, -self.clip, self.clip))

    def stats(self): return {a: {"mean": self._means[a], "std": math.sqrt(self._vars[a])} for a in self._means}


def build_curriculum(strategy="adaptive", **kwargs):
    return {"easy": EasyCurriculumScheduler, "mixed": MixedCurriculumScheduler,
            "adaptive": AdaptiveCurriculumScheduler}[strategy](**kwargs)
