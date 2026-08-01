"""LUMINA — Multi-Agent Orchestrator: chunk → encode → CGR route → dispatch → synthesise → judge → PPO."""
from __future__ import annotations
import logging, os, time, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
logger = logging.getLogger(__name__)

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY", "hdd30lQaVEcAv9WWugWTh1nOxoO1a3hH")


def encode_chunk(text: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        _enc = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _enc.encode(text, normalize_embeddings=True)
        if emb.shape[0] < 768: emb = np.pad(emb, (0, 768 - emb.shape[0]))
        return emb[:768].astype(np.float32)
    except Exception:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(768).astype(np.float32)


def chunk_document(text: str, chunk_size=512, overlap=64) -> List[str]:
    words = text.split(); chunks = []; step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        c = " ".join(words[start:start+chunk_size])
        if c.strip(): chunks.append(c)
    return chunks


def _mistral_synthesise(agent_outputs: List["AgentOutput"], query: str, api_key: str) -> str:
    import httpx
    outputs_text = "\n\n".join(f"[{ao.agent_name.upper()}]\n{ao.output}"
                               for ao in agent_outputs if ao.success)
    if not outputs_text: return "No agent produced valid output."
    try:
        r = httpx.post("https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "temperature": 0.3, "max_tokens": 1024,
                  "messages": [
                      {"role": "system", "content": "Synthesise agent outputs into a single coherent response. Be concise and factual."},
                      {"role": "user", "content": f"Query: {query}\n\nAgent outputs:\n{outputs_text}\n\nSynthesise:"}]},
            timeout=30.0)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("Mistral synthesis failed: %s", e)
        return "\n\n".join(f"**{ao.agent_name}**: {ao.output}" for ao in agent_outputs if ao.success)


@dataclass
class AgentOutput:
    agent_name: str; output: str
    metadata: Dict[str,Any] = field(default_factory=dict)
    latency_ms: float = 0.0; success: bool = True


class BaseAgent:
    name: str = "base"
    def process(self, chunk: str, document_id="") -> AgentOutput:
        raise NotImplementedError


@dataclass
class OrchestrationResult:
    document_id: str; query: str; final_answer: str
    agent_outputs: List[AgentOutput]
    routing_decisions: list
    judge_scores: list
    ppo_metrics: Dict[str,float]
    total_latency_ms: float
    mean_composite_score: float
    quality_passed: bool

    def summary(self):
        return (f"doc={self.document_id} score={self.mean_composite_score:.3f} "
                f"pass={self.quality_passed} latency={self.total_latency_ms:.0f}ms")


class LuminaOrchestrator:
    def __init__(self, agents: Dict[str, BaseAgent], router=None, judge=None,
                 mistral_api_key="", mlflow_enabled=True, update_policy_every=10):
        from routing.cgr_algorithm import CGRRouter
        from evaluation.llm_judge import LLMJudge
        self.agents = agents
        self.router = router or CGRRouter()
        self.judge = judge or LLMJudge(api_key=mistral_api_key or MISTRAL_KEY)
        self.mistral_key = mistral_api_key or MISTRAL_KEY
        self.mlflow_enabled = mlflow_enabled
        self.update_policy_every = update_policy_every
        self._processed = 0
        if mlflow_enabled:
            try:
                import mlflow; mlflow.set_experiment("lumina_orchestration")
            except Exception: self.mlflow_enabled = False

    def run(self, document_text: str, query: str, document_id=None, chunk_size=512) -> OrchestrationResult:
        doc_id = document_id or str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        chunks = chunk_document(document_text, chunk_size=chunk_size)
        all_outputs, all_decisions, all_scores = [], [], []
        for i, chunk in enumerate(chunks):
            state = encode_chunk(chunk)
            decision = self.router.route(state)
            all_decisions.append(decision)
            for agent_name in decision.selected_agents:
                agent = self.agents.get(agent_name)
                if not agent: continue
                at = time.perf_counter()
                try:
                    out = agent.process(chunk, document_id=doc_id)
                    out.latency_ms = (time.perf_counter()-at)*1000
                    all_outputs.append(out)
                except Exception as e:
                    logger.error("Agent %s failed: %s", agent_name, e)
                    all_outputs.append(AgentOutput(agent_name, "", success=False, metadata={"error":str(e)}))
            for ao in [o for o in all_outputs[-len(decision.selected_agents):] if o.success]:
                score = self.judge.score(chunk, ao.output, ao.agent_name,
                                         f"{doc_id}_c{i:03d}", ao.metadata.get("task_type","doc_qa"))
                all_scores.append(score)
                self.router.record(state, decision, score.reward)
        self._processed += 1
        ppo = {}
        if self._processed % self.update_policy_every == 0:
            ppo = self.router.update()
        final = _mistral_synthesise(all_outputs, query, self.mistral_key)
        if chunks and final:
            synth_score = self.judge.score(" ".join(chunks[:2]), final, "synthesiser", f"{doc_id}_synth")
            all_scores.append(synth_score)
        composites = [s.composite for s in all_scores]
        mean_comp = sum(composites)/len(composites) if composites else 0.5
        total_ms = (time.perf_counter()-t0)*1000
        result = OrchestrationResult(doc_id, query, final, all_outputs, all_decisions,
                                     all_scores, ppo, total_ms, mean_comp, mean_comp >= 0.65)
        if self.mlflow_enabled:
            try:
                import mlflow
                with mlflow.start_run(run_name=f"doc_{doc_id}", nested=True):
                    mlflow.log_metrics({"mean_composite": mean_comp, "latency_ms": total_ms, **ppo})
            except Exception: pass
        return result
