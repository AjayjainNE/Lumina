"""LUMINA — LLM-as-Judge: Mistral 4-axis scoring → RL reward signal."""
from __future__ import annotations
import json, os, time, logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "hdd30lQaVEcAv9WWugWTh1nOxoO1a3hH")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-large-latest"

JUDGE_SYSTEM = """You are an expert AI evaluation judge. Score the agent output on 4 axes (0.0-1.0).
Return ONLY valid JSON, no markdown, no preamble:
{"accuracy":<float>,"faithfulness":<float>,"completeness":<float>,"coherence":<float>,"reasoning":"<str>"}"""


def _safe_parse(raw: str) -> Optional[Dict]:
    """Tolerant JSON parser: strips fences, repairs trailing commas."""
    import re
    if not raw:
        return None
    s = raw.strip()
    # strip markdown fences
    for fence in ["```json", "```JSON", "```"]:
        if s.startswith(fence):
            s = s[len(fence):]
            break
    s = s.rstrip("`").strip()
    # try direct
    for attempt in [s, s.replace("'", '"'),
                    re.sub(r',\s*([}\]])', r'\1', s)]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    # extract first {...}
    m = re.search(r'\{[^{}]+\}', s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class JudgeScore:
    accuracy: float; faithfulness: float
    completeness: float; coherence: float
    reasoning: str
    composite: float = 0.0; reward: float = 0.0; latency_ms: float = 0.0

    def __post_init__(self):
        w = dict(accuracy=0.30, faithfulness=0.35, completeness=0.20, coherence=0.15)
        self.composite = sum(w[k] * getattr(self, k) for k in w)
        self.reward = 2.0 * self.composite - 1.0

    def to_dict(self): return asdict(self)
    def passed(self, threshold=0.65): return self.composite >= threshold


@dataclass
class EvalRecord:
    chunk_id: str; agent_name: str
    source_excerpt: str; agent_output: str
    score: JudgeScore; model: str = MISTRAL_MODEL
    timestamp: float = 0.0
    def to_dict(self):
        d = asdict(self); d["score"] = self.score.to_dict(); return d


class LLMJudge:
    def __init__(self, api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL, timeout=30.0):
        self.api_key = api_key; self.model = model; self.timeout = timeout
        self._history: List[EvalRecord] = []

    def score(self, source_excerpt: str, agent_output: str,
              agent_name="unknown", chunk_id="", task_type="document_qa") -> JudgeScore:
        prompt = (f"Task: {task_type}\n\n=== SOURCE ===\n{source_excerpt}\n\n"
                  f"=== OUTPUT ===\n{agent_output}\n\nReturn JSON only.")
        t0 = time.perf_counter()
        raw = self._call_mistral(prompt)
        latency = (time.perf_counter() - t0) * 1000
        score = self._parse(raw, latency)
        self._history.append(EvalRecord(
            chunk_id=chunk_id or f"chunk_{len(self._history)}",
            agent_name=agent_name, source_excerpt=source_excerpt[:500],
            agent_output=agent_output[:500], score=score, timestamp=time.time()))
        logger.info("Judge [%s] composite=%.3f reward=%.3f", agent_name, score.composite, score.reward)
        return score

    def session_summary(self) -> Dict:
        if not self._history: return {}
        composites = [r.score.composite for r in self._history]
        by_agent: Dict[str, List[float]] = {}
        for r in self._history:
            by_agent.setdefault(r.agent_name, []).append(r.score.composite)
        return {
            "total_evaluations": len(self._history),
            "mean_composite": round(sum(composites)/len(composites), 4),
            "pass_rate": round(sum(s >= 0.65 for s in composites)/len(composites), 4),
            "by_agent": {k: round(sum(v)/len(v), 4) for k, v in by_agent.items()},
            "axis_means": {
                ax: round(sum(getattr(r.score, ax) for r in self._history)/len(self._history), 4)
                for ax in ["accuracy","faithfulness","completeness","coherence"]},
        }

    def export_history(self, path: str):
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._history], f, indent=2)

    def _call_mistral(self, user_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "temperature": 0.1, "max_tokens": 256,
                   "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                 {"role": "user", "content": user_prompt}]}
        for attempt in range(1, 4):
            try:
                r = httpx.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == 3: return self._fallback()
                time.sleep(1.5 * attempt)
        return self._fallback()

    def _parse(self, raw: str, latency_ms=0.0) -> JudgeScore:
        data = _safe_parse(raw)
        if not data:
            return JudgeScore(0.5, 0.5, 0.5, 0.5, "parse error", latency_ms=latency_ms)
        try:
            return JudgeScore(
                accuracy=float(data.get("accuracy", 0.5)),
                faithfulness=float(data.get("faithfulness", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                coherence=float(data.get("coherence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                latency_ms=latency_ms)
        except Exception:
            return JudgeScore(0.5, 0.5, 0.5, 0.5, "parse error", latency_ms=latency_ms)

    @staticmethod
    def _fallback(): return '{"accuracy":0.5,"faithfulness":0.5,"completeness":0.5,"coherence":0.5,"reasoning":"fallback"}'
