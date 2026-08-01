"""LUMINA — LLMOps: ExperimentTracker, PromptRegistry, DriftDetector."""
from __future__ import annotations
import json, logging, math, os, statistics, time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


class ExperimentTracker:
    def __init__(self, experiment_name="lumina_v1", tracking_uri=MLFLOW_URI):
        self.experiment_name = experiment_name
        self._run_registry: Dict[str,str] = {}
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except ImportError:
            self._mlflow = None

    def start_run(self, document_id: str, tags=None) -> str:
        run_id = f"local_{document_id}_{int(time.time())}"
        if self._mlflow:
            try:
                r = self._mlflow.start_run(run_name=f"doc_{document_id}",
                                            tags={**(tags or {}), "document_id": document_id})
                run_id = r.info.run_id
            except Exception: pass
        self._run_registry[document_id] = run_id; return run_id

    def log_metrics(self, metrics: Dict, step=None):
        if self._mlflow:
            try: self._mlflow.log_metrics(metrics, step=step)
            except Exception: pass

    def log_params(self, params: Dict):
        if self._mlflow:
            try: self._mlflow.log_params({k: str(v)[:250] for k,v in params.items()})
            except Exception: pass

    def end_run(self):
        if self._mlflow:
            try: self._mlflow.end_run()
            except Exception: pass


@dataclass
class PromptVersion:
    name: str; version: str; template: str
    description: str = ""; created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0; mean_score: float = 0.0

    def render(self, **kw) -> str: return self.template.format(**kw)
    def to_dict(self): return asdict(self)


class PromptRegistry:
    def __init__(self, store_path="config/prompt_registry.json"):
        self.store_path = store_path; self._store: Dict[str, List[PromptVersion]] = {}
        self._load(); self._seed_defaults()

    def register(self, name, version, template, description="", tags=None) -> PromptVersion:
        pv = PromptVersion(name=name, version=version, template=template,
                           description=description, tags=tags or [])
        self._store.setdefault(name, []).append(pv); self._save(); return pv

    def get(self, name, version="latest") -> Optional[PromptVersion]:
        vs = self._store.get(name, [])
        if not vs: return None
        return vs[-1] if version == "latest" else next((v for v in vs if v.version == version), None)

    def update_score(self, name, version, score):
        pv = self.get(name, version)
        if pv:
            pv.usage_count += 1; pv.mean_score = 0.1*score + 0.9*pv.mean_score; self._save()

    def ab_route(self, name, traffic_split=0.5) -> Optional[PromptVersion]:
        import random; vs = self._store.get(name, [])
        if len(vs) < 2: return self.get(name)
        return vs[-1] if random.random() < traffic_split else vs[-2]

    def list_prompts(self): return {k: [v.version for v in vs] for k,vs in self._store.items()}

    def _seed_defaults(self):
        if "judge_system" not in self._store:
            self.register("judge_system","v1",
                'You are an expert AI judge. Score on 4 axes (0-1): accuracy, faithfulness, completeness, coherence. Return JSON only: {{"accuracy":...,"faithfulness":...,"completeness":...,"coherence":...,"reasoning":"..."}}',
                "Primary judge prompt")
        if "synthesiser" not in self._store:
            self.register("synthesiser","v1",
                "Synthesise agent outputs for: {query}\n\nOutputs:\n{outputs}",
                "Synthesis prompt")

    def _load(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    raw = json.load(f)
                self._store = {k: [PromptVersion(**v) for v in vs] for k,vs in raw.items()}
            except Exception: pass

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        try:
            with open(self.store_path,"w") as f:
                json.dump({k:[pv.to_dict() for pv in vs] for k,vs in self._store.items()}, f, indent=2)
        except Exception: pass


@dataclass
class DriftAlert:
    agent_name: str; axis: str; baseline_mean: float; current_mean: float
    z_score: float; timestamp: float = field(default_factory=time.time); severity: str = "WARNING"


class DriftDetector:
    def __init__(self, window_size=50, baseline_size=200, z_threshold=2.5,
                 variance_factor=3.0, alert_callbacks=None):
        self.window_size = window_size; self.baseline_size = baseline_size
        self.z_threshold = z_threshold; self.variance_factor = variance_factor
        self.alert_callbacks = alert_callbacks or []
        self._buffers: Dict[str, Dict[str, deque]] = {}
        self._alerts: List[DriftAlert] = []

    def update(self, agent_name, scores: Dict[str,float], tracker=None) -> List[DriftAlert]:
        new_alerts = []
        buf = self._buffers.setdefault(agent_name, {})
        for axis, value in scores.items():
            b = buf.setdefault(axis, deque(maxlen=self.baseline_size))
            b.append(value)
            alert = self._check(agent_name, axis, b)
            if alert:
                new_alerts.append(alert); self._alerts.append(alert)
                for cb in self.alert_callbacks:
                    try: cb(alert)
                    except Exception: pass
                if tracker:
                    try: tracker.log_metrics({f"drift_{agent_name}_{axis}": alert.z_score})
                    except Exception: pass
        return new_alerts

    def _check(self, agent, axis, buf: deque) -> Optional[DriftAlert]:
        if len(buf) < self.window_size + 10: return None
        history = list(buf)
        baseline = history[:-self.window_size]; recent = history[-self.window_size:]
        if len(baseline) < 5: return None
        mu = statistics.mean(baseline)
        std = statistics.stdev(baseline) if len(baseline) > 1 else 1e-6
        if std < 1e-6: return None
        z = (statistics.mean(recent) - mu) / std
        if abs(z) > self.z_threshold:
            sev = "CRITICAL" if abs(z) > self.z_threshold*1.5 else "WARNING"
            logger.warning("[DRIFT %s] %s.%s z=%.2f", sev, agent, axis, z)
            return DriftAlert(agent, axis, round(mu,4), round(statistics.mean(recent),4), round(z,3), severity=sev)
        return None

    def get_alerts(self, since=0.0): return [a for a in self._alerts if a.timestamp >= since]
    def summary(self):
        return {"total_alerts": len(self._alerts),
                "agents_monitored": list(self._buffers.keys()),
                "recent_alerts": [{"agent":a.agent_name,"axis":a.axis,"z":a.z_score,"severity":a.severity}
                                   for a in self._alerts[-5:]]}
