"""LUMINA — Robustness Utilities: retry, circuit-breaker, rate-limiter, safe-json, timer, sanitiser."""
from __future__ import annotations
import json, logging, math, random, re, threading, time, unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Retry ─────────────────────────────────────────────────────────────────

class RetryError(Exception): pass

@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)

def with_retry(config: Optional[RetryConfig] = None):
    cfg = config or RetryConfig()
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, cfg.max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except cfg.retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == cfg.max_attempts: break
                    delay = min(cfg.base_delay * (cfg.backoff_factor ** (attempt-1)), cfg.max_delay)
                    if cfg.jitter: delay *= (0.75 + random.random() * 0.5)
                    logger.warning("[retry] %s attempt %d/%d: %s — %.2fs", fn.__name__, attempt, cfg.max_attempts, exc, delay)
                    time.sleep(delay)
            raise RetryError(f"{fn.__name__} failed after {cfg.max_attempts} attempts") from last_exc
        return wrapper
    return decorator


# ── Circuit Breaker ───────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = auto(); OPEN = auto(); HALF_OPEN = auto()

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    cooldown_seconds: float = 30.0
    name: str = "default"

class CircuitBreaker:
    class CircuitOpenError(Exception): pass

    def __init__(self, config: CircuitBreakerConfig):
        self.cfg = config
        self._state = CircuitState.CLOSED
        self._failures = 0; self._successes = 0
        self._last_fail: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_fail and time.time() - self._last_fail >= self.cfg.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def __enter__(self):
        if self.state == CircuitState.OPEN:
            raise self.CircuitOpenError(f"Circuit '{self.cfg.name}' OPEN")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._lock:
            if exc_type is None: self._on_success()
            else: self._on_failure()
        return False

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.cfg.success_threshold:
                self._state = CircuitState.CLOSED; self._failures = 0; self._successes = 0
        else:
            self._failures = 0

    def _on_failure(self):
        self._failures += 1; self._last_fail = time.time()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN; self._successes = 0
        elif self._failures >= self.cfg.failure_threshold:
            logger.error("[circuit:%s] OPEN after %d failures", self.cfg.name, self._failures)
            self._state = CircuitState.OPEN

    def status(self): return {"name": self.cfg.name, "state": self.state.name, "failures": self._failures}


# ── Rate Limiter ──────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rate=1.0, burst=1):
        self.rate = rate; self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout=None):
        deadline = time.monotonic() + (timeout or math.inf)
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0; return True
                wait = (1.0 - self._tokens) / self.rate
            if time.monotonic() + wait > deadline:
                raise TimeoutError(f"RateLimiter timeout after {timeout}s")
            time.sleep(min(wait, 0.05))


# ── SafeJSON ──────────────────────────────────────────────────────────────

class SafeJSON:
    @staticmethod
    def parse(raw: str, default=None):
        if not raw: return default
        s = raw.strip()
        for fence in ["```json", "```JSON", "```"]:
            if s.startswith(fence):
                s = s[len(fence):]; break
        s = s.rstrip("`").strip()
        for attempt in [s, s.replace("'", '"'), re.sub(r',\s*([}\]])', r'\1', s)]:
            try: return json.loads(attempt)
            except json.JSONDecodeError: pass
        m = re.search(r'\{[^{}]+\}', s, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except json.JSONDecodeError: pass
        return default

    @staticmethod
    def extract_field(raw: str, field: str, default=None):
        parsed = SafeJSON.parse(raw)
        if parsed: return parsed.get(field, default)
        m = re.search(rf'"{re.escape(field)}"\s*:\s*(["\d\.\-truefalsnil][^,}}\n]*)', raw)
        if m:
            val = m.group(1).strip().strip('"')
            try: return float(val)
            except ValueError: return val
        return default


# ── Timer ─────────────────────────────────────────────────────────────────

@dataclass
class TimerResult:
    name: str; elapsed_ms: float; success: bool = True; error: Optional[str] = None

class Timer:
    def __init__(self, name: str, log=True):
        self.name = name; self.log = log
        self.result: Optional[TimerResult] = None
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter(); return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self._start) * 1000
        self.result = TimerResult(self.name, round(elapsed, 2),
                                  exc_type is None, str(exc_val) if exc_val else None)
        if self.log:
            logger.log(logging.DEBUG if self.result.success else logging.WARNING,
                       "[timer] %s: %.1fms %s", self.name, elapsed, "✓" if self.result.success else "✗")
        return False


# ── Sanitiser ─────────────────────────────────────────────────────────────

class Sanitiser:
    MAX_CHUNK_CHARS = 8192
    MIN_CHUNK_CHARS = 10
    _INJECTION_RE = re.compile(
        r"ignore previous instructions|you are now|system prompt|jailbreak|<\|im_start\|>|<\|system\|>",
        re.IGNORECASE)

    def clean(self, text: str, source="unknown") -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < self.MIN_CHUNK_CHARS:
            raise ValueError(f"[{source}] too short ({len(text)} chars)")
        if len(text) > self.MAX_CHUNK_CHARS:
            logger.warning("[sanitiser:%s] truncating %d→%d", source, len(text), self.MAX_CHUNK_CHARS)
            text = text[:self.MAX_CHUNK_CHARS]
        if self._INJECTION_RE.search(text):
            logger.warning("[sanitiser:%s] injection detected — scrubbing", source)
            text = self._INJECTION_RE.sub("[REDACTED]", text)
        return text

    def validate_query(self, q: str) -> str:
        return self.clean(q, "query")[:1024]

    def is_meaningful(self, text: str, min_words=5) -> bool:
        return len([w for w in text.split() if len(w) > 1]) >= min_words


# ── HealthCheck ───────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    name: str; healthy: bool; latency_ms: float; detail: str = ""

class HealthCheck:
    @staticmethod
    def check_torch() -> HealthStatus:
        t0 = time.perf_counter()
        try:
            import torch
            torch.randn(4, 4).matmul(torch.randn(4, 4))
            return HealthStatus("pytorch", True, round((time.perf_counter()-t0)*1000, 1),
                                f"torch {torch.__version__}")
        except Exception as e:
            return HealthStatus("pytorch", False, round((time.perf_counter()-t0)*1000, 1), str(e)[:80])

    @staticmethod
    def check_mistral(api_key: str, timeout=5.0) -> HealthStatus:
        import httpx; t0 = time.perf_counter()
        try:
            r = httpx.post("https://api.mistral.ai/v1/chat/completions",
                           headers={"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"},
                           json={"model":"mistral-small-latest","max_tokens":1,"messages":[{"role":"user","content":"ping"}]},
                           timeout=timeout)
            return HealthStatus("mistral", r.status_code in (200,201), round((time.perf_counter()-t0)*1000,1), f"HTTP {r.status_code}")
        except Exception as e:
            return HealthStatus("mistral", False, round((time.perf_counter()-t0)*1000,1), str(e)[:80])
