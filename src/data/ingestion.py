"""LUMINA — Ingestion Pipeline: multi-format parsing, quality scoring, dedup cache."""
from __future__ import annotations
import hashlib, json, logging, math, os, re, time, unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    doc_id: str; source: str; doc_type: str; text: str
    metadata: Dict = field(default_factory=dict)
    fingerprint: str = ""; word_count: int = 0
    quality_score: float = 0.0; ingested_at: float = field(default_factory=time.time)
    chunks: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(" ".join(self.text.lower().split()).encode()).hexdigest()[:16]
        if not self.word_count:
            self.word_count = len(self.text.split())

    @property
    def is_usable(self): return self.quality_score >= 0.30 and self.word_count >= 50


class DocumentQualityScorer:
    def score(self, text: str) -> Tuple[float, Dict]:
        if not text or len(text.split()) < 5:
            return 0.0, {"reason": "too_short"}
        words = text.split(); n = len(words)
        alnum = sum(c.isalnum() for c in text) / max(len(text), 1)
        sents = [len(s.split()) for s in re.split(r'[.!?]+', text) if s.strip()]
        mean_sent = sum(sents) / max(len(sents), 1)
        sent_score = max(0.0, 1.0 - abs(mean_sent - 22) / 22)
        unique = min(len(set(w.lower() for w in words)) / max(n, 1) * 1.5, 1.0)
        numeric = min(len(re.findall(r'\d+\.?\d*', text)) / max(n, 1) * 5, 1.0)
        chars = {}
        for c in text.lower(): chars[c] = chars.get(c, 0) + 1
        total = sum(chars.values())
        entropy = -sum((v/total)*math.log2(v/total) for v in chars.values() if v > 0)
        entropy_score = min(entropy / math.log2(26), 1.0)
        composite = (0.25*alnum + 0.20*sent_score + 0.25*unique + 0.10*numeric + 0.20*entropy_score)
        bd = {"alnum_ratio": round(alnum,4), "sent_length_score": round(sent_score,4),
              "unique_word_ratio": round(unique,4), "numeric_density": round(numeric,4),
              "entropy_score": round(entropy_score,4), "composite": round(composite,4)}
        return round(composite, 4), bd


class IngestCache:
    def __init__(self, cache_path="data/processed/ingest_cache.json"):
        self.cache_path = cache_path; self._cache: Dict = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f: self._cache = json.load(f)
            except Exception: pass

    def contains(self, fp): return fp in self._cache

    def add(self, rec: DocumentRecord):
        self._cache[rec.fingerprint] = {"doc_id": rec.doc_id, "source": rec.source,
                                        "quality": rec.quality_score}
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        try:
            with open(self.cache_path, "w") as f: json.dump(self._cache, f)
        except Exception: pass

    def size(self): return len(self._cache)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _strip_html(html: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    for ent, rep in [('&nbsp;',' '),('&amp;','&'),('&lt;','<'),('&gt;','>')]:
        text = text.replace(ent, rep)
    return re.sub(r'\s+', ' ', text).strip()


class IngestionPipeline:
    def __init__(self, chunk_size=512, chunk_overlap=64, min_quality=0.25,
                 cache_path="data/processed/ingest_cache.json", use_cache=True):
        self.chunk_size = chunk_size; self.chunk_overlap = chunk_overlap
        self.min_quality = min_quality
        self.scorer = DocumentQualityScorer()
        self.cache = IngestCache(cache_path) if use_cache else None
        from utils.robustness import Sanitiser
        self.san = Sanitiser()
        self._stats = {"total": 0, "cached": 0, "failed": 0, "low_quality": 0}

    def ingest_text(self, text: str, doc_id="", source="raw", metadata=None) -> DocumentRecord:
        return self._build(text, source, "raw",
                           doc_id or hashlib.md5(text[:200].encode()).hexdigest()[:8],
                           metadata or {})

    def ingest_file(self, path: str, doc_id=None) -> DocumentRecord:
        path = str(Path(path).resolve())
        if not os.path.exists(path): raise FileNotFoundError(path)
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            doc_type = "pdf"
        elif ext in (".docx", ".doc"):
            import docx as _docx
            text = "\n".join(p.text for p in _docx.Document(path).paragraphs if p.text.strip())
            doc_type = "docx"
        elif ext in (".html", ".htm"):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = _strip_html(f.read())
            doc_type = "html"
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            doc_type = "txt"
        return self._build(text, path, doc_type, doc_id or Path(path).stem,
                           {"file_size": os.path.getsize(path)})

    def ingest_batch(self, paths: List[str], skip_errors=True) -> Generator:
        for p in paths:
            try: yield self.ingest_file(p)
            except Exception as e:
                self._stats["failed"] += 1
                if skip_errors: logger.error("Batch ingest failed %s: %s", p, e)
                else: raise

    def stats(self): return {**self._stats, "cache_size": self.cache.size() if self.cache else 0}

    def _build(self, text, source, doc_type, doc_id, metadata) -> DocumentRecord:
        self._stats["total"] += 1
        text = _normalise(text)
        try: text = self.san.clean(text, source=source)
        except ValueError as e:
            self._stats["failed"] += 1; raise
        quality, breakdown = self.scorer.score(text)
        if quality < self.min_quality:
            logger.warning("Low quality %s (%.3f)", source, quality)
            self._stats["low_quality"] += 1
        rec = DocumentRecord(doc_id=doc_id, source=source, doc_type=doc_type, text=text,
                             metadata={**metadata, "quality_breakdown": breakdown},
                             quality_score=quality)
        if self.cache:
            if self.cache.contains(rec.fingerprint):
                self._stats["cached"] += 1; return rec
            self.cache.add(rec)
        rec.chunks = self._chunk(text)
        logger.info("Ingested %s: words=%d chunks=%d quality=%.3f", doc_id, rec.word_count, len(rec.chunks), quality)
        return rec

    def _chunk(self, text: str) -> List[str]:
        words = text.split(); chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start:start + self.chunk_size])
            if self.san.is_meaningful(chunk): chunks.append(chunk)
        return chunks
