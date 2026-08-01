"""LUMINA — Summariser Agent: BART abstractive + extractive fallback + attention attribution."""
from __future__ import annotations
import re, logging
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)

class SummariserAgent:
    name = "summariser"
    HF_MODEL = "facebook/bart-large-cnn"

    def __init__(self, model_name=HF_MODEL, device="cpu", max_len=256, min_len=60):
        self.model_name = model_name; self.device = device
        self.max_len = max_len; self.min_len = min_len; self._pipeline = None

    def _load(self):
        if self._pipeline: return
        try:
            from transformers import pipeline
            self._pipeline = pipeline("summarization", model=self.model_name,
                                      device=0 if self.device=="cuda" else -1)
        except Exception as e:
            logger.warning("BART load failed: %s — extractive fallback", e)
            self._pipeline = "fallback"

    def process(self, chunk: str, document_id=""):
        from agents.orchestrator import AgentOutput
        self._load()
        words = chunk.split()
        if len(words) < 30:
            return AgentOutput(agent_name=self.name, output=chunk,
                               metadata={"task_type":"summarisation","mode":"passthrough",
                                         "compression_ratio":1.0,"input_word_count":len(words),
                                         "summary_word_count":len(words),"attention_top_tokens":{}})
        if self._pipeline and self._pipeline != "fallback":
            summary, attn = self._abstractive(chunk)
            mode = "abstractive"
        else:
            summary = self._extractive(chunk)
            attn = self._pseudo_attention(chunk, summary)
            mode = "extractive"
        return AgentOutput(agent_name=self.name, output=summary,
                           metadata={"task_type":"summarisation","model":self.model_name,
                                     "mode":mode,"attention_top_tokens":attn,
                                     "input_word_count":len(words),
                                     "summary_word_count":len(summary.split()),
                                     "compression_ratio":round(len(words)/max(len(summary.split()),1),2),
                                     "document_id":document_id},
                           success=bool(summary))

    def _abstractive(self, text: str) -> Tuple[str, Dict]:
        try:
            text = " ".join(text.split()[:900])
            result = self._pipeline(text, max_length=self.max_len, min_length=self.min_len,
                                    do_sample=False, num_beams=4)
            summary = result[0]["summary_text"]
            return summary, self._pseudo_attention(text, summary)
        except Exception as e:
            logger.error("BART error: %s", e)
            return self._extractive(text), {}

    def _extractive(self, text: str, top_k=3) -> str:
        stop = {"the","a","an","in","of","to","and","is","was","were","are","for","on","with","at","by","from","that","this","it","be","as","or","its","has","have","had"}
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: return text[:500]
        freq: Dict[str,int] = {}
        for s in sentences:
            for w in s.lower().split():
                w = re.sub(r'[^a-z]','',w)
                if w and w not in stop: freq[w] = freq.get(w,0)+1
        scored = []
        for i, s in enumerate(sentences):
            words = [re.sub(r'[^a-z]','',w.lower()) for w in s.split()]
            score = sum(freq.get(w,0) for w in words if w and w not in stop)
            pos_w = 1.0 / (1.0 + 0.3*i)
            ent_b = sum(1 for w in s.split() if w and w[0].isupper())
            scored.append((score*pos_w + 0.1*ent_b, i, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scored[:top_k], key=lambda x: x[1])
        return " ".join(s[2] for s in top)

    def _pseudo_attention(self, source: str, summary: str) -> Dict[str, float]:
        summary_words = set(summary.lower().split())
        source_words = source.lower().split()
        total = len(source_words) or 1
        weights: Dict[str,float] = {}
        for w in source_words:
            clean = re.sub(r'[^a-z]','',w)
            if clean in summary_words and len(clean)>3:
                weights[clean] = weights.get(clean,0) + 1.0/total
        mx = max(weights.values()) if weights else 1
        return {k: round(v/mx,3) for k,v in sorted(weights.items(), key=lambda x:-x[1])[:15]}
