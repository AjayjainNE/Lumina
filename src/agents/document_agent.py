"""LUMINA — Document QA Agent: LayoutLMv3 + text-only fallback."""
from __future__ import annotations
import re, logging
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)

class DocumentQAAgent:
    name = "document_qa"
    HF_MODEL = "impira/layoutlm-document-qa"
    CONFIDENCE_THRESHOLD = 0.40

    def __init__(self, model_name=HF_MODEL, device="cpu", max_answer_length=512):
        self.model_name = model_name; self.device = device
        self.max_answer_length = max_answer_length; self._pipeline = None

    def _load(self):
        if self._pipeline: return
        try:
            from transformers import pipeline
            self._pipeline = pipeline("document-question-answering", model=self.model_name,
                                      device=0 if self.device=="cuda" else -1)
        except Exception as e:
            logger.warning("LayoutLMv3 load failed: %s — text fallback", e)
            self._pipeline = "fallback"

    def process(self, chunk: str, document_id="",
                query="What are the key facts, numbers, and entities in this document?", image=None):
        from agents.orchestrator import AgentOutput
        self._load()
        result = (self._layout_qa(chunk, query, image)
                  if self._pipeline != "fallback" and image is not None
                  else self._text_qa(chunk, query))
        return AgentOutput(agent_name=self.name, output=result["answer"],
                           metadata={"task_type":"document_qa","model":self.model_name,
                                     "confidence":result.get("score",0.0),
                                     "query":query,"document_id":document_id},
                           success=bool(result.get("answer")))

    def _layout_qa(self, chunk, query, image) -> Dict:
        try:
            out = self._pipeline(image=image, question=query)
            if isinstance(out, list): out = out[0]
            if out.get("score", 0) < self.CONFIDENCE_THRESHOLD:
                text = self._text_qa(chunk, query)
                return {"answer": f"{out['answer']} [supp: {text['answer']}]", "score": out["score"]}
            return out
        except Exception as e:
            logger.error("LayoutLMv3 error: %s", e); return self._text_qa(chunk, query)

    def _text_qa(self, text: str, query: str) -> Dict:
        nums    = re.findall(r'\$[\d,\.]+\s*(?:billion|million)?|\d+\.?\d*\s*(?:percent|%)', text, re.IGNORECASE)
        entities= re.findall(r'[A-Z][a-z]+ (?:Inc\.|Corp\.|Ltd\.|LLC|Group|Holdings)', text)
        dates   = re.findall(r'(?:fiscal\s+)?(?:year|quarter|Q[1-4])\s+\d{4}', text, re.IGNORECASE)
        facts = []
        if nums:     facts.append(f"Key figures: {', '.join(nums[:5])}")
        if entities: facts.append(f"Entities: {', '.join(set(entities)[:3])}")
        if dates:    facts.append(f"Periods: {', '.join(set(dates)[:3])}")
        query_words = set(query.lower().split())
        sentences   = text.split('. ')
        top = sorted(sentences, key=lambda s: len(set(s.lower().split()) & query_words), reverse=True)
        answer = " | ".join([top[0]] + facts)[:self.max_answer_length]
        return {"answer": answer, "score": 0.65}

    def extract_table_kv(self, text: str) -> List[Dict]:
        pattern = re.compile(r'([A-Za-z][^\t\n:$]{3,50})[:\t]+\s*(\$?[\d,\.]+\s*(?:billion|million|%)?)', re.IGNORECASE)
        return [{"key": m.group(1).strip(), "value": m.group(2).strip()} for m in pattern.finditer(text)][:20]
