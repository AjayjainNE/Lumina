"""LUMINA — NER Agent: Token Classification with financial entity specialisation."""
from __future__ import annotations
import re, logging
from collections import defaultdict
from typing import Dict, List
logger = logging.getLogger(__name__)

ENTITY_GROUPS = {"ORG":"Organisation","PER":"Person","LOC":"Location",
                 "MISC":"Miscellaneous","MONEY":"Financial Value","DATE":"Date/Time","PERC":"Percentage"}
REGEX_PATTERNS = {
    "MONEY": r'\$[\d,\.]+\s*(?:billion|million|thousand|B|M|K)?',
    "PERC":  r'\d+\.?\d*\s*(?:percent|%)',
    "DATE":  r'(?:Q[1-4]\s+)?\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
    "ORG":   r'[A-Z][a-zA-Z]+(?:\s+(?:Inc\.|Corp\.|Ltd\.|LLC|Group|Holdings|Company|Co\.))',
}

class NERAgent:
    name = "ner"
    HF_MODEL = "dslim/bert-large-NER"

    def __init__(self, model_name=HF_MODEL, device="cpu"):
        self.model_name = model_name; self.device = device; self._pipeline = None

    def _load(self):
        if self._pipeline: return
        try:
            from transformers import pipeline
            self._pipeline = pipeline("token-classification", model=self.model_name,
                                      aggregation_strategy="simple",
                                      device=0 if self.device=="cuda" else -1)
        except Exception as e:
            logger.warning("NER model load failed: %s — regex fallback", e)
            self._pipeline = "fallback"

    def process(self, chunk: str, document_id=""):
        from agents.orchestrator import AgentOutput
        self._load()
        entities = self.extract(chunk)
        formatted = "\n".join(f"{ENTITY_GROUPS.get(k,k)}: {', '.join(v[:8])}"
                              for k, v in entities.items()) or "No entities detected."
        return AgentOutput(agent_name=self.name, output=formatted,
                           metadata={"task_type":"token_classification","model":self.model_name,
                                     "entity_counts":{k:len(v) for k,v in entities.items()},
                                     "document_id":document_id},
                           success=bool(entities))

    def extract(self, text: str) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        if self._pipeline and self._pipeline != "fallback":
            try:
                for ent in self._pipeline(text):
                    et = ent.get("entity_group","MISC"); w = ent.get("word","").strip()
                    if w and ent.get("score",0) > 0.70 and w not in grouped[et]:
                        grouped[et].append(w)
            except Exception as e:
                logger.error("NER inference error: %s", e)
        for etype, pattern in REGEX_PATTERNS.items():
            for m in re.findall(pattern, text, re.IGNORECASE):
                m = m.strip()
                if m and m not in grouped[etype]: grouped[etype].append(m)
        return dict(grouped)
