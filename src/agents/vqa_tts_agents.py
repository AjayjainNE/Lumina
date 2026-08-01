"""LUMINA — VQA Agent (Image-Text-to-Text) + TTS Agent (Text-to-Speech)."""
from __future__ import annotations
import logging, os, re
from typing import Optional
logger = logging.getLogger(__name__)


class VQAAgent:
    name = "vqa"
    HF_MODEL = "Salesforce/blip2-opt-2.7b"
    DEFAULT_Q = "Describe all data, labels, and values visible in this image."

    def __init__(self, model_name=None, device="cpu", use_light_model=True):
        self.model_name = model_name or self.HF_MODEL
        self.device = device; self._pipe = None

    def _load(self):
        if self._pipe: return
        try:
            from transformers import pipeline
            self._pipe = pipeline("image-text-to-text", model=self.model_name,
                                  device=0 if self.device=="cuda" else -1)
        except Exception as e:
            logger.warning("VQA load failed: %s", e); self._pipe = "unavailable"

    def process(self, chunk: str, document_id="", image=None, question=None):
        from agents.orchestrator import AgentOutput
        if image is None:
            fig_refs = re.findall(r'(?:Figure|Chart|Graph|Table)\s+\d+[^.]*\.', chunk, re.IGNORECASE)
            data_pts = re.findall(r'\d+\.?\d*\s*(?:percent|%|billion|million)', chunk, re.IGNORECASE)
            parts = []
            if fig_refs: parts.append(f"Visual refs: {'; '.join(fig_refs[:3])}")
            if data_pts: parts.append(f"Data points: {', '.join(data_pts[:8])}")
            output = "[VQA text mode] " + " | ".join(parts) if parts else f"[VQA] No image. Context: {chunk[:200]}"
            return AgentOutput(agent_name=self.name, output=output,
                               metadata={"task_type":"vqa","mode":"text_fallback"})
        self._load()
        if self._pipe == "unavailable":
            return AgentOutput(agent_name=self.name, output=f"[VQA] Model unavailable. Context: {chunk[:200]}",
                               metadata={"task_type":"vqa"}, success=False)
        try:
            result = self._pipe(image, question=question or self.DEFAULT_Q)
            answer = result[0].get("generated_text","") if isinstance(result, list) else str(result)
            return AgentOutput(agent_name=self.name, output=answer,
                               metadata={"task_type":"vqa","model":self.model_name,"document_id":document_id})
        except Exception as e:
            logger.error("VQA inference error: %s", e)
            return AgentOutput(agent_name=self.name, output=f"[VQA error] {chunk[:200]}",
                               metadata={"task_type":"vqa","error":str(e)})


class TTSAgent:
    name = "tts"
    MAX_CHARS = 2000

    def __init__(self, output_dir="outputs/audio", lang="en"):
        self.output_dir = output_dir; self.lang = lang
        os.makedirs(output_dir, exist_ok=True)

    def process(self, chunk: str, document_id=""):
        from agents.orchestrator import AgentOutput
        text = chunk[:self.MAX_CHARS]
        path = self._synthesise(text, document_id)
        return AgentOutput(agent_name=self.name, output=f"Audio: {path}",
                           metadata={"task_type":"text_to_speech","audio_path":path,
                                     "char_count":len(text),"document_id":document_id},
                           success=bool(path))

    def _synthesise(self, text: str, doc_id: str) -> str:
        try:
            from gtts import gTTS
            path = os.path.join(self.output_dir, f"{doc_id}_briefing.mp3")
            gTTS(text=text, lang=self.lang, slow=False).save(path)
            return path
        except Exception:
            path = os.path.join(self.output_dir, f"{doc_id}_briefing.txt")
            with open(path, "w") as f: f.write(text)
            return path
