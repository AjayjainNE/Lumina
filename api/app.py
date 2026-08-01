"""LUMINA — FastAPI Production Endpoint."""
from __future__ import annotations
import os, sys, uuid, logging, time
from typing import Optional, List, Dict, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
app = FastAPI(title="LUMINA Document Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_SYSTEM = None
_START  = time.time()
_CACHE: Dict[str, Any] = {}
_COUNT  = 0


def _init_system():
    global _SYSTEM
    if _SYSTEM: return _SYSTEM
    from agents.orchestrator import LuminaOrchestrator
    from agents.document_agent import DocumentQAAgent
    from agents.ner_agent import NERAgent
    from agents.summariser_agent import SummariserAgent
    from agents.vqa_tts_agents import VQAAgent, TTSAgent
    from evaluation.llm_judge import LLMJudge
    from llmops.experiment_tracker import ExperimentTracker, DriftDetector, PromptRegistry
    key = os.getenv("MISTRAL_API_KEY", "Insert your own key")
    agents = {"document_qa": DocumentQAAgent(), "ner": NERAgent(),
              "summariser": SummariserAgent(), "vqa": VQAAgent(), "tts": TTSAgent()}
    orch = LuminaOrchestrator(agents=agents, mistral_api_key=key,
                              mlflow_enabled=False, update_policy_every=5)
    _SYSTEM = {"orchestrator": orch, "judge": orch.judge,
               "drift": DriftDetector(window_size=20, baseline_size=100, z_threshold=2.5),
               "tracker": ExperimentTracker(), "registry": PromptRegistry()}
    return _SYSTEM


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    document_id: Optional[str] = None
    chunk_size: int = Field(512, ge=128, le=2048)

class QueryRequest(BaseModel):
    document_id: str
    query: str = Field(..., min_length=3)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0",
            "uptime_seconds": round(time.time()-_START, 1),
            "documents_processed": _COUNT}


@app.post("/ingest")
def ingest(req: IngestRequest):
    global _COUNT
    doc_id = req.document_id or str(uuid.uuid4())[:12]
    from agents.orchestrator import chunk_document
    chunks = chunk_document(req.text, chunk_size=req.chunk_size)
    _CACHE[doc_id] = {"text": req.text, "result": None}
    _COUNT += 1
    return {"document_id": doc_id, "status": "ingested", "chunk_count": len(chunks)}


@app.post("/query")
def query(req: QueryRequest):
    if req.document_id not in _CACHE:
        raise HTTPException(404, f"Document '{req.document_id}' not found. POST to /ingest first.")
    sys = _init_system()
    t0 = time.perf_counter()
    result = sys["orchestrator"].run(_CACHE[req.document_id]["text"],
                                     req.query, req.document_id)
    _CACHE[req.document_id]["result"] = result
    for score in result.judge_scores:
        sys["drift"].update("pipeline", {"composite": score.composite})
    return {"document_id": req.document_id, "query": req.query,
            "answer": result.final_answer,
            "agents_used": list({ao.agent_name for ao in result.agent_outputs}),
            "mean_quality_score": round(result.mean_composite_score, 4),
            "quality_passed": result.quality_passed,
            "latency_ms": round((time.perf_counter()-t0)*1000, 2),
            "judge_summary": sys["judge"].session_summary()}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), chunk_size: int = 512):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(400, "Only .txt and .pdf supported")
    content = await file.read()
    doc_id = str(uuid.uuid4())[:12]
    if file.filename.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            raise HTTPException(500, "pdfplumber not installed")
    else:
        text = content.decode("utf-8", errors="replace")
    _CACHE[doc_id] = {"text": text, "result": None}
    return {"document_id": doc_id, "filename": file.filename, "char_count": len(text)}


@app.get("/routing/{document_id}")
def routing(document_id: str):
    entry = _CACHE.get(document_id, {})
    result = entry.get("result")
    if not result:
        raise HTTPException(404, "Document not processed yet. POST to /query first.")
    rds = result.routing_decisions
    return {"document_id": document_id, "total_chunks": len(rds),
            "ensemble_rate": sum(rd.ensemble for rd in rds)/max(len(rds),1),
            "decisions": [{"chunk": i, "agents": rd.selected_agents,
                           "ensemble": rd.ensemble, "confidence": rd.confidence_scores}
                          for i, rd in enumerate(rds)]}


@app.get("/judge/{document_id}")
def judge_scores(document_id: str):
    entry = _CACHE.get(document_id, {})
    result = entry.get("result")
    if not result: raise HTTPException(404, "Not processed yet.")
    return {"document_id": document_id, "total": len(result.judge_scores),
            "mean_composite": result.mean_composite_score,
            "quality_passed": result.quality_passed,
            "scores": [s.to_dict() for s in result.judge_scores]}


@app.get("/metrics")
def metrics():
    if not _SYSTEM: return {"status": "not initialised"}
    return {"judge_summary": _SYSTEM["judge"].session_summary(),
            "drift_summary": _SYSTEM["drift"].summary(),
            "documents_processed": _COUNT,
            "uptime_seconds": round(time.time()-_START, 1)}


@app.get("/drift/alerts")
def drift_alerts(since_minutes: float = 60.0):
    if not _SYSTEM: return {"alerts": []}
    since = time.time() - since_minutes*60
    return {"alerts": [{"agent": a.agent_name, "axis": a.axis,
                        "z_score": a.z_score, "severity": a.severity}
                       for a in _SYSTEM["drift"].get_alerts(since=since)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
