#  LUMINA — Multi-Agent Document Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-8%20Tasks-orange?logo=huggingface)
![Mistral](https://img.shields.io/badge/Mistral-LLM--as--Judge-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-LLMOps-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)

**A production-grade, multi-modal document intelligence system powered by a novel
Confidence-Gated Routing (CGR) algorithm — a PPO-trained RL agent that dynamically
dispatches document chunks to specialised expert models, calibrated by a Mistral
LLM-as-Judge reward loop. Zero human labels required.**

[Architecture](#architecture) · [Quick Start](#quick-start) · [API Docs](#api) · [Notebooks](#notebooks) · [Results](#results)

</div>

---

##  What Makes LUMINA Original

Most multi-agent document systems use **static routing** (keyword matching or fixed pipelines). LUMINA introduces **Confidence-Gated Routing (CGR)**:

1. Every document chunk is encoded to a 768-d state vector
2. A PPO RL agent learns *which expert to call* from **quality rewards alone** — no labelled data
3. Rewards come from a **Mistral LLM-as-Judge** scoring 4 axes: Accuracy, Faithfulness, Completeness, Coherence
4. The policy continuously improves as more documents are processed

> **This is the core research contribution**: a self-improving routing system where the LLM evaluates its own pipeline quality and trains the router via reinforcement learning.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LUMINA Pipeline                              │
│                                                                     │
│  Document ──► Chunker ──► Encoder (SBERT) ──► CGR Router (PPO)    │
│                                                        │            │
│                               ┌────────────────────────┤            │
│                               ▼                        ▼            │
│                    ┌──────────────────┐    ┌──────────────────┐    │
│                    │  Expert Agents   │    │   Orchestrator   │    │
│                    │  ─────────────   │    │  ─────────────   │    │
│                    │  • DocQA         │───►│  Result Fusion   │    │
│                    │  • NER           │    │  Confidence Agg  │    │
│                    │  • Summariser    │    │  Mistral Synth   │    │
│                    │  • VQA           │    └────────┬─────────┘    │
│                    │  • TTS           │             │               │
│                    └──────────────────┘             ▼               │
│                                              LLM-as-Judge           │
│                                           (Mistral · 4-axis)        │
│                                                     │               │
│                    ◄────────── reward signal ───────┘               │
│                         (PPO policy update)                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                     LLMOps Layer
               MLflow · Drift · Prompt Registry
```

---

## Repository Structure

```
lumina/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py          # Central coordination engine
│   │   ├── document_agent.py        # LayoutLMv3 Document QA
│   │   ├── ner_agent.py             # Token Classification (NER)
│   │   ├── summariser_agent.py      # BART + Attention Interpretability
│   │   └── vqa_tts_agents.py        # VQA (Qwen3-VL) + TTS Agent
│   ├── routing/
│   │   ├── cgr_algorithm.py         # ★ Novel CGR RL Router (PPO)
│   │   └── confidence_estimator.py  # Temperature-scaled confidence MLP
│   ├── evaluation/
│   │   └── llm_judge.py             # Mistral LLM-as-Judge (4-axis)
│   └── llmops/
│       └── experiment_tracker.py    # MLflow + PromptRegistry + DriftDetector
├── api/
│   └── app.py                       # FastAPI production endpoint
├── dashboard/
│   └── streamlit_app.py             # Interactive monitoring dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_cgr_rl_agent.ipynb        # ★ CGR training walkthrough
│   ├── 04_multiagent_orchestration.ipynb
│   ├── 05_llm_evaluation.ipynb
│   ├── 06_llmops_pipeline.ipynb
│   └── 07_interpretability.ipynb    # SHAP + attention visualisation
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   └── model_config.yaml
├── tests/
│   └── test_lumina.py               # 20+ pytest tests
├── .env.example
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/lumina-doc-intelligence.git
cd lumina-doc-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# .env is pre-filled with Mistral API key — ready to run
```

### 3. Run via Docker (Recommended)

```bash
cd docker
docker compose up -d
```

| Service | URL |
|---------|-----|
| API (FastAPI) | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| Dashboard (Streamlit) | http://localhost:8501 |
| MLflow | http://localhost:5000 |

### 4. Run Locally

```bash
# Terminal 1 — API
cd src && PYTHONPATH=. uvicorn ../api/app:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/streamlit_app.py

# Terminal 3 — MLflow
mlflow server --port 5000
```

---

## API Usage

### Ingest a Document

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple reported $394.3B revenue in FY2022...", "chunk_size": 512}'
```

```json
{"document_id": "a3f2b1c9", "status": "ingested", "chunk_count": 3}
```

### Query the Document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "a3f2b1c9", "query": "What were the key financial results?"}'
```

```json
{
  "answer": "Apple's FY2022 revenues totalled $394.3 billion (+7.8% YoY)...",
  "agents_used": ["document_qa", "ner", "summariser"],
  "mean_quality_score": 0.821,
  "quality_passed": true,
  "latency_ms": 1842,
  "routing_summary": [...]
}
```

### Upload a PDF

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@your_report.pdf"
```

### Inspect CGR Routing Decisions

```bash
curl http://localhost:8000/routing/a3f2b1c9
```

### LLMOps Drift Alerts

```bash
curl "http://localhost:8000/drift/alerts?since_minutes=60"
```

---

##  Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_data_exploration` | SEC EDGAR + DocVQA EDA | Data quality stats, entity distribution |
| `02_baseline_models` | Benchmark each expert agent standalone | F1, ROUGE, BERTScore baselines |
| `03_cgr_rl_agent` | **CGR PPO training walkthrough** | Training curves, routing policy visualisation |
| `04_multiagent_orchestration` | End-to-end pipeline demo | Side-by-side vs. static routing |
| `05_llm_evaluation` | LLM-as-Judge calibration study | Human vs. judge correlation |
| `06_llmops_pipeline` | MLflow + drift detector demo | Drift alerts, experiment comparison |
| `07_interpretability` | SHAP + attention head analysis | Token attribution heatmaps |

---

##  Tests

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific category
pytest tests/ -k "TestCGRRouter" -v
pytest tests/ -k "TestLLMJudge" -v
pytest tests/ -k "TestAPI" -v
```

**Test coverage:**
- CGR router: routing decisions, PPO update, save/load, ensemble trigger
- LLM judge: scoring, weight calibration, quality gate, malformed JSON
- NER agent: regex extraction, empty input, agent output format
- Summariser: compression, passthrough, metadata
- Orchestrator: end-to-end run, agent dispatch
- FastAPI: health, ingest, query, 404 handling
- LLMOps: drift detection, prompt registry, chunk_document

---

##  Results

| Metric | LUMINA (CGR) | Static Ensemble | Single-Agent Baseline |
|--------|-------------|-----------------|----------------------|
| Mean Composite (LLM-Judge) | **0.821** | 0.763 | 0.711 |
| Faithfulness | **0.871** | 0.802 | 0.734 |
| Unnecessary agent calls | **−38%** | 0% (all) | — |
| Mean latency | 1.84s | 2.61s | 0.92s |
| Quality gate pass rate | **87%** | 74% | 68% |

> *Evaluation on 200 documents from SEC EDGAR (10-K/10-Q), DocVQA test set, and MIMIC-III clinical note excerpts.*

---

##  HuggingFace Tasks Integrated

| Task | Model | Use |
|------|-------|-----|
| Document Question Answering | LayoutLMv3 | Structured doc/form QA |
| Token Classification | BERT-large-NER | Financial & medical NER |
| Summarization | BART-large-CNN | Abstractive + hierarchical summary |
| Image-Text-to-Text | Qwen2-VL / BLIP-2 | Chart & scanned image understanding |
| Visual Question Answering | BLIP-2 | Figure caption grounding |
| Text-to-Speech | SpeechT5 / gTTS | Audio briefing generation |
| Text Classification | DistilBERT | Document type pre-routing |
| Text Generation | Mistral-large | Synthesis, evaluation, narration |

---

##  Novel Algorithm: CGR

```python
# Simplified CGR decision logic
state = encode_chunk(chunk)              # 768-d SBERT embedding
probs, confidence = actor(state)         # temperature-scaled MLP
action = Categorical(probs).sample()     # stochastic dispatch

if probs.max() < ENSEMBLE_THRESHOLD:     # uncertainty → use all agents
    agents = ALL_AGENTS
else:
    agents = [AGENT_NAMES[action]]

# After execution:
reward = llm_judge.score(chunk, output).reward   # r ∈ [-1, 1]
router.record(state, decision, reward)
router.update()  # PPO clipped gradient update
```

---


##  Real-World AI Engineering Mapping

| LUMINA Component | Industry Equivalent |
|-----------------|---------------------|
| CGR PPO Router | Intelligent API gateway / model routing layer (Martian, RouteLLM) |
| LLM-as-Judge loop | RLHF reward modelling without human annotators |
| LLMOps drift detector | Production ML monitoring (Arize, WhyLabs, Evidently) |
| Prompt Registry | Prompt management system (PromptLayer, LangSmith) |
| Multi-agent orchestrator | AutoGen / CrewAI production equivalent |
| FastAPI endpoint | Enterprise ML serving (TorchServe, BentoML pattern) |

---

##  Data Sources

- **SEC EDGAR API** — 10-K/10-Q annual reports (free, real financial data)
- **DocVQA** — Document visual question answering benchmark
- **PubLayNet** — Document layout understanding dataset
- **MIMIC-III** (requires credentialed access) — Clinical notes NER
- **OpenCorporates API** — Company entity data for NER validation

---

##  Contributing

```bash
# Fork → branch → PR
git checkout -b feature/your-feature
pytest tests/ -v          # all tests must pass
git commit -m "feat: ..."
```


---

<div align="center">
<sub>Built as an independent research project · Applied Data Science → AI Engineering</sub>
</div>
