#!/usr/bin/env python3
"""
LUMINA — Quick-Start Demo Script
==================================
Runs the full LUMINA pipeline end-to-end in ~30 seconds.
No GPU required. Uses stub agents + Mistral API for judge.

Usage:
    python run_demo.py
    python run_demo.py --company apple
    python run_demo.py --query "What was the revenue?"
    python run_demo.py --no-mistral   # offline mode (simulated judge)

Author: LUMINA Project
"""

import sys, os, argparse, logging, time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ.setdefault('MISTRAL_API_KEY', 'Insert your own API key')

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s | %(name)s | %(message)s'
)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🔬 LUMINA — Document Intelligence Platform          ║
║   Multi-Agent RL Routing + LLM-as-Judge Self-Calibration    ║
╚══════════════════════════════════════════════════════════════╝
"""

SAMPLE_DOCS = {
    'apple': (
        "Apple Inc. reported total net revenues of $394.3 billion for fiscal year 2022, "
        "compared to $365.8 billion in fiscal 2021, representing an increase of 7.8 percent. "
        "Net income was $99.8 billion, up from $94.7 billion. iPhone revenue increased 6.6% "
        "year over year to $205.5 billion, driven by strong demand for the iPhone 14 lineup. "
        "Mac revenue reached $40.2 billion. iPad revenue was $29.3 billion. "
        "Services revenue hit a record $78.1 billion, representing 19.8% of total revenue. "
        "Wearables, Home and Accessories revenue was $41.2 billion. "
        "International sales accounted for 57% of the quarter's revenue. "
        "The Board approved a new $90 billion share repurchase programme and increased "
        "the quarterly dividend by 5 percent to $0.23 per share."
    ),
    'microsoft': (
        "Microsoft reported revenue of $198.3 billion for fiscal year 2022, up 18% year-over-year. "
        "Azure and other cloud services revenue increased 28% in constant currency. "
        "Commercial cloud revenue was $91.2 billion, up 32%. Operating income was $83.4 billion. "
        "Net income was $72.7 billion. Productivity and Business Processes revenue was $63.4 billion. "
        "Intelligent Cloud revenue was $75.3 billion. More Personal Computing revenue was $59.6 billion. "
        "The company returned $12.4 billion to shareholders through dividends and share repurchases."
    ),
}


def run_demo(company: str = 'apple', query: str = None, use_mistral: bool = True):
    print(BANNER)

    from agents.orchestrator import LuminaOrchestrator, BaseAgent, AgentOutput
    from agents.document_agent import DocumentQAAgent
    from agents.ner_agent import NERAgent
    from agents.summariser_agent import SummariserAgent
    from evaluation.llm_judge import LLMJudge

    doc_text = SAMPLE_DOCS.get(company, SAMPLE_DOCS['apple'])
    query = query or "What were the key financial results and growth drivers?"

    print(f"  Document : {company.upper()} Annual Report (excerpt)")
    print(f"  Query    : {query}")
    print(f"  Mode     : {'Live Mistral API' if use_mistral else 'Offline (simulated judge)'}")
    print()

    # ── Agents ───────────────────────────────────────────────
    ner = NERAgent()
    ner._pipeline = 'fallback'
    summariser = SummariserAgent()
    summariser._pipeline = 'fallback'

    agents = {
        'document_qa': DocumentQAAgent(),
        'ner':          ner,
        'summariser':   summariser,
    }

    judge = LLMJudge()

    # ── Orchestrator ─────────────────────────────────────────
    orchestrator = LuminaOrchestrator(
        agents=agents,
        judge=judge,
        mlflow_enabled=False,
        update_policy_every=1,
    )

    MOCK_SCORE = (
        '{"accuracy":0.87,"faithfulness":0.91,"completeness":0.80,"coherence":0.88,'
        '"reasoning":"Output is factually grounded and well-structured."}'
    )

    print("⚡ Running LUMINA pipeline...\n")
    t0 = time.perf_counter()

    # Use mock judge if offline mode requested
    ctx = patch.object(judge, '_call_mistral', return_value=MOCK_SCORE) if not use_mistral else __import__('contextlib').nullcontext()
    synth_ctx = patch('agents.orchestrator._mistral_synthesise',
                      side_effect=lambda outputs, q, k: (
                          "📊 **Synthesised Answer**\n\n" +
                          "\n\n".join(f"• [{ao.agent_name.upper()}] {ao.output}" for ao in outputs if ao.success)
                      )) if not use_mistral else __import__('contextlib').nullcontext()

    with ctx:
        with synth_ctx:
            result = orchestrator.run(
                document_text=doc_text,
                query=query,
                document_id=f"{company}_demo",
                chunk_size=256,
            )

    elapsed = time.perf_counter() - t0

    # ── Output ───────────────────────────────────────────────
    print("═" * 62)
    print("  FINAL ANSWER")
    print("═" * 62)
    print(result.final_answer[:1200])
    print()

    print("═" * 62)
    print("  PIPELINE METRICS")
    print("═" * 62)
    print(f"  Agents dispatched  : {list({ao.agent_name for ao in result.agent_outputs})}")
    print(f"  Chunks processed   : {len(result.routing_decisions)}")
    print(f"  Ensemble rate      : {sum(rd.ensemble for rd in result.routing_decisions)/max(len(result.routing_decisions),1):.1%}")
    print(f"  LLM-Judge score    : {result.mean_composite_score:.3f} / 1.000")
    print(f"  Quality gate       : {'✅ PASS' if result.quality_passed else '❌ FAIL'}")
    print(f"  Total latency      : {elapsed*1000:.0f}ms")
    print()

    if result.routing_decisions:
        rd = result.routing_decisions[0]
        print("  Routing snapshot (chunk 0):")
        for agent, score in sorted(rd.confidence_scores.items(), key=lambda x: -x[1]):
            bar = '█' * int(score * 20)
            marker = ' ← selected' if agent in rd.selected_agents else ''
            print(f"    {agent:15s} {bar:<20s} {score:.3f}{marker}")
    print()

    if result.ppo_metrics:
        print("  PPO Update:")
        for k, v in result.ppo_metrics.items():
            print(f"    {k}: {v:.4f}")
    print()

    judge_summary = judge.session_summary()
    print("  Judge Summary:")
    for axis in ['accuracy', 'faithfulness', 'completeness', 'coherence']:
        mean = judge_summary.get('axis_means', {}).get(axis, 0)
        print(f"    {axis:14s}: {mean:.3f}")
    print(f"    {'pass_rate':14s}: {judge_summary.get('pass_rate', 0):.1%}")
    print()

    print("═" * 62)
    print("  Ready to run: uvicorn api.app:app --port 8000")
    print("  Dashboard:    streamlit run dashboard/streamlit_app.py")
    print("  Notebooks:    jupyter lab notebooks/")
    print("═" * 62)


def main():
    parser = argparse.ArgumentParser(description='LUMINA demo')
    parser.add_argument('--company', default='apple', choices=['apple', 'microsoft'])
    parser.add_argument('--query',   default=None)
    parser.add_argument('--no-mistral', action='store_true', help='Use simulated judge (offline)')
    args = parser.parse_args()

    run_demo(
        company=args.company,
        query=args.query,
        use_mistral=not args.no_mistral,
    )


if __name__ == '__main__':
    main()
