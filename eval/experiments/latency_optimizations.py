"""Measure the two latency optimizations against the shipped config.

Run:
    python -m eval.experiments.latency_optimizations

Configs, one variable each:
  baseline  llm claim extraction, max_iterations 2   (what the full-set run used)
  A         agent.claim_extraction: clause           removes an LLM call per loop
                                                     iteration; the eval metric
                                                     already splits this way
  B         A + agent.max_iterations: 1              retries cost 31% of wall
                                                     clock across the full run
                                                     for mixed benefit

Why it lives here rather than in a scratch directory: each config takes ~50
minutes, and two separate sessions were interrupted mid-run — once losing the
work entirely, once losing the script itself. Results are appended to
`eval/results/opt_progress.json` as each config finishes, and completed configs
are skipped on restart, so an interruption costs at most the config in flight.

All three run in one process with the model pre-warmed: cross-run latency
comparisons on this machine are confounded by Ollama model residency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

PROGRESS = Path("eval/results/opt_progress.json")
MODEL = "qwen2.5:7b"

BASE = {
    "mode": "local",
    "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama",
    "llm.ollama_model": MODEL,
    "vector_store.persist_dir": ".arag_index_local",
    "eval.subset": 40,
    "retrieval.use_hybrid": True,
    "retrieval.use_rerank": True,
    "agent.enabled": True,
    "agent.crag.enabled": True,
    "agent.critic": "nli",
}

RUNS = [
    ("baseline (llm claims, max_iter 2)", "opt_base", {}),
    ("A: clause claim extraction", "opt_clause", {"agent.claim_extraction": "clause"}),
    ("B: clause + max_iterations 1", "opt_clause_it1",
     {"agent.claim_extraction": "clause", "agent.max_iterations": 1}),
]

KEYS = ["hallucination_rate", "hallucination_rate_strict", "faithfulness",
        "citation_precision", "answer_correctness", "over_abstention_rate",
        "correct_abstention_rate", "adversarial_robustness_rate", "latency_p50_ms"]

REPORT = ["hallucination_rate", "faithfulness", "citation_precision",
          "answer_correctness", "correct_abstention_rate", "per_answer_flag",
          "latency_p50_ms", "query_time_min"]


def _warm(model: str) -> None:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": model, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})


def main() -> int:
    _warm(MODEL)
    print(f"warmed {MODEL}", flush=True)

    done = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    completed = {r["tag"] for r in done}
    cfg = load_config("config/config.yaml")

    for label, tag, over in RUNS:
        if tag in completed:
            print(f"skip (already done): {label}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {label}", flush=True)
        d = run_eval(cfg.with_overrides({**BASE, **over}), tag=tag)
        s = d["summary"]
        abstained = sum(1 for r in d["detail"] if r["abstained"])
        answered = len(d["detail"]) - abstained
        flagged = sum(1 for r in d["detail"] if r["metrics"].get("hallucination", 0) > 0)
        row = {
            "label": label, "tag": tag,
            **{k: s.get(k) for k in KEYS},
            "answered": answered, "flagged": flagged,
            "per_answer_flag": round(flagged / max(answered, 1), 3),
            "query_time_min": round(sum(r["latency_ms"] for r in d["detail"]) / 60000, 1),
            "wall_min": round((time.time() - t0) / 60, 1),
        }
        done.append(row)
        PROGRESS.write_text(json.dumps(done, indent=2))  # persist before continuing
        print(json.dumps(row, indent=2), flush=True)

    print("\n=== comparison ===", flush=True)
    print(f"{'config':36s} " + " ".join(f"{h[:13]:>14}" for h in REPORT), flush=True)
    for r in done:
        print(f"{r['label']:36s} " + " ".join(f"{r.get(h, 0):>14.3f}" for h in REPORT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
