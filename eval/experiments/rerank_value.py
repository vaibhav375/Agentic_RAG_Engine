"""Does the cross-encoder reranker earn its place?

It measurably *hurt* ranking in mock (recall@1 0.917 -> 0.833) and has never been
re-measured on real models, or since the citation/prompt/claim-extraction fixes.
It is only ~2% of query latency, so this is purely an accuracy question.

Three configs on the same subset, one variable:
  off      no rerank at all
  replace  reranker's order wins outright  (current default)
  rrf      reranker's order fused with the retrieval order
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

PROGRESS = Path("eval/results/rerank_progress.json")
MODEL = "qwen2.5:3b"          # measured: same quality as 7b, 3x faster
BASE = {
    "mode": "local", "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama", "llm.ollama_model": MODEL,
    "vector_store.persist_dir": ".arag_index_local", "eval.subset": 40,
    "retrieval.use_hybrid": True,
    "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
}
RUNS = [
    ("rerank off", "rr_off", {"retrieval.use_rerank": False}),
    ("rerank replace (default)", "rr_replace",
     {"retrieval.use_rerank": True, "retrieval.rerank_fusion": "replace"}),
    ("rerank rrf", "rr_rrf",
     {"retrieval.use_rerank": True, "retrieval.rerank_fusion": "rrf"}),
]
KEYS = ["recall_at_1", "recall_at_3", "recall_at_k", "mrr", "context_precision",
        "hallucination_rate", "faithfulness", "citation_precision",
        "answer_correctness", "correct_abstention_rate", "latency_p50_ms"]


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    done = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    completed = {r["tag"] for r in done}
    cfg = load_config("config/config.yaml")
    for label, tag, over in RUNS:
        if tag in completed:
            print(f"skip (done): {label}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {label}", flush=True)
        d = run_eval(cfg.with_overrides({**BASE, **over}), tag=tag)
        s = d["summary"]
        row = {"label": label, "tag": tag, **{k: s.get(k) for k in KEYS},
               "query_time_min": round(sum(r["latency_ms"] for r in d["detail"]) / 60000, 1),
               "wall_min": round((time.time() - t0) / 60, 1)}
        done.append(row)
        PROGRESS.write_text(json.dumps(done, indent=2))
        print(json.dumps(row, indent=2), flush=True)

    print("\n=== does the reranker earn its place? ===", flush=True)
    cols = ["recall_at_1", "recall_at_3", "mrr", "context_precision",
            "hallucination_rate", "answer_correctness", "query_time_min"]
    print(f"{'config':26s} " + " ".join(f"{c[:12]:>13}" for c in cols), flush=True)
    for r in done:
        print(f"{r['label']:26s} " + " ".join(f"{r.get(c, 0):>13.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
