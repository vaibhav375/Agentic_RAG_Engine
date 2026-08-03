"""qwen2.5:3b vs qwen2.5:7b — same family, so size is the only variable.

The earlier 3B-vs-7B comparison used llama3.2:3b against qwen2.5:7b, which
confounded model family with model size. This isolates size.

Both run in one process with each model warmed before its own run, and results
persist per config so an interruption costs at most the run in flight.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from eval.run_eval import run_eval  # noqa: E402

PROGRESS = Path("eval/results/model_size_progress.json")
BASE = {
    "mode": "local", "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama", "vector_store.persist_dir": ".arag_index_local",
    "eval.subset": 40,
    "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
    "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
}
KEYS = ["hallucination_rate", "hallucination_rate_strict", "faithfulness",
        "citation_precision", "answer_correctness", "over_abstention_rate",
        "correct_abstention_rate", "adversarial_robustness_rate", "latency_p50_ms"]


def main() -> int:
    import httpx

    done = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else []
    completed = {r["tag"] for r in done}
    cfg = load_config("config/config.yaml")
    for model in ["qwen2.5:3b", "qwen2.5:7b"]:
        tag = "size_" + model.replace(":", "").replace(".", "")
        if tag in completed:
            print(f"skip (done): {model}", flush=True)
            continue
        httpx.post("http://localhost:11434/api/chat", timeout=600,
                   json={"model": model, "stream": False,
                         "messages": [{"role": "user", "content": "hi"}]})
        t0 = time.time()
        print(f"\n>>> {model}", flush=True)
        d = run_eval(cfg.with_overrides({**BASE, "llm.ollama_model": model}), tag=tag)
        s = d["summary"]
        abst = sum(1 for r in d["detail"] if r["abstained"])
        answered = len(d["detail"]) - abst
        flagged = sum(1 for r in d["detail"] if r["metrics"].get("hallucination", 0) > 0)
        row = {"model": model, "tag": tag, **{k: s.get(k) for k in KEYS},
               "answered": answered, "flagged": flagged,
               "query_time_min": round(sum(r["latency_ms"] for r in d["detail"]) / 60000, 1),
               "wall_min": round((time.time() - t0) / 60, 1)}
        done.append(row)
        PROGRESS.write_text(json.dumps(done, indent=2))
        print(json.dumps(row, indent=2), flush=True)

    print("\n=== qwen2.5 3b vs 7b ===", flush=True)
    cols = ["hallucination_rate", "faithfulness", "citation_precision",
            "answer_correctness", "correct_abstention_rate", "latency_p50_ms",
            "query_time_min"]
    print(f"{'model':14s} " + " ".join(f"{c[:13]:>14}" for c in cols), flush=True)
    for r in done:
        print(f"{r['model']:14s} " + " ".join(f"{r.get(c, 0):>14.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
