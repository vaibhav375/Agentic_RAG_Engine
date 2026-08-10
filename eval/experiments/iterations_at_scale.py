"""Do self-correction retries earn their cost on a realistic corpus?

Profiling the scaled corpus (3173 chunks) showed per-query latency 20.9s -> 53.9s,
and the cause is the retry rate, not retrieval: iterations went [1,1,1] -> [2,1,3],
which doubles generate and critique and adds reformulate calls. Isolated dense
search, BM25 and reranking are all *no slower* at scale.

On the small corpus, capping the third iteration cost nothing in quality. Retries
could plausibly matter more when retrieval is noisier, so this measures it rather
than assuming the earlier result transfers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.experiments._harness import run_arm  # noqa: E402

OUT = Path("eval/results/iters_scale_progress.json")
MODEL = "qwen2.5:3b"
KEYS = ["hallucination_rate", "faithfulness", "citation_precision", "answer_correctness",
        "over_abstention_rate", "correct_abstention_rate", "adversarial_robustness_rate",
        "recall_at_1", "latency_p50_ms"]


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for max_iter in [2, 1]:
        tag = f"iters{max_iter}_scaled"
        if tag in done:
            print(f"skip (done): max_iterations={max_iter}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> max_iterations={max_iter}", flush=True)
        # One process per arm — this comparison is about cost, and a shared
        # process has produced a 4x phantom on exactly these two configs before.
        d = run_arm({
            "mode": "local", "embeddings.provider": "sentence_transformers",
            "llm.provider": "ollama", "llm.ollama_model": MODEL,
            "corpus_dir": "data/corpus_scaled",
            "vector_store.persist_dir": ".arag_index_local_scaled",
            "eval.subset": 40,
            "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
            "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
            "agent.max_iterations": max_iter,
        }, tag=tag)
        s = d["summary"]
        iters = [r["iterations"] for r in d["detail"]]
        done[tag] = {"max_iterations": max_iter, **{k: s.get(k) for k in KEYS},
                     "total_iterations": sum(iters),
                     "query_time_min": round(sum(r["latency_ms"] for r in d["detail"]) / 60000, 1),
                     "wall_min": round((time.time() - t0) / 60, 1)}
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps(done[tag], indent=2), flush=True)

    if len(done) == 2:
        cols = ["hallucination_rate", "faithfulness", "citation_precision",
                "answer_correctness", "correct_abstention_rate",
                "adversarial_robustness_rate", "total_iterations", "query_time_min"]
        print("\n=== do retries earn their cost at scale? ===", flush=True)
        print(f"{'max_iter':9s} " + " ".join(f"{c[:13]:>14}" for c in cols), flush=True)
        for k in ("iters2_scaled", "iters1_scaled"):
            r = done[k]
            print(f"{r['max_iterations']:<9} " + " ".join(f"{r.get(c, 0):>14.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
