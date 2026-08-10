"""Does the pipeline hold up when the corpus stops being tiny?

The bundled corpus is 10 documents / 47 chunks, where retrieval saturates
(recall@1 ~0.95) and no retrieval component can be told apart from any other.
This adds 51 real FastAPI tutorial pages as distractors — 61 docs, 3173 chunks —
while keeping the same verified gold set, whose answers still live in the
original documents.

Note this is a *hard* distractor set on purpose: FastAPI documents the same
concepts (path parameters, dependencies, middleware) under different names, so
the noise is topically confusable rather than irrelevant.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.experiments._harness import run_arm  # noqa: E402

OUT = Path("eval/results/corpus_scale_progress.json")
MODEL = "qwen2.5:3b"
KEYS = ["n", "recall_at_1", "recall_at_3", "recall_at_k", "mrr", "context_precision",
        "hallucination_rate", "faithfulness", "correct_abstention_rate",
        "adversarial_robustness_rate", "citation_precision", "latency_p50_ms"]
RUNS = [
    ("original (10 docs)", "scale_local_small", "data/corpus", ".arag_index_local"),
    ("scaled (61 docs)", "scale_local_big", "data/corpus_scaled", ".arag_index_local_scaled"),
]


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for label, tag, corpus, persist in RUNS:
        if tag in done:
            print(f"skip (done): {label}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {label}", flush=True)
        # One process per arm. Run together, the second arm inherits the first's
        # resident embedder/reranker/NLI and its latency inflates ~4x — which is
        # exactly how this script first reported a phantom corpus-scale slowdown.
        s = run_arm({
            "mode": "local", "embeddings.provider": "sentence_transformers",
            "llm.provider": "ollama", "llm.ollama_model": MODEL,
            "corpus_dir": corpus, "vector_store.persist_dir": persist,
            "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
            "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
        }, tag=tag)["summary"]
        done[tag] = {"label": label, **{k: s.get(k) for k in KEYS},
                     "wall_min": round((time.time() - t0) / 60, 1)}
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps(done[tag], indent=2), flush=True)

    if len(done) == 2:
        cols = ["recall_at_1", "recall_at_3", "mrr", "context_precision",
                "hallucination_rate", "correct_abstention_rate",
                "adversarial_robustness_rate"]
        print("\n=== real embeddings, corpus scale ===", flush=True)
        print(f"{'corpus':22s} " + " ".join(f"{c[:13]:>14}" for c in cols), flush=True)
        for _, tag, _, _ in RUNS:
            r = done[tag]
            print(f"{r['label']:22s} " + " ".join(f"{r.get(c, 0):>14.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
