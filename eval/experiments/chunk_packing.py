"""Does packing short blocks into chunks help on a realistic corpus?

`chunking.pack_blocks` merges consecutive short blocks from the same section up
to `chunk_size`. Without it every paragraph becomes its own chunk however short,
so `chunk_size` is only ever an upper bound. On the hand-written corpus that was
invisible (paragraphs were uniformly ~33 words); on real FastAPI markdown 76% of
blocks are under 20 words, giving 3173 chunks averaging 14 words — 2.7% of the
512-word budget, and 52 near-duplicate candidates per document competing in the
ranking. Packing collapses that to ~545 chunks at ~80 words.

Two effects are plausible and pull in opposite directions, which is why this is
measured rather than assumed:
  - better: a chunk large enough to contain the answer, and far fewer decoys
  - worse: coarser granularity, so a retrieved chunk carries more irrelevant text

The scaled-corpus baseline to beat (`scale_local_big`): context_precision 0.297
and recall_at_1 0.810, against 0.501 / 0.948 on the small corpus. That precision
drop is what packing is meant to address.

Latency is not the target here. `scale_local_big`'s apparent 4x slowdown was an
artifact of sharing a process with the previous arm (see `_harness`); measured
alone, the same 40 questions run 17.8s -> 25.7s p50 going from 47 to 3173 chunks.
Fewer, denser chunks should still help a little by cutting rerank candidates.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.experiments._harness import run_arm  # noqa: E402

OUT = Path("eval/results/chunk_packing_progress.json")
MODEL = "qwen2.5:3b"
KEYS = [
    "hallucination_rate", "faithfulness", "citation_precision", "answer_correctness",
    "context_precision", "context_recall", "over_abstention_rate",
    "correct_abstention_rate", "adversarial_robustness_rate",
    "recall_at_1", "recall_at_3", "mrr", "latency_p50_ms",
]
ARMS = [("packed", True), ("unpacked", False)]


def main() -> int:
    import httpx

    httpx.post(
        "http://localhost:11434/api/chat", timeout=600,
        json={"model": MODEL, "stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    done = json.loads(OUT.read_text()) if OUT.exists() else {}

    for name, pack in ARMS:
        if name in done:
            print(f"skip (done): {name}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {name} (pack_blocks={pack})", flush=True)
        d = run_arm({
            "mode": "local", "embeddings.provider": "sentence_transformers",
            "llm.provider": "ollama", "llm.ollama_model": MODEL,
            "corpus_dir": "data/corpus_scaled",
            # Separate index per arm: different chunking means a different index,
            # and sharing a persist_dir would silently reuse the other arm's.
            "vector_store.persist_dir": f".arag_index_pack_{name}",
            "chunking.pack_blocks": pack,
            "eval.subset": 40,
            "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
            "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
        }, tag=f"pack_{name}")
        s = d["summary"]
        det = d["detail"]
        done[name] = {
            "pack_blocks": pack,
            **{k: s.get(k) for k in KEYS},
            "n_chunks": s.get("n_chunks"),
            "total_iterations": sum(r["iterations"] for r in det),
            "abstained": sum(1 for r in det if r["abstained"]),
            # Which gate produced each abstention, now that the harness records it.
            "crag_incorrect": sum(1 for r in det if r.get("retrieval_grade") == "incorrect"),
            "query_time_min": round(sum(r["latency_ms"] for r in det) / 60000, 1),
            "wall_min": round((time.time() - t0) / 60, 1),
        }
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps(done[name], indent=2), flush=True)

    if len(done) == len(ARMS):
        cols = ["context_precision", "recall_at_1", "recall_at_3", "answer_correctness",
                "hallucination_rate", "over_abstention_rate", "total_iterations",
                "query_time_min"]
        print("\n=== does packing help at scale? ===", flush=True)
        print(f"{'arm':10s} " + " ".join(f"{c[:13]:>14}" for c in cols), flush=True)
        for name, _ in ARMS:
            r = done[name]
            print(f"{name:10s} " + " ".join(f"{r.get(c) or 0:>14.3f}" for c in cols), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
