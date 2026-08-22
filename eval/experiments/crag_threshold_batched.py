"""Run CRAG t045 in three batches to avoid memory accumulation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.build_gold_set import load_gold  # noqa: E402
from eval.experiments._harness import run_arm  # noqa: E402
from eval.metrics import aggregate  # noqa: E402
from eval.run_eval import split_gold  # noqa: E402

OUT = Path("eval/results/crag_threshold_progress.json")
MODEL = "qwen2.5:3b"
BASE = {
    "mode": "local", "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama", "llm.ollama_model": MODEL,
    "vector_store.persist_dir": ".arag_index_local",
    "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
    "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
}

WATCH = ["e53", "e55", "e74", "m02", "m13"]
KEYS = ["n", "hallucination_rate", "faithfulness", "citation_precision",
        "answer_correctness", "over_abstention_rate", "correct_abstention_rate",
        "adversarial_robustness_rate", "recall_at_1", "latency_p50_ms"]


def _by_split(detail: list[dict]) -> dict:
    gold = load_gold("data/eval/gold_qa.jsonl")
    out = {}
    for name in ("dev", "holdout"):
        ids = {g.id for g in split_gold(gold, name, 0.25, 20260804)}
        rows = [r for r in detail if r["id"] in ids]
        answerable = [r for r in rows if r["difficulty"] in ("easy", "multi_hop")]
        refusal = [r for r in rows if r["difficulty"] in ("unanswerable", "adversarial")]
        out[name] = {
            "n": len(rows),
            "over_abstention": round(
                sum(1 for r in answerable if r["abstained"]) / max(len(answerable), 1), 4
            ),
            "refusal_abstained": round(
                sum(1 for r in refusal if r["abstained"]) / max(len(refusal), 1), 4
            ),
            "hallucinated": round(
                sum(1 for r in rows if r["metrics"].get("hallucination")) / max(len(rows), 1), 4
            ),
        }
    return out


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})

    # Load gold and partition into three batches
    gold = load_gold("data/eval/gold_qa.jsonl")
    all_ids = [g.id for g in gold]

    # Split evenly: 40, 40, 37 (total 117)
    batch_size = 40
    batches = [
        all_ids[0:batch_size],
        all_ids[batch_size:batch_size*2],
        all_ids[batch_size*2:],
    ]

    print(f"Batching {len(all_ids)} examples: {[len(b) for b in batches]}", flush=True)

    all_detail = []
    all_records = []
    total_time = 0

    for batch_num, batch_ids in enumerate(batches, 1):
        print(f"\n>>> Batch {batch_num}/{len(batches)} ({len(batch_ids)} examples)", flush=True)
        t0 = time.time()

        # Run with only_ids to restrict to this batch
        d = run_arm(
            {**BASE, "agent.crag.incorrect_threshold": 0.45, "eval.only_ids": batch_ids},
            tag=f"crag_t045_batch{batch_num}"
        )

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"Batch {batch_num} done in {elapsed/60:.1f} min", flush=True)

        all_detail.extend(d["detail"])
        all_records.extend([{k: v for k, v in r.items() if k in ("id", "difficulty", "metrics", "latency_ms", "cost_usd", "from_cache")} for r in d["detail"]])

    print(f"\n=== Combining {len(all_detail)} records ===", flush=True)

    # Recompute aggregate metrics on combined results
    summary = aggregate(all_records)
    rows = {r["id"]: r for r in all_detail}

    # Write final combined result
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    done["t045_middle"] = {
        "incorrect_threshold": 0.45,
        **{k: summary.get(k) for k in KEYS},
        "splits": _by_split(all_detail),
        "watch": {
            i: {
                "abstained": rows[i]["abstained"],
                "grade": rows[i].get("retrieval_grade"),
                "correct": rows[i]["metrics"].get("answer_correctness"),
            }
            for i in WATCH if i in rows
        },
        "wall_min": round(total_time / 60, 1),
    }
    OUT.write_text(json.dumps(done, indent=2))

    r = done["t045_middle"]
    print(json.dumps({k: v for k, v in r.items() if k != "watch"}, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
