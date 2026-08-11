"""Does lowering the CRAG incorrect_threshold recover coverage without costing safety?

`incorrect_threshold: 0.51` is commented "tuned on the gold set", and it declines
five in-scope questions before generating — three of them by less than 0.03
(idf 0.497, 0.483, 0.441 against a 0.51 cut). A threshold sitting that close to
real answerable questions is fitted to the set it is scored on.

Chosen on the dev split only, to avoid repeating that: the largest threshold that
falsely declines nothing on dev is 0.267, so 0.25 is the round value just below
it. On holdout that predicts 0/21 answerable declined while still turning away
2/7 unanswerable before generation.

Safety is expected to cost something, and the `no_crag` arm bounds it: removing
the gate entirely kept correct_abstention at 1.000 (the post-generation critic
catches the unanswerable on its own) but moved hallucination 0.000 -> 0.046 and
adversarial robustness 1.000 -> 0.889. At 0.25 the gate still fires on the
lowest-coverage queries, so the cost should land below that bound — which is the
thing to verify rather than assume.

Full gold set, both arms, one process each. Reported overall and split by
dev/holdout, since the point of the change is that the old value was in-sample.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.build_gold_set import load_gold  # noqa: E402
from eval.experiments._harness import run_arm  # noqa: E402
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
ARMS = [("t051_shipped", 0.51), ("t025_candidate", 0.25)]
KEYS = ["n", "hallucination_rate", "faithfulness", "citation_precision",
        "answer_correctness", "over_abstention_rate", "correct_abstention_rate",
        "adversarial_robustness_rate", "recall_at_1", "latency_p50_ms"]
# The five the gate declined before generating, at 0.51.
WATCH = ["e53", "e55", "e74", "m02", "m13"]


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

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for tag, thresh in ARMS:
        if tag in done:
            print(f"skip (done): {tag}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> incorrect_threshold={thresh}", flush=True)
        d = run_arm({**BASE, "agent.crag.incorrect_threshold": thresh}, tag=f"crag_{tag}")
        s, detail = d["summary"], d["detail"]
        rows = {r["id"]: r for r in detail}
        done[tag] = {
            "incorrect_threshold": thresh,
            **{k: s.get(k) for k in KEYS},
            "splits": _by_split(detail),
            # Did the five actually recover, and did they grade differently?
            "watch": {
                i: {
                    "abstained": rows[i]["abstained"],
                    "grade": rows[i].get("retrieval_grade"),
                    "correct": rows[i]["metrics"].get("answer_correctness"),
                }
                for i in WATCH if i in rows
            },
            "wall_min": round((time.time() - t0) / 60, 1),
        }
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps({k: v for k, v in done[tag].items() if k != "watch"}, indent=2),
              flush=True)

    if len(done) == len(ARMS):
        cols = ["over_abstention_rate", "hallucination_rate", "correct_abstention_rate",
                "adversarial_robustness_rate", "faithfulness"]
        print("\n=== coverage gained vs safety spent ===", flush=True)
        print(f"{'threshold':>10} " + " ".join(f"{c[:17]:>19}" for c in cols), flush=True)
        for tag, thresh in ARMS:
            r = done[tag]
            print(f"{thresh:>10.2f} " + " ".join(f"{r[c]:>19.4f}" for c in cols), flush=True)

        print("\n=== the five, at each threshold ===", flush=True)
        for tag, thresh in ARMS:
            w = done[tag]["watch"]
            answered = [i for i, v in w.items() if not v["abstained"]]
            print(f"  {thresh:.2f}: answered {len(answered)}/{len(w)} {sorted(answered)}",
                  flush=True)

        print("\n=== dev vs holdout (over-abstention) ===", flush=True)
        for tag, thresh in ARMS:
            sp = done[tag]["splits"]
            print(f"  {thresh:.2f}: dev {sp['dev']['over_abstention']:.4f}  "
                  f"holdout {sp['holdout']['over_abstention']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
