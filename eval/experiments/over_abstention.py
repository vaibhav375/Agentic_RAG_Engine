"""Why does the pipeline decline 14 questions it had already retrieved answers for?

over_abstention_rate 0.161 is the worst remaining metric on the definitive local
run, and 13 of its 14 cases had recall_at_3 = 1.0 — retrieval found the answer and
the pipeline still declined. The 14 split into two unrelated causes:

  - 5 abstain on the first pass: CRAG grades retrieval "incorrect" and declines
    before generating anything (graph.py, the pre-generation gate).
  - 9 abstain after exhausting retries: the critic never accepts the answer even
    though its context was correct.

Those need different fixes, so this measures each cause separately rather than
moving one threshold and hoping the aggregate improves.

"Keep the best-supported answer across iterations" is NOT among the arms: the
retry loop exits as soon as `critique.supported`, so an unsupported final critique
means every iteration was unsupported and best-of would change nothing.

The question each arm answers is whether the discarded answer was actually right.
If it was, over-abstention is pure lost coverage and worth spending safety margin
on; if the answers were wrong, the abstention was correct behaviour reported under
a misleading metric name, and the gold set's phrasing is the thing to fix.

Arms (one process each):
  shipped         current config — reproduces the 14
  no_post_abstain support_threshold 0 — the critic can no longer force an
                  abstention, so the 9 produce their answer and it can be judged
  no_crag         answerability gate off — the 5 reach the generator

Safety is expected to degrade on the last two: CRAG is what delivers
correct_abstention 1.000 and adversarial_robustness 1.000. That trade is the
point of measuring. The unanswerable and adversarial slices are included for
exactly that reason — an arm that fixes coverage by breaking refusal is not a fix.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.experiments._harness import run_arm  # noqa: E402

OUT = Path("eval/results/over_abstention_progress.json")
SOURCE = Path("eval/results/full_local_hard.json")
MODEL = "qwen2.5:3b"

BASE = {
    "mode": "local", "embeddings.provider": "sentence_transformers",
    "llm.provider": "ollama", "llm.ollama_model": MODEL,
    "vector_store.persist_dir": ".arag_index_local",
    "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
    "agent.enabled": True, "agent.crag.enabled": True, "agent.critic": "nli",
}
ARMS = [
    ("shipped", {}),
    ("no_post_abstain", {"agent.support_threshold": 0.0}),
    ("no_crag", {"agent.crag.enabled": False}),
]


def _target_ids() -> tuple[list[str], dict[str, str]]:
    """The over-abstained ids from the saved run, plus every refusal case.

    Derived from the run that reported them rather than hardcoded, so the list
    can't drift away from the numbers it came from. The unanswerable and
    adversarial questions come along to keep the safety cost visible.
    """
    d = json.loads(SOURCE.read_text())
    over = [r for r in d["detail"] if r["metrics"].get("over_abstention")]
    cause = {
        # iterations == 1 with an abstention means the pre-generation gate fired;
        # anything higher means the retry loop ran and the critic refused.
        r["id"]: ("crag_pre_generation" if r["iterations"] == 1 else "critic_exhausted")
        for r in over
    }
    refusal = [r["id"] for r in d["detail"] if r["difficulty"] in ("unanswerable", "adversarial")]
    return sorted({*cause, *refusal}), cause


def main() -> int:
    import httpx

    httpx.post("http://localhost:11434/api/chat", timeout=600,
               json={"model": MODEL, "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})

    ids, cause = _target_ids()
    n_crag = sum(1 for v in cause.values() if v == "crag_pre_generation")
    print(f"{len(cause)} over-abstentions ({n_crag} CRAG pre-generation, "
          f"{len(cause) - n_crag} critic-exhausted) + "
          f"{len(ids) - len(cause)} refusal cases = {len(ids)} questions/arm", flush=True)

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, over in ARMS:
        if name in done:
            print(f"skip (done): {name}", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {name}", flush=True)
        d = run_arm({**BASE, **over, "eval.only_ids": ids}, tag=f"oa_{name}")

        rows = {}
        for r in d["detail"]:
            rows[r["id"]] = {
                "cause": cause.get(r["id"], "refusal_case"),
                "difficulty": r["difficulty"],
                "abstained": r["abstained"],
                "grade": r.get("retrieval_grade"),
                "support": r.get("support_fraction"),
                "correct": r["metrics"].get("answer_correctness"),
                "hallucinated": bool(r["metrics"].get("hallucination")),
                "answer": (r["predicted"] or "")[:160],
            }
        recovered = [
            i for i, v in rows.items()
            if v["cause"] != "refusal_case" and not v["abstained"]
        ]
        done[name] = {
            "arm": name,
            # Did the previously-declined questions now answer, and correctly?
            "recovered": len(recovered),
            "recovered_correct": round(
                sum(rows[i]["correct"] or 0 for i in recovered) / max(len(recovered), 1), 3
            ),
            # What that coverage cost on the questions that should be refused.
            "refusal_cases": sum(1 for v in rows.values() if v["cause"] == "refusal_case"),
            "refusals_kept": sum(
                1 for v in rows.values() if v["cause"] == "refusal_case" and v["abstained"]
            ),
            "refusals_hallucinated": sum(
                1 for v in rows.values() if v["cause"] == "refusal_case" and v["hallucinated"]
            ),
            "wall_min": round((time.time() - t0) / 60, 1),
            "rows": rows,
        }
        OUT.write_text(json.dumps(done, indent=2))
        print(json.dumps({k: v for k, v in done[name].items() if k != "rows"}, indent=2),
              flush=True)

    if len(done) == len(ARMS):
        print("\n=== coverage recovered vs refusal kept ===", flush=True)
        hdr = ["recovered", "recovered_correct", "refusals_kept", "refusals_hallucinated"]
        print(f"{'arm':16s} " + " ".join(f"{h[:18]:>21}" for h in hdr), flush=True)
        for name, _ in ARMS:
            r = done[name]
            print(f"{name:16s} " + " ".join(f"{r[h]:>21}" for h in hdr), flush=True)

        print("\n=== per-cause recovery (no_post_abstain / no_crag) ===", flush=True)
        for name in ("no_post_abstain", "no_crag"):
            for cause_name in ("crag_pre_generation", "critic_exhausted"):
                sel = [v for v in done[name]["rows"].values() if v["cause"] == cause_name]
                got = [v for v in sel if not v["abstained"]]
                corr = sum(v["correct"] or 0 for v in got) / max(len(got), 1)
                print(f"  {name:16s} {cause_name:20s} answered {len(got)}/{len(sel)}"
                      f"  mean correctness {corr:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
