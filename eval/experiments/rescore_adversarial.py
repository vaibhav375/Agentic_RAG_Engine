"""Recompute adversarial robustness from a stored run, without re-running it.

The refutation check only needs the answer text and the planted false claim, so a
completed eval artifact can be rescored offline in seconds instead of spending
another half hour of generation to change how existing answers are graded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from arag.providers.base import split_sentences  # noqa: E402
from arag.providers.nli import make_nli  # noqa: E402
from eval.build_gold_set import load_gold  # noqa: E402


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "full_local_hard"
    path = Path(f"eval/results/{tag}.json")
    if not path.exists():
        print(f"no artifact at {path}")
        return 1
    detail = json.loads(path.read_text())["detail"]
    gold = {g.id: g for g in load_gold("data/eval/gold_qa.jsonl")}
    cfg = load_config("config/config.yaml").with_overrides({"mode": "local"})
    nli = make_nli(cfg)
    thresh = float(cfg.get("agent.nli_contradiction_threshold", 0.5))

    rows, passes = [], 0
    for r in detail:
        if r["difficulty"] != "adversarial":
            continue
        g = gold.get(r["id"])
        false_claim = getattr(g, "must_refute", None)
        if r["abstained"]:
            verdict, contra = "abstained", None
        elif not false_claim:
            verdict, contra = "no must_refute (grounded rule)", None
        else:
            contra = max(
                (nli.entail(s, false_claim).contradiction
                 for s in (split_sentences(r["predicted"]) or [r["predicted"]])),
                default=0.0)
            verdict = "REFUTED" if contra >= thresh else "left standing"
        ok = r["abstained"] or (contra is not None and contra >= thresh) or not false_claim
        passes += bool(ok)
        rows.append((r["id"], verdict, contra, ok, r["question"][:60]))

    print(f"{'id':5s} {'verdict':22s} {'contra':>7}  {'pass':4s} question")
    for rid, verdict, contra, ok, q in rows:
        c = "  -  " if contra is None else f"{contra:.3f}"
        print(f"{rid:5s} {verdict:22s} {c:>7}  {'yes' if ok else 'NO ':4s} {q}")
    print(f"\nrefutation-aware robustness: {passes}/{len(rows)} = {passes/max(len(rows),1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
