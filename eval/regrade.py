"""Grade answers semantically, after the run, with a judge bigger than the generator.

`answer_correctness` is token-F1 and it inverts: on 9 hand-verified labels it
ranked them close to backwards, scoring a wrong answer 0.632 and a correct one
0.000. A judge does the job — but only a big enough one. Measured on those labels:

    token-F1     ranks them close to backwards
    qwen2.5:3b   7/9   (never accepts a wrong answer; misses two correct ones)
    qwen2.5:7b   9/9

So the judge wants to be a larger model than the generator, and that is exactly
what cannot be done inline. Running a 7b judge inside a 3b eval leaves two models
resident and makes Ollama swap per record — the thrashing that once produced a
phantom "7b costs 11x more" conclusion, and which corrupts every timing in the
run it contaminates.

Grading after the fact removes the conflict. Generation is finished, its model can
be released, and the judge is the only thing loaded. Latency numbers in the file
were measured before this pass and are untouched by it.

This is the ONLY place `answer_equivalence` is computed. An inline version existed
briefly and was removed: this project has already been bitten by two
implementations of one number drifting apart (see the note on holdout_check.py).

Run:
    python -m eval.regrade eval/results/current.json --judge qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arag.common.config import load_config  # noqa: E402
from arag.providers import make_llm  # noqa: E402
from arag.providers.llm import MockLLM  # noqa: E402

_ANSWERABLE = ("easy", "multi_hop")


def regrade(path: str, cfg, judge_model: str | None = None) -> dict:
    """Add `answer_equivalence` to a finished results file, in place."""
    data = json.loads(Path(path).read_text())
    detail = data.get("detail") or []
    if not detail:
        raise ValueError(f"{path} has no detail records to regrade")

    overrides = {"llm.judge_model": judge_model} if judge_model else {}
    judge = make_llm(cfg.with_overrides(overrides), role="judge")

    # Asking for a real judge and silently getting the mock one is how this pass
    # first reported a confident, entirely fake number: config.yaml ships
    # `mode: mock`, so `make_llm` returned MockLLM and every score came from the
    # lexical fallback's 0.6 cutoff while the summary recorded "qwen2.5:7b".
    # The give-away was e01 — token-F1 0.5946, rejected; e02 "The default type of
    # a path parameter is a string" against gold "A string.", rejected.
    if judge_model and isinstance(judge, MockLLM):
        raise ValueError(
            f"--judge {judge_model} was requested but the config resolves to the mock "
            f"LLM (mode={cfg.get('mode')!r}, llm.provider={cfg.get('llm.provider')!r}). "
            f"Scores would come from a lexical stand-in, not the judge. Re-run with a "
            f"real backend, e.g.:\n"
            f"  ARAG_MODE=local ARAG_LLM__PROVIDER=ollama python -m eval.regrade ..."
        )

    graded = 0
    for rec in detail:
        # Abstentions make no factual claim to compare, and unanswerable questions
        # have no gold answer to compare against — scoring either would invent a
        # number rather than measure one.
        if rec.get("abstained") or rec.get("difficulty") not in _ANSWERABLE:
            rec["metrics"].pop("answer_equivalence", None)
            continue
        rec["metrics"]["answer_equivalence"] = float(
            judge.judge_equivalence(
                rec.get("question", ""), rec.get("gold_answer") or "", rec.get("predicted") or ""
            )
        )
        graded += 1

    scores = [
        r["metrics"]["answer_equivalence"]
        for r in detail
        if "answer_equivalence" in r["metrics"]
    ]
    answered = [r for r in detail if r.get("difficulty") in _ANSWERABLE]
    summary = data.setdefault("summary", {})
    summary["answer_equivalence"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    # Abstentions are excluded above, so record what the number is over — a mean
    # whose denominator moved would look like a quality change.
    summary["answer_equivalence_n"] = len(scores)
    summary["answer_equivalence_judge"] = judge_model or cfg.get("llm.judge_model") or cfg.get(
        "llm.ollama_model"
    )

    Path(path).write_text(json.dumps(data, indent=2))
    return {
        "graded": graded,
        "answered": len(answered),
        "answer_equivalence": summary["answer_equivalence"],
        "answer_correctness": summary.get("answer_correctness"),
        "judge": summary["answer_equivalence_judge"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="path to eval/results/<tag>.json")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--judge", default=None,
                    help="judge model id; larger than the generator is the point")
    args = ap.parse_args()

    out = regrade(args.results, load_config(args.config), args.judge)
    print(json.dumps(out, indent=2))
    if out["answer_correctness"] is not None:
        print(f"\ntoken-F1 {out['answer_correctness']:.4f} vs semantic "
              f"{out['answer_equivalence']:.4f} over {out['graded']} answered questions")
        print("Both are reported. token-F1 is the series every earlier number sits on;")
        print("the semantic column is the one to read when they disagree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
