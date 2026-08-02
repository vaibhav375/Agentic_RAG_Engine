"""Check that numbers quoted in the docs still match the stored eval artifacts.

Hand-written result tables drift: a table gets relabelled, a row is copied from a
different run, a rerun moves a value. That happened here — a summary table was
labelled as one configuration while three of its rows came from an earlier run,
which overstated the result until an audit caught it.

This re-derives the headline figures from `eval/results/*.json` plus the
experiment registry and compares them to what the docs claim. Run it whenever a
result table changes:

    python -m eval.verify_docs        # exits non-zero on a mismatch
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("eval/results")
DOC = Path("docs/local-mode-eval.md")

# (doc label, artifact, metric key, expected — read from the artifact at check time)
_CHECKS = [
    ("first local baseline", "local_llama32", "hallucination_rate"),
    ("first local baseline", "local_llama32", "over_abstention_rate"),
    ("first local baseline", "local_llama32", "answer_correctness"),
    ("final config", "local_nli_severity", "hallucination_rate"),
    ("final config", "local_nli_severity", "over_abstention_rate"),
    ("final config", "local_nli_severity", "answer_correctness"),
    ("final config", "local_nli_severity", "faithfulness"),
    ("NLI-only critic", "local_nli", "hallucination_rate"),
    ("7B judge", "local_judge7b", "hallucination_rate"),
    ("tightened prompt", "local_nli_tightprompt", "faithfulness"),
]


def _quoted_numbers(text: str) -> list[float]:
    """Every decimal figure appearing in the doc, for numeric comparison."""
    return [float(m) for m in re.findall(r"\d+\.\d+", text)]


def _load(tag: str) -> dict | None:
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def flag_rate(tag: str) -> tuple[int, int, float] | None:
    """Records flagged per record actually answered — the comparison that is
    robust to coverage differences between configurations."""
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    detail = json.loads(p.read_text())["detail"]
    answered = [r for r in detail if not r["abstained"]]
    flagged = [r for r in answered if r["metrics"].get("hallucination", 0) > 0]
    if not answered:
        return None
    return len(flagged), len(answered), round(len(flagged) / len(answered), 3)


def main() -> int:
    if not DOC.exists():
        print(f"{DOC} not found")
        return 1
    text = DOC.read_text()
    problems: list[str] = []
    missing: list[str] = []

    for label, tag, key in _CHECKS:
        summary = _load(tag)
        if summary is None:
            missing.append(f"{tag} (for '{label}')")
            continue
        value = summary.get(key)
        if value is None:
            problems.append(f"{tag}: no {key} recorded")
            continue
        # Compare numerically, not as strings: a value like 0.3125 is a rounding
        # tie that may legitimately appear as 0.312 or 0.313.
        if not any(abs(q - value) <= 0.0006 for q in _quoted_numbers(text)):
            problems.append(
                f"{label} / {tag}.{key} = {value:.4f} is not quoted anywhere in {DOC.name}"
            )

    print("Per-answer flag rates (recomputed from stored records):")
    for tag in ["local_llama32", "local_nli", "local_judge7b",
                "local_nli_oldprompt_newmetric", "local_nli_tightprompt",
                "local_nli_severity"]:
        fr = flag_rate(tag)
        if fr:
            print(f"  {tag:32s} {fr[0]}/{fr[1]} answers = {fr[2]:.3f}")

    if missing:
        print("\nArtifacts absent (tag reuse overwrites them — check the registry):")
        for m in missing:
            print(f"  - {m}")
    if problems:
        print("\nMISMATCHES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll checked figures appear in the doc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
