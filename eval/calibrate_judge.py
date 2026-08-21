"""Judge calibration.

An LLM-as-judge (or an NLI proxy) is only trustworthy if it agrees with humans.
This measures the critic against a small human-labeled set of (context, claim,
label) triples and reports accuracy, precision/recall/F1, and Cohen's kappa —
before you rely on it to grade the pipeline.

Run: `python -m eval.calibrate_judge [config.yaml]`  (or `arag calibrate-judge`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arag.common.config import load_config  # noqa: E402
from arag.providers import make_llm, make_nli  # noqa: E402
from arag.providers.llm import MockLLM  # noqa: E402


def _load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _confusion(preds: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(preds, gold, strict=True))
    fp = sum(p and not g for p, g in zip(preds, gold, strict=True))
    tn = sum((not p) and (not g) for p, g in zip(preds, gold, strict=True))
    fn = sum((not p) and g for p, g in zip(preds, gold, strict=True))
    n = len(gold)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Cohen's kappa
    po = acc
    p_pred_pos = (tp + fp) / n
    p_gold_pos = (tp + fn) / n
    pe = p_pred_pos * p_gold_pos + (1 - p_pred_pos) * (1 - p_gold_pos)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

    return {
        "n": n,
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "cohens_kappa": round(kappa, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def calibrate(cfg) -> dict:
    """Calibrate every judged task in the set, not just claim support.

    The file carries two tasks now. They are scored separately because they are
    different jobs with different failure modes, and a pooled kappa would hide
    which one is untrustworthy:

    - claim_support     : is this claim entailed by this context? (drives the
                          self-correction loop). LLM judge and NLI both scored.
    - answer_equivalence: does this answer mean the same as the gold answer?
                          (the `answer_equivalence` metric). NLI cannot do this —
                          measured ~0.00 entailment even for correct answers — so
                          only the LLM judge is scored.
    """
    path = cfg.get("eval.calibration_path", "data/eval/judge_calibration.jsonl")
    rows = _load(path)

    llm = make_llm(cfg, role="judge")
    nli = make_nli(cfg)
    nli_thresh = float(cfg.get("agent.nli_entail_threshold", 0.5))

    support = [r for r in rows if r.get("task", "claim_support") == "claim_support"]
    equivalence = [r for r in rows if r.get("task") == "answer_equivalence"]

    report: dict = {}
    disagreements = []

    if support:
        gold = [bool(r["human_label"]) for r in support]
        llm_preds, nli_preds = [], []
        for r in support:
            supported, _, _ = llm.judge_claim(r["claim"], r["context"])
            llm_preds.append(bool(supported))
            nli_preds.append(nli.entail(r["context"], r["claim"]).entailment >= nli_thresh)
            if bool(supported) != bool(r["human_label"]):
                disagreements.append({"task": "claim_support", "id": r["id"],
                                      "item": r["claim"], "judge": bool(supported),
                                      "human": bool(r["human_label"])})
        report["claim_support"] = {
            "llm_judge": _confusion(llm_preds, gold),
            "nli_judge": _confusion(nli_preds, gold),
        }

    if equivalence:
        gold = [bool(r["human_label"]) for r in equivalence]
        preds = []
        for r in equivalence:
            verdict = bool(llm.judge_equivalence(r["question"], r["gold"], r["candidate"]))
            preds.append(verdict)
            if verdict != bool(r["human_label"]):
                disagreements.append({"task": "answer_equivalence", "id": r["id"],
                                      "item": r["candidate"][:90], "judge": verdict,
                                      "human": bool(r["human_label"])})
        # Accepting a wrong answer is the failure that matters here: it inflates a
        # quality metric. Precision on the positive class is that number.
        report["answer_equivalence"] = {"llm_judge": _confusion(preds, gold)}
        # Mock has no judge — `judge_equivalence` falls back to token overlap,
        # which is the very thing this metric exists to replace. Reporting that as
        # a calibration result would read as "the judge is bad" rather than "no
        # judge ran", so it is labelled at the point of output.
        if isinstance(llm, MockLLM):
            report["answer_equivalence"]["NOTE"] = (
                "mock backend: scored by the lexical stand-in, NOT a judge. "
                "Run with ARAG_MODE=local ARAG_LLM__PROVIDER=ollama and "
                "llm.judge_model set for a real calibration."
            )

    report["llm_disagreements"] = disagreements
    return report


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    report = calibrate(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "llm_disagreements"}, indent=2))
    if report["llm_disagreements"]:
        print("\nLLM-judge disagreements with humans (calibration failures):")
        for d in report["llm_disagreements"]:
            print(f"  [{d['task']}] {d['id']}: judge={d['judge']} human={d['human']}  {d['item']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
