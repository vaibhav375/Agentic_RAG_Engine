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
    path = cfg.get("eval.calibration_path", "data/eval/judge_calibration.jsonl")
    rows = _load(path)
    gold = [bool(r["human_label"]) for r in rows]

    llm = make_llm(cfg)
    nli = make_nli(cfg)
    nli_thresh = float(cfg.get("agent.nli_entail_threshold", 0.5))

    llm_preds, nli_preds = [], []
    disagreements = []
    for r in rows:
        supported, _, _ = llm.judge_claim(r["claim"], r["context"])
        llm_preds.append(bool(supported))
        entail = nli.entail(r["context"], r["claim"]).entailment
        nli_preds.append(entail >= nli_thresh)
        if bool(supported) != bool(r["human_label"]):
            disagreements.append({"id": r["id"], "claim": r["claim"], "judge": bool(supported), "human": bool(r["human_label"])})

    return {
        "llm_judge": _confusion(llm_preds, gold),
        "nli_judge": _confusion(nli_preds, gold),
        "llm_disagreements": disagreements,
    }


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    report = calibrate(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "llm_disagreements"}, indent=2))
    if report["llm_disagreements"]:
        print("\nLLM-judge disagreements with humans (calibration failures):")
        for d in report["llm_disagreements"]:
            print(f"  {d['id']}: judge={d['judge']} human={d['human']}  {d['claim']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
