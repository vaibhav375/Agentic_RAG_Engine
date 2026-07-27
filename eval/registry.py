"""Lightweight experiment registry.

Every eval run appends a line to `eval/results/history.jsonl` tagged with the git
SHA, a hash of the config flags, a timestamp, and the headline metrics — so any
metric movement is traceable to the exact code + config that produced it (the
minimum viable "experiment tracking" every LLM team needs). `compare_runs`
diffs two runs so you can see what a prompt/model/config change actually did.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

HISTORY = "eval/results/history.jsonl"

_TRACKED = [
    "hallucination_rate",
    "faithfulness",
    "answer_correctness",
    "recall_at_k",
    "mrr",
    "context_precision",
    "correct_abstention_rate",
    "over_abstention_rate",
    "adversarial_robustness_rate",
    "latency_p95_ms",
    "cost_usd_mean",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def config_hash(flags: dict) -> str:
    return hashlib.md5(json.dumps(flags, sort_keys=True).encode()).hexdigest()[:8]


def record_run(tag: str, flags: dict, summary: dict, path: str = HISTORY) -> dict:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "git_sha": _git_sha(),
        "config_hash": config_hash(flags),
        "flags": flags,
        "metrics": {k: summary.get(k) for k in _TRACKED},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_history(path: str = HISTORY) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def compare_runs(a: dict, b: dict) -> dict:
    """b relative to a: positive delta = b higher."""
    out = {}
    for k in _TRACKED:
        va, vb = a["metrics"].get(k), b["metrics"].get(k)
        if va is None or vb is None:
            continue
        out[k] = {"from": va, "to": vb, "delta": round(vb - va, 4)}
    return out


def main() -> int:
    import sys

    hist = load_history()
    if not hist:
        print("No runs recorded yet. Run `make eval` or `make ablation` first.")
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "compare":
        by_tag = {h["tag"]: h for h in hist}
        a, b = by_tag.get(sys.argv[2]), by_tag.get(sys.argv[3])
        if not a or not b:
            print("Unknown tag(s).")
            return 1
        print(json.dumps(compare_runs(a, b), indent=2))
        return 0
    # default: print the last few runs
    for h in hist[-10:]:
        m = h["metrics"]
        print(
            f"{h['ts']}  {h['git_sha']:8s} cfg:{h['config_hash']}  {h['tag']:14s} "
            f"hallu={m.get('hallucination_rate')} faith={m.get('faithfulness')} "
            f"recall@k={m.get('recall_at_k')} advRobust={m.get('adversarial_robustness_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
