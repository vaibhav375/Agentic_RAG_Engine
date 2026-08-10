"""Run each experiment arm in its own process.

Three wrong conclusions on this project came from running two arms in a single
process, because each `run_eval` builds a fresh embedder, cross-encoder and NLI
model without releasing the previous arm's:

  - "qwen2.5:7b costs 11x more than 3b" — actually Ollama swapping two models
    in and out of one process for 3.7 of the run's 4.6 hours.
  - "retrieval is 7.7x slower at corpus scale" — a stage profile that built both
    component sets before timing either.
  - "corpus scale costs 4x latency" — the same 40 questions, same config and the
    same 1.30 mean iterations measured 106.4s p50 as the second arm in a shared
    process versus 25.7s alone. The real cost was 1.4x.

Every one of those looked like a finding and would have driven real work. The
arms were never the problem; sharing an address space was. So arms don't share
one any more: `run_arm` spawns a child, the child does one eval and exits, and
the OS reclaims everything before the next arm starts.

Quality metrics survive contamination — latency does not. Anything timed has to
run alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def run_arm(
    overrides: dict,
    tag: str,
    config: str = "config/config.yaml",
    results_dir: str = "eval/results",
    timeout: float | None = None,
) -> dict:
    """Evaluate one config in a fresh process; return the run's result dict.

    Raises `subprocess.CalledProcessError` if the child fails, so a crashed arm
    is never silently recorded as a result.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "eval.experiments._harness", config, tag, json.dumps(overrides)],
        cwd=_ROOT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, f"arm {tag}")
    return json.loads((_ROOT / results_dir / f"{tag}.json").read_text())


def _main(argv: list[str]) -> int:
    """Child entrypoint: one config, one eval, then exit so memory is reclaimed."""
    sys.path.insert(0, str(_ROOT))
    from arag.common.config import load_config
    from eval.run_eval import run_eval

    config, tag, overrides = argv[0], argv[1], json.loads(argv[2])
    run_eval(load_config(config).with_overrides(overrides), tag=tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
