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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FROZEN: dict[str, str] = {}


def _freeze(config: str) -> str:
    """Snapshot the config once per experiment, so arms cannot read different ones.

    Each arm is a fresh child that loads the config file when it starts, which
    means an edit landing mid-experiment splits the run: a `chunk_packing` run
    had its first arm read `incorrect_threshold: 0.51` and its second read 0.25,
    because a config commit landed during the 4.7 hours between them. The arms
    graded 9 and 3 questions "incorrect" respectively, and the comparison was
    worthless — the variable under test was not the only thing that changed.

    Freezing costs nothing and removes a whole class of silent invalidation:
    every arm of one experiment now reads one immutable snapshot, taken before
    the first arm starts.
    """
    if config not in _FROZEN:
        import yaml

        sys.path.insert(0, str(_ROOT))
        from arag.common.config import load_config

        path = Path(tempfile.mkdtemp(prefix="arag-frozen-")) / "config.yaml"
        path.write_text(yaml.safe_dump(load_config(config).as_dict()))
        _FROZEN[config] = str(path)
        print(f"[harness] config frozen for this experiment: {path}", flush=True)
    return _FROZEN[config]


def memory_snapshot() -> dict:
    """Best-effort view of whether this machine can actually hold the workload.

    A timing taken while the process is paged out looks exactly like a real
    result, which is how several measurements on this project were quietly
    corrupted. One local eval holds bge-small (~130 MB), bge-reranker-base
    (~1.1 GB) and deberta-v3-base (~700 MB) plus the index — roughly 2.5-3 GB —
    while Ollama holds another ~3.1 GB for a 3B model. On the 8 GB machine this
    was developed on that is ~6 GB before the OS, and a full 117-question run
    drove swap to 12.25 GB of 13.3 GB with the eval process holding 2 MB resident
    and burning 10% CPU. It was not hung; it was thrashing.

    Returns {} when the platform is not recognised — a missing reading must not
    stop a run, only an informed one.
    """
    import subprocess

    out: dict = {}
    try:
        if sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True).stdout.strip())
            out["total_gb"] = round(total / 1024**3, 1)
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            m = re.search(r"Pages free:\s+(\d+)", vm)
            if m:
                out["free_gb"] = round(int(m.group(1)) * 4096 / 1024**3, 2)
            swap = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                                  capture_output=True, text=True).stdout
            m = re.search(r"used\s*=\s*([\d.]+)M", swap)
            if m:
                out["swap_used_gb"] = round(float(m.group(1)) / 1024, 2)
        elif sys.platform.startswith("linux"):
            info = {}
            for line in open("/proc/meminfo"):
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0])
            out["total_gb"] = round(info["MemTotal"] / 1024**2, 1)
            out["free_gb"] = round(info.get("MemAvailable", info["MemFree"]) / 1024**2, 2)
            out["swap_used_gb"] = round(
                (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024**2, 2
            )
    except Exception:
        return {}
    return out


def _warn_if_starved(mem: dict, tag: str) -> None:
    """Say so before the arm runs, not after its numbers are already written."""
    if not mem:
        return
    swap = mem.get("swap_used_gb", 0.0)
    free = mem.get("free_gb", 99.0)
    if swap >= 4.0 or free <= 0.5:
        print(
            f"[harness] WARNING before arm {tag}: machine looks memory-starved "
            f"(total {mem.get('total_gb', '?')} GB, free {free} GB, swap used "
            f"{swap} GB). Timings from this arm will measure paging, not the "
            f"pipeline. Quality metrics survive; latency does not.",
            file=sys.stderr, flush=True,
        )


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
    mem = memory_snapshot()
    _warn_if_starved(mem, tag)
    proc = subprocess.run(
        [sys.executable, "-m", "eval.experiments._harness",
         _freeze(config), tag, json.dumps(overrides)],
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
