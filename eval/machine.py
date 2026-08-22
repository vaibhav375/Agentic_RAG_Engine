"""What the machine could actually give a run.

Lives here rather than in experiments/ because run_eval records it and
experiments/ already imports run_eval — the reverse would be a cycle.
"""

from __future__ import annotations

import re
import sys


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
