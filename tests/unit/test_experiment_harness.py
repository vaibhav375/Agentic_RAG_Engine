"""Experiment arms must not share a process.

Three false conclusions on this project traced to two arms in one process, the
worst being a phantom 4x corpus-scale slowdown: the same 40 questions at the same
1.30 mean iterations measured 106.4s p50 as the second arm in a shared process
and 25.7s alone. Quality metrics survive that contamination; timings do not.

So the isolation is a tested property, not a convention someone has to remember.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.experiments._harness import run_arm


def test_run_arm_spawns_a_child_rather_than_evaluating_in_process(monkeypatch, tmp_path):
    """The eval must happen in a child, so its models die with it.

    Asserted on the call `run_arm` makes, because that is the guarantee: if this
    ever became an in-process call again, every timing comparison built on it
    would quietly regain the 4x contamination bug.
    """
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "t_x.json").write_text(json.dumps({"summary": {"n": 0}}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_arm({"eval.subset": 1}, tag="t_x", results_dir=str(tmp_path))

    assert seen["cmd"][0] == sys.executable
    assert seen["cmd"][1:3] == ["-m", "eval.experiments._harness"]
    assert json.loads(seen["cmd"][-1]) == {"eval.subset": 1}


def test_run_arm_applies_overrides_per_arm(tmp_path):
    """Two arms in one call must not bleed config into each other."""
    common = {"eval.subset": 2, "eval.results_dir": str(tmp_path)}
    a = run_arm(
        {**common, "vector_store.persist_dir": str(tmp_path / "a")},
        tag="t_a",
        results_dir=str(tmp_path),
    )
    b = run_arm(
        {**common, "vector_store.persist_dir": str(tmp_path / "b"), "retrieval.use_hybrid": True},
        tag="t_b",
        results_dir=str(tmp_path),
    )
    assert a["config_flags"]["use_hybrid"] is False
    assert b["config_flags"]["use_hybrid"] is True
    assert a["summary"]["n"] == b["summary"]["n"]


def test_every_arm_reads_one_frozen_config(monkeypatch, tmp_path):
    """A config edit landing mid-experiment must not split the arms.

    It happened: `chunk_packing`'s first arm read incorrect_threshold 0.51 and its
    second read 0.25, because a commit landed during the 4.7 hours between them.
    They graded 9 and 3 questions "incorrect" and the comparison was worthless —
    the variable under test was not the only thing that changed.
    """
    import eval.experiments._harness as h

    h._FROZEN.clear()
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd[3])  # the config path handed to the child
        (tmp_path / f"{cmd[4]}.json").write_text(json.dumps({"summary": {"n": 0}}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    h.run_arm({}, tag="a", results_dir=str(tmp_path))
    h.run_arm({}, tag="b", results_dir=str(tmp_path))

    assert seen[0] == seen[1], "arms read different config files"
    assert seen[0] != "config/config.yaml", "child read the live file, not a snapshot"
    assert Path(seen[0]).exists()
    h._FROZEN.clear()


def test_a_failed_arm_raises_instead_of_recording_a_result(tmp_path):
    """A crashed arm must never be silently written down as a data point."""
    with pytest.raises(subprocess.CalledProcessError):
        run_arm(
            {"corpus_dir": str(tmp_path / "does-not-exist")},
            tag="t_boom",
            results_dir=str(tmp_path),
        )


def test_a_starved_machine_is_flagged_before_the_arm_runs(capsys):
    """A timing taken while paged out looks exactly like a real result.

    A full local run drove an 8 GB machine to 12.25 GB of swap with the eval
    process holding 2 MB resident at 10% CPU. It was not hung, and nothing in the
    output said so — which is how several measurements were quietly corrupted.
    """
    from eval.experiments._harness import _warn_if_starved

    _warn_if_starved({"total_gb": 8.0, "free_gb": 0.02, "swap_used_gb": 12.25}, "t")
    err = capsys.readouterr().err
    assert "memory-starved" in err
    assert "Quality metrics survive; latency does not" in err


def test_a_healthy_machine_is_not_flagged(capsys):
    from eval.experiments._harness import _warn_if_starved

    _warn_if_starved({"total_gb": 64.0, "free_gb": 40.0, "swap_used_gb": 0.0}, "t")
    assert capsys.readouterr().err == ""


def test_an_unreadable_platform_does_not_block_a_run(capsys):
    """A missing reading must produce an uninformed run, never no run."""
    from eval.experiments._harness import _warn_if_starved

    _warn_if_starved({}, "t")
    assert capsys.readouterr().err == ""


def test_memory_snapshot_is_shaped_or_empty():
    from eval.experiments._harness import memory_snapshot

    snap = memory_snapshot()
    assert snap == {} or snap["total_gb"] > 0
