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


def test_a_failed_arm_raises_instead_of_recording_a_result(tmp_path):
    """A crashed arm must never be silently written down as a data point."""
    with pytest.raises(subprocess.CalledProcessError):
        run_arm(
            {"corpus_dir": str(tmp_path / "does-not-exist")},
            tag="t_boom",
            results_dir=str(tmp_path),
        )
