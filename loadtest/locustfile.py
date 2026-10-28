"""Locust load test for POST /query, with questions from the gold set.

Each simulated user sends gold questions back to back with no think time, so
the measured rate is what the server sustains at that concurrency.

    locust -f loadtest/locustfile.py --headless -u 10 -r 10 -t 2m --host http://127.0.0.1:8765

`loadtest/run.py` (`make loadtest`) starts the server and runs the full matrix.
Locust exits 1 when the failure ratio is above `loadtest.max_error_rate`.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, constant, events, task

from arag.common.config import load_config

_cfg = load_config(os.environ.get("ARAG_CONFIG", "config/config.yaml"))
QUESTIONS = [
    json.loads(line)["question"]
    for line in Path(_cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl")).read_text().splitlines()
    if line.strip()
]


class QueryUser(HttpUser):
    wait_time = constant(0)

    @task
    def query(self) -> None:
        with self.client.post(
            "/query", json={"query": random.choice(QUESTIONS)}, name="/query", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "answer" not in resp.json():
                resp.failure("response has no answer field")


@events.quitting.add_listener
def _gate_on_error_rate(environment, **_kw) -> None:
    limit = float(_cfg.get("loadtest.max_error_rate", 0.01))
    total = environment.stats.total
    ratio = total.fail_ratio
    # Locust's --csv files are periodic snapshots; the last one can be up to a
    # second short of the run (measured: 23,009 of 23,893 requests). Write the
    # final totals — the numbers the printed summary shows — next to them.
    prefix = getattr(environment.parsed_options, "csv_prefix", None)
    if prefix:
        Path(f"{prefix}_final.json").write_text(json.dumps({
            "requests": total.num_requests,
            "failures": total.num_failures,
            "rps": total.total_rps,
            **{f"p{p}_ms": total.get_response_time_percentile(p / 100) for p in (50, 95, 99)},
        }))
    environment.process_exit_code = 1 if ratio > limit else 0
    print(f"error rate {ratio:.4%} (limit {limit:.2%}) -> exit {environment.process_exit_code}")
