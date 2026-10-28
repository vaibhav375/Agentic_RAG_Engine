# Checkpoint

## 2026-10-28 — serving load test

Done:
- `loadtest/locustfile.py`: Locust user sending gold-set questions to `POST /query`
  with no think time. It exits 1 above `loadtest.max_error_rate`.
- `loadtest/run.py` / `make loadtest`: the matrix of 1/10/50 users × Redis cache
  cold/warm, 2 min each, with a fresh server per run. `--smoke` is the 20 s CI
  run.
- `loadtest:` section in `config/config.yaml`. `loadtest` extra (locust, redis)
  in `pyproject.toml`.
- CI job `loadtest-smoke` (Redis service container). It was checked locally from
  a clean worktree and against a dead host (exit 1). It has not run on GitHub
  Actions yet.
- Results at commit `714f0f3`: `loadtest/results/` (raw) and
  `docs/load-test.md` (table, setup, exact command, pasted output). README
  links it.

Findings:
- Every cell had a 0.00% error rate. Throughput levels off at about
  1,250–1,320 req/s from 10 to 50 users on an M1 Air (load generator on the
  same machine).
- Cold ≈ warm: a cold run misses only 117 times (once per gold question), so
  both measure the cached path. These are not RAG-pipeline latency numbers.
- Locust's `--csv` lags the final totals by up to ~1 s. The summary uses its
  final in-process stats (`*_final.json`) instead.

Open:
- To measure the uncached pipeline, run with the cache off (or with unique
  queries), and in `local` mode on a machine that doesn't swap.
- Repeat each cell to get variance. The 50-user warm run was slower than cold
  for no reason the cache explains.
- Run Locust on a separate machine (or `--processes`) to tell whether the
  plateau is the server or the load generator.
