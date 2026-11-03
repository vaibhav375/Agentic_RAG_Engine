# Checkpoint

## 2026-11-03 — GraphQL API

Done:
- `src/arag/serve/gql.py`: Strawberry schema mounted at `/graphql`, with
  `ask(question)`, `health` and `documents(limit = 20)`. GraphiQL is on by default.
- Refactor (its own commit, `920fd45`): `run_query()` and `list_documents()` in
  `serve/api.py`. REST `/query`, `/corpus` and `/health` and the GraphQL
  resolvers now share them, so retrieval and generation are not duplicated.
- New core dependency: `strawberry-graphql>=0.339` (0.339.1 tested). It brings in
  graphql-core, cross-web, python-dateutil and packaging. I did not add the
  `[fastapi]` extra because it only adds python-multipart, which file uploads
  need, and uploads are off.
- `tests/integration/test_graphql.py` (8 tests): REST and GraphQL return the
  same answer and citations for the same question; an out-of-scope question with
  agent and CRAG on gives `declined: true`; five kinds of malformed query return
  GraphQL errors (200 or 400), never 500; `health` and `documents`. They run in
  CI's existing `pytest -q` step. I checked them in a clean venv with CI's
  `.[dev,viz]` install, where python-multipart is absent.
- README: "GraphQL API" section with a real mock-mode request and response.
- Full suite: 190 passed (182 before). The eval gate passes. Its small
  differences from `ci_baseline.json` (for example recall@1 0.879 → 0.856) are
  the same at `386b65c`, before this work, so this change did not cause them.

Decisions (mine, not given in the spec):
- `Citation.source` is the `doc_id`, the same id `documents` returns.
  `snippet` is the citation quote.
- `declined` maps to REST `abstained`. `latencyMs` is the trace stage's `ms`.
- Top-level fields are nullable, as in the spec. Inner fields are non-null,
  except `snippet`.
- `ask` has no flags argument, so declines depend on server config: with the
  default config (agent off), nothing is declined.
- `documents(limit: -1)` is rejected with a GraphQL error.

Open:
- `ask` is not cached separately from REST. It uses whatever `cache.enabled`
  says, the same as `/query`.
- The load test covers REST `/query` only. GraphQL has not been load-tested.

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
