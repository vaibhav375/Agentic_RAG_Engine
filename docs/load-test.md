# Load test — POST /query

Throughput and latency of the FastAPI `/query` endpoint under Locust, with the
Redis semantic cache on, in `mock` mode (no API keys, no models). One run per
cell, 2 minutes each.

## Results

| users | cache | req/s | p50 ms | p95 ms | p99 ms | error rate | cache hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | cold | 633.8 | 1 | 2 | 3 | 0.00% | 99.9% |
| 1 | warm | 658.1 | 1 | 2 | 2 | 0.00% | 100.0% |
| 10 | cold | 1243.1 | 8 | 10 | 12 | 0.00% | 99.9% |
| 10 | warm | 1244.3 | 8 | 10 | 12 | 0.00% | 100.0% |
| 50 | cold | 1321.4 | 36 | 48 | 57 | 0.00% | 99.9% |
| 50 | warm | 1174.3 | 38 | 61 | 110 | 0.00% | 100.0% |

Raw data: [`loadtest/results/`](../loadtest/results/) — Locust's CSVs per run
(`u{users}_{cold|warm}_*.csv`), its final totals (`*_final.json`), the summary
(`summary.csv`) and the complete console output (`run_output.txt`).

## Read this before quoting the numbers

- **This measures the cached path, not the RAG pipeline.** The gold set has 117
  questions. Each cold run missed exactly 117 times (once per question), and
  every later request for a question it had already seen was a hit. So cold
  runs still hit 99.9%, and cold and warm both measure a cache lookup plus HTTP.
  On top of that, `mock` mode's miss path is itself a few milliseconds of
  deterministic stand-ins. None of this is LLM, embedding or reranker latency.
- **Load generator and server share one fanless laptop.** Locust (one process)
  and uvicorn (one worker) ran on the same M1 MacBook Air. Throughput levels
  off at about 1,250–1,320 req/s from 10 to 50 users while latency grows with
  the queue (Little's law: 50 users / 1321 req/s ≈ 38 ms, against a measured
  mean of 36). That pattern points to saturation, but these runs don't show
  whether the server or Locust was the limit. Locust logged no CPU warning.
- **Single runs, no repeats.** 50 users warm came out slower than 50 users cold
  (1174 vs 1321 req/s, p99 110 vs 57 ms) even though warm had zero misses.
  Nothing in the cache explains that. Treat it as run-to-run variance (thermal
  throttling is plausible on a fanless machine) until repeated runs say
  otherwise.
- **Cache hit rate** is the server's own count (`GET /stats` from a fresh
  process per run). In warm runs the server counts 1–5 more lookups than Locust
  counts requests: those requests were still in flight when Locust stopped
  counting.
- **Numbers come from Locust's final in-process stats, not its CSVs.** The
  `--csv` files are periodic snapshots, and the last one can be up to about a
  second short of the run. In a smoke run the CSV had 23,009 requests, while
  the printed summary and the server both counted 23,893. The runner reads
  `*_final.json`, which matches the printed summary. Locust rounds percentiles
  above 100 ms to two significant figures.
- **"Warm" is warm in Redis, not just in memory.** Each run starts a new server
  process. Cold deletes the `arag:semcache` key first. Warm reloads the entries
  the cold run wrote. Every run left 117 entries in Redis. The runner aborts if
  a cold run leaves none, because the cache silently falls back to in-process
  memory when Redis is down.

## Setup

| | |
|---|---|
| Commit | `714f0f31a2553916592016d9067f33c9db23c144` (clean tree) |
| Date | 2026-10-28 |
| Machine | MacBook Air, Apple M1 (8 cores: 4 performance + 4 efficiency), 8 GB RAM |
| OS | macOS 26.6.2 (25G83) |
| Python | 3.12.6 |
| Locust / FastAPI / uvicorn | 2.43.1 / 0.135.3 / 0.44.0 |
| Redis | 8.2.2, local, `--save '' --appendonly no` |
| Config | `config/config.yaml`, `mode: mock`, plus `ARAG_CACHE__ENABLED=true ARAG_CACHE__BACKEND=redis` (set by the runner) |
| Server | `python -m arag.cli serve --host 127.0.0.1 --port 8765`, one uvicorn worker |
| Load | `loadtest/locustfile.py`: random question from `data/eval/gold_qa.jsonl`, no think time, all users spawned at once |

## Command

```
redis-server --daemonize yes --save '' --appendonly no
python3 loadtest/run.py          # = make loadtest
```

which ran, per cell (exactly as printed):

```
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 1 -r 1 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u1_cold --only-summary
```

The user count (`-u`/`-r`) and `--csv` prefix change per run. Matrix, duration,
port and the error-rate limit are under `loadtest:` in `config/config.yaml`.

## Run output

These are the aggregate lines from `loadtest/results/run_output.txt` (the full
file has the per-endpoint and percentile tables). Filtered with
`grep -E '^\$|Aggregated|error rate|^\{|^\|'`:

```
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 1 -r 1 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u1_cold --only-summary
         Aggregated                                                                     75881     0(0.00%) |      1       1     140      1 |  633.77        0.00
         Aggregated                                                                              1      1      1      2      2      2      2      3      8     31    140  75881
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 1, 'cache': 'cold', 'duration': '2m', 'requests': 75881, 'rps': 633.8, 'p50_ms': 1, 'p95_ms': 2, 'p99_ms': 3, 'error_rate': 0.0, 'cache_hit_rate': 0.9985, 'cache_hits': 75764, 'cache_misses': 117, 'redis_entries_after': 117, 'locust_exit': 0}
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 1 -r 1 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u1_warm --only-summary
         Aggregated                                                                     78726     0(0.00%) |      1       1      61      1 |  658.11        0.00
         Aggregated                                                                              1      1      1      1      2      2      2      2      4     14     61  78726
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 1, 'cache': 'warm', 'duration': '2m', 'requests': 78726, 'rps': 658.1, 'p50_ms': 1, 'p95_ms': 2, 'p99_ms': 2, 'error_rate': 0.0, 'cache_hit_rate': 1.0, 'cache_hits': 78727, 'cache_misses': 0, 'redis_entries_after': 117, 'locust_exit': 0}
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 10 -r 10 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u10_cold --only-summary
         Aggregated                                                                    148849     0(0.00%) |      7       2     118      8 | 1243.11        0.00
         Aggregated                                                                              8      8      8      9      9     10     11     12     24     83    120 148849
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 10, 'cache': 'cold', 'duration': '2m', 'requests': 148849, 'rps': 1243.1, 'p50_ms': 8, 'p95_ms': 10, 'p99_ms': 12, 'error_rate': 0.0, 'cache_hit_rate': 0.9992, 'cache_hits': 148733, 'cache_misses': 117, 'redis_entries_after': 117, 'locust_exit': 0}
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 10 -r 10 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u10_warm --only-summary
         Aggregated                                                                    148998     0(0.00%) |      7       2     136      8 | 1244.30        0.00
         Aggregated                                                                              8      8      8      9      9     10     11     12     24     79    140 148998
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 10, 'cache': 'warm', 'duration': '2m', 'requests': 148998, 'rps': 1244.3, 'p50_ms': 8, 'p95_ms': 10, 'p99_ms': 12, 'error_rate': 0.0, 'cache_hit_rate': 1.0, 'cache_hits': 148999, 'cache_misses': 0, 'redis_entries_after': 117, 'locust_exit': 0}
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 50 -r 50 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u50_cold --only-summary
         Aggregated                                                                    158235     0(0.00%) |     36      11     220     36 | 1321.37        0.00
         Aggregated                                                                             36     38     39     40     43     48     53     57     98    210    220 158235
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 50, 'cache': 'cold', 'duration': '2m', 'requests': 158235, 'rps': 1321.4, 'p50_ms': 36, 'p95_ms': 48, 'p99_ms': 57, 'error_rate': 0.0, 'cache_hit_rate': 0.9993, 'cache_hits': 158119, 'cache_misses': 117, 'redis_entries_after': 117, 'locust_exit': 0}
$ /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m locust -f loadtest/locustfile.py --headless -u 50 -r 50 -t 2m --host http://127.0.0.1:8765 --csv loadtest/results/u50_warm --only-summary
         Aggregated                                                                    140626     0(0.00%) |     41      10     354     38 | 1174.28        0.00
         Aggregated                                                                             38     40     43     45     52     61     83    110    250    340    350 140626
error rate 0.0000% (limit 1.00%) -> exit 0
{'users': 50, 'cache': 'warm', 'duration': '2m', 'requests': 140626, 'rps': 1174.3, 'p50_ms': 38, 'p95_ms': 61, 'p99_ms': 110.0, 'error_rate': 0.0, 'cache_hit_rate': 1.0, 'cache_hits': 140631, 'cache_misses': 0, 'redis_entries_after': 117, 'locust_exit': 0}
| users | cache | req/s | p50 ms | p95 ms | p99 ms | error rate | cache hit rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | cold | 633.8 | 1 | 2 | 3 | 0.00% | 99.9% |
| 1 | warm | 658.1 | 1 | 2 | 2 | 0.00% | 100.0% |
| 10 | cold | 1243.1 | 8 | 10 | 12 | 0.00% | 99.9% |
| 10 | warm | 1244.3 | 8 | 10 | 12 | 0.00% | 100.0% |
| 50 | cold | 1321.4 | 36 | 48 | 57 | 0.00% | 99.9% |
| 50 | warm | 1174.3 | 38 | 61 | 110.0 | 0.00% | 100.0% |
```

## CI

Each push and PR runs the `loadtest-smoke` job in `.github/workflows/ci.yml`:
`python loadtest/run.py --smoke`, which sends 10 users for 20 s at a fresh
server backed by a Redis service container. Locust exits 1, failing the job,
when the error rate is above `loadtest.max_error_rate` (1%). The smoke gates
errors only; it does not gate latency, since shared runners are too noisy.
