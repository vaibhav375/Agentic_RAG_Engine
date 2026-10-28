"""Run the load-test matrix: each `loadtest.users` count, Redis cache cold then warm.

    python loadtest/run.py            # full matrix -> loadtest/results/
    python loadtest/run.py --smoke    # one short run; exit code is locust's error-rate gate

Every run gets a fresh server process with the Redis semantic cache switched on,
so `/stats` after the run counts that run's hits and misses alone. Cold deletes
the cache key first; warm keeps what the cold run wrote, which the new process
reloads from Redis. Abstentions are never cached, so warm is not 100% hits.

The cache falls back to in-process memory when Redis is unreachable, which would
make a "Redis" number a lie — so Redis is pinged up front, and a cold run that
leaves no entries in Redis aborts the matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import redis

from arag.common.config import load_config

RESULTS = Path("loadtest/results")
CACHE_KEY = "arag:semcache"  # RedisSemanticCache's list key


def start_server(config: str, port: int, log) -> subprocess.Popen:
    env = {**os.environ, "ARAG_CACHE__ENABLED": "true", "ARAG_CACHE__BACKEND": "redis"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "arag.cli", "serve", "--config", config,
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        if proc.poll() is not None:
            sys.exit(f"server exited with {proc.returncode}; see {log.name}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").json()["status"] == "ok":
                return proc
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"server not healthy after 60s; see {log.name}")


def run_one(config: str, cfg, r: redis.Redis, users: int, duration: str, cache: str, prefix: str):
    port = int(cfg.get("loadtest.port", 8765))
    if cache == "cold":
        r.delete(CACHE_KEY)
    with open(RESULTS / "server.log", "a") as log:
        server = start_server(config, port, log)
        try:
            cmd = [sys.executable, "-m", "locust", "-f", "loadtest/locustfile.py", "--headless",
                   "-u", str(users), "-r", str(users), "-t", duration,
                   "--host", f"http://127.0.0.1:{port}", "--csv", str(RESULTS / prefix), "--only-summary"]
            print("$", " ".join(cmd), flush=True)
            code = subprocess.run(cmd).returncode
            cache_stats = httpx.get(f"http://127.0.0.1:{port}/stats").json()["cache"]
        finally:
            server.terminate()
            server.wait()
    entries = r.llen(CACHE_KEY)
    if cache == "cold" and entries == 0:
        sys.exit("cold run left nothing in Redis: the server is not using the Redis cache")

    final = json.loads((RESULTS / f"{prefix}_final.json").read_text())
    n, fails = final["requests"], final["failures"]
    lookups = cache_stats["hits"] + cache_stats["misses"]
    row = {
        "users": users, "cache": cache, "duration": duration, "requests": n,
        "rps": round(final["rps"], 1),
        "p50_ms": final["p50_ms"], "p95_ms": final["p95_ms"], "p99_ms": final["p99_ms"],
        "error_rate": round(fails / n, 4) if n else None,
        "cache_hit_rate": round(cache_stats["hits"] / lookups, 4) if lookups else None,
        "cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"],
        "redis_entries_after": entries, "locust_exit": code,
    }
    print(row, flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("ARAG_CONFIG", "config/config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = redis.from_url(cfg.get("cache.redis_url", "redis://localhost:6379/0"))
    try:
        r.ping()
    except redis.ConnectionError as exc:
        sys.exit(f"Redis unreachable at {cfg.get('cache.redis_url')}: {exc} (start redis-server first)")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        users, duration = int(cfg.get("loadtest.smoke_users", 10)), str(cfg.get("loadtest.smoke_duration", "20s"))
        return run_one(args.config, cfg, r, users, duration, "cold", "smoke")["locust_exit"]

    duration = str(cfg.get("loadtest.duration", "2m"))
    rows = [
        run_one(args.config, cfg, r, users, duration, cache, f"u{users}_{cache}")
        for users in cfg.get("loadtest.users", [1, 10, 50])
        for cache in ("cold", "warm")
    ]
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n| users | cache | req/s | p50 ms | p95 ms | p99 ms | error rate | cache hit rate |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        print(f"| {x['users']} | {x['cache']} | {x['rps']} | {x['p50_ms']} | {x['p95_ms']} | "
              f"{x['p99_ms']} | {x['error_rate']:.2%} | {x['cache_hit_rate']:.1%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
