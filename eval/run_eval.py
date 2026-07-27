"""Run one pipeline config over the gold set and record metrics.

Rebuilds the index for the given config (chunking/embeddings are ablation
variables that change the index), runs each gold question end-to-end through the
same `answer_query` used online, computes per-record and aggregate metrics, and
writes `eval/results/<tag>.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from arag.common.config import load_config
from arag.engine import answer_query, build_components
from arag.ingest.index import build_index
from eval.build_gold_set import load_gold
from eval.metrics import aggregate, evaluate_record


def run_eval(cfg, tag: str = "current", rebuild: bool = True, comp=None) -> dict:
    gold = load_gold(cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl"))
    subset = cfg.get("eval.subset")
    if subset:
        gold = gold[: int(subset)]

    if comp is None:
        store = build_index(cfg) if rebuild else None
        comp = build_components(cfg, store=store)

    records = []
    detail = []
    for g in gold:
        t0 = time.perf_counter()
        ans = answer_query(comp, g.question)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        m = evaluate_record(comp, g, ans)
        rec = {
            "id": g.id,
            "difficulty": g.difficulty.value,
            "metrics": m,
            "latency_ms": round(latency_ms, 3),
            "cost_usd": ans.cost_usd,
            "from_cache": ans.from_cache,
        }
        records.append(rec)
        detail.append(
            {
                **rec,
                "question": g.question,
                "predicted": ans.answer,
                "gold_answer": g.answer,
                "abstained": ans.abstained,
                "route": ans.route,
                "iterations": ans.iterations,
                "retrieved_doc_ids": [rc.chunk.doc_id for rc in ans.contexts],
            }
        )

    summary = aggregate(records)
    if comp.cache is not None:
        summary["cache_stats"] = comp.cache.stats()

    out = {"tag": tag, "config_flags": _flags(cfg), "summary": summary, "detail": detail}

    results_dir = Path(cfg.get("eval.results_dir", "eval/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))

    # Experiment tracking: append this run to the history log (git sha + config hash).
    try:
        from eval.registry import record_run

        record_run(tag, out["config_flags"], summary)
    except Exception:
        pass
    return out


def _flags(cfg) -> dict:
    return {
        "mode": cfg.get("mode"),
        "chunking": cfg.chunking.as_dict(),
        "use_hybrid": bool(cfg.get("retrieval.use_hybrid", False)),
        "use_rerank": bool(cfg.get("retrieval.use_rerank", False)),
        "agent_enabled": bool(cfg.get("agent.enabled", False)),
        "critic": cfg.get("agent.critic"),
        "router_enabled": bool(cfg.get("agent.router.enabled", False)),
        "cache_enabled": bool(cfg.get("cache.enabled", False)),
    }


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    res = run_eval(cfg, tag="current")
    print(json.dumps(res["summary"], indent=2))
