"""Run one pipeline config over the gold set and record metrics.

Rebuilds the index for the given config (chunking/embeddings are ablation
variables that change the index), runs each gold question end-to-end through the
same `answer_query` used online, computes per-record and aggregate metrics, and
writes `eval/results/<tag>.json`.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from arag.common.config import load_config
from arag.engine import answer_query, build_components
from arag.ingest.index import build_index
from eval.build_gold_set import load_gold
from eval.metrics import aggregate, evaluate_record


def stratified_subset(gold: list, n: int) -> list:
    """First `n` questions, but proportionally across difficulty slices.

    The gold file is grouped by difficulty (all 40 easy questions first), so a
    plain `gold[:n]` subset is 100% easy — it reports `hallucination_rate` and
    `correct_abstention_rate` without a single unanswerable or adversarial
    question behind them. Any subset run has to keep every slice represented or
    the numbers aren't comparable to a full run.

    Allocation is largest-remainder proportional with at least one question per
    slice, and deterministic: it takes each slice's questions in file order.
    """
    if n >= len(gold) or n <= 0:
        return gold
    by_slice: dict[str, list] = {}
    for g in gold:
        by_slice.setdefault(g.difficulty.value, []).append(g)

    # Proportional share, floored, then hand out the remaining seats to the
    # largest fractional remainders (largest-remainder method).
    exact = {k: len(v) * n / len(gold) for k, v in by_slice.items()}
    take = {k: max(1, int(v)) for k, v in exact.items()}
    while sum(take.values()) > n:  # over-allocated by the min-1 guarantee
        k = max(take, key=lambda k: (take[k] - exact[k], k))
        if take[k] > 1:
            take[k] -= 1
        else:
            break
    for k in sorted(by_slice, key=lambda k: (-(exact[k] - int(exact[k])), k)):
        if sum(take.values()) >= n:
            break
        if take[k] < len(by_slice[k]):
            take[k] += 1

    chosen = [g for k, items in by_slice.items() for g in items[: take[k]]]
    order = {id(g): i for i, g in enumerate(gold)}
    return sorted(chosen, key=lambda g: order[id(g)])


def split_gold(gold: list, split: str, holdout_fraction: float, seed: int) -> list:
    """Partition the gold set into `dev` and `holdout`, deterministically.

    Thresholds, prompts and the CRAG gate on this project were all tuned against
    the whole gold set, which means the reported numbers are in-sample and
    optimistic by an unknown amount. A held-out slice that nothing is tuned
    against is the only way to find out how much.

    Stratified per difficulty so both halves keep every slice, and seeded so the
    partition is stable across runs — a split that moved between runs would make
    every comparison meaningless.
    """
    if split == "all":
        return gold
    by_slice: dict[str, list] = {}
    for g in gold:
        by_slice.setdefault(g.difficulty.value, []).append(g)

    holdout_ids: set[str] = set()
    for name in sorted(by_slice):
        items = sorted(by_slice[name], key=lambda g: g.id)
        rng = random.Random(f"{seed}:{name}")          # per-slice, so adding a
        rng.shuffle(items)                             # slice can't reshuffle others
        n_hold = int(round(len(items) * holdout_fraction))
        holdout_ids.update(g.id for g in items[:n_hold])

    keep_holdout = split == "holdout"
    return [g for g in gold if (g.id in holdout_ids) == keep_holdout]


def run_eval(cfg, tag: str = "current", rebuild: bool = True, comp=None) -> dict:
    gold = load_gold(cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl"))
    gold = split_gold(
        gold,
        str(cfg.get("eval.split", "all")),
        float(cfg.get("eval.holdout_fraction", 0.25)),
        int(cfg.get("eval.split_seed", 20260804)),
    )
    # Re-run a named handful — the cases a previous run got wrong — without
    # paying for the whole gold set to reach them.
    only_ids = cfg.get("eval.only_ids")
    if only_ids:
        keep = set(only_ids)
        gold = [g for g in gold if g.id in keep]
        missing = keep - {g.id for g in gold}
        if missing:
            # Silently evaluating fewer cases than asked for would read as a
            # result rather than a typo.
            raise ValueError(
                f"eval.only_ids not found in gold set (or excluded by the "
                f"active split): {sorted(missing)}"
            )

    subset = cfg.get("eval.subset")
    if subset:
        gold = stratified_subset(gold, int(subset))

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
                # The two signals that decide an abstention, so a bad abstention
                # can be diagnosed from a saved run instead of re-run to find out
                # which gate fired: CRAG "incorrect" declines before generating,
                # an unsatisfied critic declines after the retry loop.
                "retrieval_grade": ans.retrieval_grade,
                "retrieval_grade_score": ans.retrieval_grade_score,
                "support_fraction": (
                    ans.critique.support_fraction if ans.critique is not None else None
                ),
                "retrieved_doc_ids": [rc.chunk.doc_id for rc in ans.contexts],
                # citation_precision is 0.0 both when every cited doc is wrong and
                # when the answer parsed no citations at all — two very different
                # failures. Recording the citations is what tells them apart.
                "citations": [
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id} for c in ans.citations
                ],
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


_RECORD_KEYS = ("id", "difficulty", "metrics", "latency_ms", "cost_usd", "from_cache")


def run_eval_split_report(cfg, tag: str = "current", rebuild: bool = True) -> dict:
    """Evaluate `dev` and `holdout` separately, and report both plus the pool.

    The in-sample number alone is the one that flatters the pipeline, so the
    default path reports the out-of-sample number next to it rather than making
    it something you have to remember to ask for.

    This costs no more than a single `split: all` run: dev and holdout partition
    the gold set, so the two runs together ask exactly the same questions once.
    The index is built once and both runs share the components.

    `<tag>.json` keeps the pooled summary so the CI gate and PR report are
    unaffected; per-split summaries are added under `splits`.
    """
    if str(cfg.get("eval.split", "all")) != "all":
        return run_eval(cfg, tag=tag, rebuild=rebuild)  # caller pinned one split

    store = build_index(cfg) if rebuild else None
    comp = build_components(cfg, store=store)

    runs = {
        name: run_eval(
            cfg.with_overrides({"eval.split": name}), tag=f"{tag}_{name}", comp=comp
        )
        for name in ("dev", "holdout")
    }

    detail = [d for r in runs.values() for d in r["detail"]]
    pooled = aggregate([{k: d[k] for k in _RECORD_KEYS} for d in detail])
    if comp.cache is not None:
        pooled["cache_stats"] = comp.cache.stats()

    out = {
        "tag": tag,
        "config_flags": _flags(cfg),
        "summary": pooled,
        "splits": {name: r["summary"] for name, r in runs.items()},
        "detail": detail,
    }
    results_dir = Path(cfg.get("eval.results_dir", "eval/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))
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
