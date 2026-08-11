"""Is IDF coverage the right answerability signal, or just the cheapest one?

The CRAG gate declines before generating when IDF-weighted query coverage falls
below `incorrect_threshold`. That is lexical: it asks whether the query's rare
tokens literally appear in the retrieved text. It cannot tell "the corpus does
not discuss this" from "the corpus says this in different words", and five
in-scope gold questions are declined by it — including "How many middleware
classes SHIP with Breeze?", where the answer is present but the distinctive query
token is not.

The cross-encoder reranker already scores query-chunk relevance semantically, and
it is already computed whenever use_rerank is on, so gating on it costs nothing
extra. This measures whether it separates answerable from unanswerable better
than IDF coverage does.

No LLM: retrieval and reranking only, over the whole gold set. What matters is
not which signal scores higher on average but which achieves a given safety level
(refusing the unanswerable) at a lower false-abstention cost — the two are traded
against each other, so both are reported across the full threshold sweep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.agent.retrieval_grader import idf_coverage  # noqa: E402
from arag.common.config import load_config  # noqa: E402
from arag.engine import build_components, retrieve_contexts  # noqa: E402
from arag.ingest.index import build_index  # noqa: E402
from eval.build_gold_set import load_gold  # noqa: E402

OUT = Path("eval/results/crag_signal.json")
# Answerable slices; the rest are questions the corpus genuinely cannot answer.
ANSWERABLE = {"easy", "multi_hop"}


def _sweep(rows: list[dict], key: str) -> list[dict]:
    """For each candidate threshold, what fraction of each class falls below it."""
    out = []
    for i in range(0, 101):
        t = i / 100.0
        # Declining below the threshold: unanswerable declined = safety kept,
        # answerable declined = a false abstention.
        refused_bad = [r for r in rows if not r["answerable"] and r[key] < t]
        refused_good = [r for r in rows if r["answerable"] and r[key] < t]
        n_bad = sum(1 for r in rows if not r["answerable"]) or 1
        n_good = sum(1 for r in rows if r["answerable"]) or 1
        out.append({
            "threshold": round(t, 2),
            "unanswerable_declined": round(len(refused_bad) / n_bad, 4),
            "false_abstentions": round(len(refused_good) / n_good, 4),
        })
    return out


def main() -> int:
    cfg = load_config("config/config.yaml").with_overrides({
        "mode": "local", "embeddings.provider": "sentence_transformers",
        "vector_store.persist_dir": ".arag_index_local",
        "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
    })
    comp = build_components(cfg, store=build_index(cfg))

    rows = []
    for g in load_gold(cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl")):
        ranked, contexts = retrieve_contexts(comp, g.question)
        rows.append({
            "id": g.id,
            "difficulty": g.difficulty.value,
            "answerable": g.difficulty.value in ANSWERABLE,
            "idf": round(idf_coverage(comp.store, g.question, contexts), 4),
            # Reranker scores are unbounded logits; the gate needs a comparable
            # scale, so report the top chunk's score as-is and sweep over it.
            "rerank_top": round(max((c.score for c in contexts), default=0.0), 4),
        })
        print(f"{g.id:5s} {rows[-1]['difficulty']:13s} idf={rows[-1]['idf']:.3f} "
              f"rerank={rows[-1]['rerank_top']:.3f}", flush=True)

    result = {
        "rows": rows,
        "idf_sweep": _sweep(rows, "idf"),
        "rerank_sweep": _sweep(rows, "rerank_top"),
    }
    OUT.write_text(json.dumps(result, indent=2))

    print("\n=== safety kept vs coverage lost ===", flush=True)
    print("(at each level of declining the unanswerable, what does it cost?)", flush=True)
    print(f"{'unanswerable declined':>22} {'idf false-abst':>16} {'rerank false-abst':>18}",
          flush=True)
    for target in (0.8, 0.9, 0.95, 1.0):
        best = {}
        for key, sweep in (("idf", result["idf_sweep"]), ("rerank", result["rerank_sweep"])):
            ok = [s for s in sweep if s["unanswerable_declined"] >= target]
            best[key] = min((s["false_abstentions"] for s in ok), default=float("nan"))
        print(f"{target:>22.2f} {best['idf']:>16.3f} {best['rerank']:>18.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
