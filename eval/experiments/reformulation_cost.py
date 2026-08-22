"""Do corrective retries retrieve worse than the query they replaced?

`recall_at_1` fell 0.867 -> 0.800 as retries went up, which points at the retry
loop hurting the thing it exists to help: an unsupported answer triggers a
reformulated, deliberately broadened query, and the contexts it returns replace
the originals wholesale. If the broadened query retrieves worse, the loop trades a
good context set for a worse one and then abstains because it cannot ground an
answer in it.

Retrieval only — no LLM, no NLI. The reformulated queries were recorded by real
runs; this replays retrieval for the original question and for each reformulation
and scores both against the gold supporting docs. That also makes it the one
experiment here that fits comfortably on an 8 GB machine: embedder plus reranker,
roughly 1.2 GB, with nothing else resident.

What would confirm the hypothesis: reformulated queries scoring materially lower
recall@1 than the originals they replaced.

Only some replays can show that. Unanswerable questions have no gold doc, and a
question whose gold doc is never retrieved scores 0.00 either way — both are inert
and counting them as "unchanged" doubled the apparent sample the first time this
was run. Only the informative subset is reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arag.common.config import load_config  # noqa: E402
from arag.engine import build_components, retrieve_contexts  # noqa: E402
from arag.ingest.index import build_index  # noqa: E402
from eval.build_gold_set import load_gold  # noqa: E402
from eval.metrics import recall_at_k  # noqa: E402

OUT = Path("eval/results/reformulation_cost.json")
SOURCES = [
    ("packed", "eval/results/pack_packed.json", True, ".arag_index_pack_packed"),
    ("unpacked", "eval/results/pack_unpacked.json", False, ".arag_index_pack_unpacked"),
]


def _score(comp, query: str, gold_docs: list[str]) -> tuple[float, float]:
    ranked, _ = retrieve_contexts(comp, query)
    docs = [rc.chunk.doc_id for rc in ranked]
    return recall_at_k(docs, gold_docs, 1), recall_at_k(docs, gold_docs, 3)


def main() -> int:
    gold = {g.id: g for g in load_gold("data/eval/gold_qa.jsonl")}
    rows = []

    for label, path, pack, persist in SOURCES:
        detail = json.loads(Path(path).read_text())["detail"]
        retried = [r for r in detail if r.get("reformulations")]
        if not retried:
            continue
        cfg = load_config("config/config.yaml").with_overrides({
            "mode": "local", "embeddings.provider": "sentence_transformers",
            "corpus_dir": "data/corpus_scaled", "vector_store.persist_dir": persist,
            "chunking.pack_blocks": pack,
            "retrieval.use_hybrid": True, "retrieval.use_rerank": True,
        })
        comp = build_components(cfg, store=build_index(cfg))
        print(f"\n=== {label}: {len(retried)} questions that retried", flush=True)

        for rec in retried:
            g = gold.get(rec["id"])
            if g is None:
                continue
            r1_orig, r3_orig = _score(comp, rec["question"], g.supporting_doc_ids)
            for i, qq in enumerate(rec["reformulations"], 1):
                r1_new, r3_new = _score(comp, qq, g.supporting_doc_ids)
                rows.append({
                    "arm": label, "id": rec["id"], "iteration": i,
                    "recall1_original": r1_orig, "recall1_reformulated": r1_new,
                    "recall3_original": r3_orig, "recall3_reformulated": r3_new,
                    "original": rec["question"], "reformulated": qq,
                })
                flag = "WORSE" if r1_new < r1_orig else ("better" if r1_new > r1_orig else "same ")
                print(f"  {rec['id']} it{i}: recall@1 {r1_orig:.2f} -> {r1_new:.2f}  {flag}",
                      flush=True)

    if not rows:
        print("no recorded reformulations to score", flush=True)
        return 0

    # Most rows cannot move, and counting them as "unchanged" doubles the apparent
    # sample. Two kinds are inert: unanswerable questions have no gold doc to
    # retrieve, and questions whose gold doc is never retrieved score 0.00 either
    # way — a retrieval failure, not a reformulation effect.
    for r in rows:
        g = gold.get(r["id"])
        if not (g and g.supporting_doc_ids):
            r["class"] = "no-gold-docs"
        elif r["recall1_original"] == 0 and r["recall1_reformulated"] == 0:
            r["class"] = "never-retrieved"
        else:
            r["class"] = "informative"

    inf = [r for r in rows if r["class"] == "informative"]
    worse = [r for r in inf if r["recall1_reformulated"] < r["recall1_original"]]
    better = [r for r in inf if r["recall1_reformulated"] > r["recall1_original"]]
    same = len(inf) - len(worse) - len(better)
    questions = len({r["id"] for r in inf})
    mean_o = sum(r["recall1_original"] for r in inf) / len(inf) if inf else 0.0
    mean_n = sum(r["recall1_reformulated"] for r in inf) / len(inf) if inf else 0.0

    print(f"\n=== {len(rows)} reformulations replayed, {len(inf)} of them informative "
          f"(across {questions} questions) ===", flush=True)
    print(f"  inert: {sum(1 for r in rows if r['class'] == 'no-gold-docs')} with no gold doc, "
          f"{sum(1 for r in rows if r['class'] == 'never-retrieved')} never retrieved either way",
          flush=True)
    print(f"  recall@1 worse: {len(worse)}   better: {len(better)}   unchanged: {same}", flush=True)
    print(f"  mean recall@1  original {mean_o:.4f} -> reformulated {mean_n:.4f} "
          f"({mean_n - mean_o:+.4f})", flush=True)
    print("\nThe loop replaces the original contexts with the reformulated ones, so a"
          "\nnegative delta would mean retrying costs retrieval quality outright.", flush=True)

    OUT.write_text(json.dumps({
        "n_replayed": len(rows), "n_informative": len(inf), "n_questions": questions,
        "worse": len(worse), "better": len(better), "unchanged": same,
        "mean_recall1_original": round(mean_o, 4),
        "mean_recall1_reformulated": round(mean_n, 4),
        "rows": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
