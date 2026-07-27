"""Build / validate the gold evaluation set.

Validates that every gold question references documents that exist in the corpus
and that its `supporting_quote` is actually present in some chunk after
ingestion. This is the check that keeps the metrics meaningful — a weak or
mismatched gold set silently poisons every downstream number.

Because chunking is an ablation variable (chunk ids change across configs),
gold support is anchored on `supporting_doc_ids` + a verbatim `supporting_quote`,
not on fixed chunk ids. Context precision/recall are therefore computed at the
document level.
"""

from __future__ import annotations

import json
from pathlib import Path

from arag.common.schemas import GoldQA
from arag.ingest.index import build_index, load_store


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def load_gold(path: str | Path) -> list[GoldQA]:
    rows: list[GoldQA] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(GoldQA.model_validate_json(line))
    return rows


def gold_chunk_ids(store, gold: GoldQA) -> list[str]:
    """Locate chunks that support this gold item at the current chunking, by
    matching the (whitespace-normalized) supporting_quote inside a chunk."""
    if not gold.supporting_quote:
        return []
    q = normalize_ws(gold.supporting_quote).lower()
    out = []
    for c in store.chunks:
        if q and q in normalize_ws(c.text).lower():
            out.append(c.chunk_id)
    return out


def validate_gold_set(cfg) -> dict:
    gold_path = cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl")
    gold = load_gold(gold_path)

    try:
        store = load_store(cfg)
    except FileNotFoundError:
        store = build_index(cfg)

    corpus_doc_ids = {c.doc_id for c in store.chunks}
    errors: list[str] = []
    by_difficulty: dict[str, int] = {}
    quote_hits = 0

    for g in gold:
        by_difficulty[g.difficulty.value] = by_difficulty.get(g.difficulty.value, 0) + 1
        for doc_id in g.supporting_doc_ids:
            if doc_id not in corpus_doc_ids:
                errors.append(f"{g.id}: supporting_doc_id '{doc_id}' not in corpus")
        if g.is_answerable and g.supporting_quote:
            hits = gold_chunk_ids(store, g)
            if not hits:
                errors.append(f"{g.id}: supporting_quote not found in any chunk")
            else:
                quote_hits += 1
        if g.is_answerable and not g.supporting_doc_ids:
            errors.append(f"{g.id}: answerable question has no supporting_doc_ids")

    return {
        "n_questions": len(gold),
        "by_difficulty": by_difficulty,
        "quotes_located": quote_hits,
        "n_docs": len(corpus_doc_ids),
        "errors": errors,
    }


if __name__ == "__main__":
    import sys

    from arag.common.config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
    report = validate_gold_set(cfg)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        sys.exit(1)
