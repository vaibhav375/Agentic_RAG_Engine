"""Generate docs/question-bank.md from the gold set.

Generated rather than hand-written, so it cannot drift from the benchmark it
documents. The gold file is the source of truth for what this pipeline is
measured on; a hand-maintained copy would go stale the first time a question
changed, and a question bank that lists questions the corpus no longer answers is
worse than none.

Run: `make question-bank`
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arag.common.config import load_config  # noqa: E402
from eval.build_gold_set import load_gold  # noqa: E402

OUT = Path("docs/question-bank.md")

TITLES = {
    "01_routing": "Routing",
    "02_query_params": "Query parameters",
    "03_request_body": "Request bodies",
    "04_dependency_injection": "Dependency injection",
    "05_middleware": "Middleware",
    "06_background_tasks": "Background tasks",
    "07_authentication": "Authentication",
    "08_error_handling": "Error handling",
    "09_caching": "Response caching",
    "10_configuration": "Configuration",
}

HEADER = """# Question bank

Every question the pipeline is measured on, with the answer it should give or the
reason it should refuse. Generated from `data/eval/gold_qa.jsonl` — regenerate with
`make question-bank` rather than editing by hand.

The corpus is **Breeze**, a fictional Python web framework written specifically for
this benchmark. It is invented on purpose: a model cannot answer from memory about
a framework that does not exist, so a correct specific answer must have come from
the retrieved passages. That is what makes hallucination measurable here rather
than a guess.

## How to phrase a question

The answerability gate scores **how much of your question's distinctive vocabulary
appears in the retrieved text**. Wording therefore decides the outcome more than
meaning does, which is a real limitation and not a quirk to work around:

| question | gate | outcome |
|---|---|---|
| `What does the cache decorator do?` | 1.00 | answered |
| `Which HTTP methods does the cache decorator cache?` | 0.67 | answered |
| `How does caching work in Breeze?` | 0.53 | **declined** |
| `What does the documentation say about path parameters?` | 0.31 | **declined** |

Name a concrete thing from the docs — `@cache`, `GZipMiddleware`, `Depends`,
`BackgroundTasks`, `422`, `.env`, `bearer token`. Generic connective phrasing
("how does X work", "tell me about X") strips down to almost no distinctive
vocabulary and gets declined even when the answer is sitting in the corpus. This
is the documented under-abstention/over-abstention trade: 13.8% of answerable
questions are refused.

## What each category tests

| category | count | expected behaviour |
|---|---|---|
| Answerable | {n_easy} | answer, with citations to the supporting passage |
| Multi-hop | {n_multi} | join two passages into one answer |
| Out of scope | {n_unans} | **decline** — the corpus cannot answer it |
| Adversarial | {n_adv} | refuse *or* actively refute the planted falsehood |

Adversarial scoring is **refutation-aware**: a grounded answer that never addresses
the planted falsehood does not pass, because the reader still leaves believing it.
"""


def main() -> int:
    cfg = load_config("config/config.yaml")
    gold = load_gold(cfg.get("eval.gold_path", "data/eval/gold_qa.jsonl"))

    by_diff: dict[str, list] = defaultdict(list)
    for g in gold:
        by_diff[g.difficulty.value].append(g)

    out = [
        HEADER.format(
            n_easy=len(by_diff["easy"]),
            n_multi=len(by_diff["multi_hop"]),
            n_unans=len(by_diff["unanswerable"]),
            n_adv=len(by_diff["adversarial"]),
        )
    ]

    # Answerable, grouped by the document that answers them.
    by_doc: dict[str, list] = defaultdict(list)
    for g in by_diff["easy"]:
        by_doc[(g.supporting_doc_ids or ["other"])[0]].append(g)

    out.append("\n## Answerable\n")
    out.append("Each should produce an answer citing the listed document.\n")
    for doc in sorted(by_doc):
        out.append(f"\n### {TITLES.get(doc, doc)}  \n`{doc}`\n")
        out.append("| # | Question | Expected answer |")
        out.append("|---|---|---|")
        for g in by_doc[doc]:
            out.append(f"| `{g.id}` | {_cell(g.question)} | {_cell(g.answer)} |")

    out.append("\n## Multi-hop\n")
    out.append("Each needs two passages joined into a single answer.\n")
    out.append("| # | Question | Expected answer |")
    out.append("|---|---|---|")
    for g in by_diff["multi_hop"]:
        out.append(f"| `{g.id}` | {_cell(g.question)} | {_cell(g.answer)} |")

    out.append("\n## Out of scope — should decline\n")
    out.append(
        "Nothing in the corpus answers these. A correct run refuses rather than "
        "guessing; the benchmark records 12 of 12 declined.\n"
    )
    out.append("| # | Question |")
    out.append("|---|---|")
    for g in by_diff["unanswerable"]:
        out.append(f"| `{g.id}` | {_cell(g.question)} |")

    out.append("\n## Adversarial — should resist or refute\n")
    out.append(
        "Prompt injections and false premises stated as established fact. Where a "
        "question plants a specific falsehood, passing requires abstaining *or* "
        "contradicting it — staying silent about it is not a pass.\n"
    )
    out.append("| # | Question | Planted falsehood to refute |")
    out.append("|---|---|---|")
    for g in by_diff["adversarial"]:
        out.append(f"| `{g.id}` | {_cell(g.question)} | {_cell(g.must_refute) or '—'} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT} — {len(gold)} questions")
    return 0


def _cell(text: str | None) -> str:
    """Markdown-table-safe single line."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    raise SystemExit(main())
