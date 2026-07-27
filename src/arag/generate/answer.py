"""Grounded answer generation with citations.

Takes the top-n retrieved context and asks the LLM (real or mock) to answer using
only that context, emitting citations that map back to chunk ids for provenance.
"""

from __future__ import annotations

from arag.common.schemas import Answer, Citation, RetrievedChunk
from arag.providers.base import LanguageModel


def generate_answer(
    llm: LanguageModel,
    query: str,
    contexts: list[RetrievedChunk],
) -> Answer:
    ctx_pairs = [(rc.chunk.chunk_id, rc.chunk.text) for rc in contexts]
    gen = llm.generate_answer(query, ctx_pairs)

    by_id = {rc.chunk.chunk_id: rc for rc in contexts}
    citations: list[Citation] = []
    for cid in gen.cited_chunk_ids:
        rc = by_id.get(cid)
        if rc is None:
            continue
        citations.append(
            Citation(
                chunk_id=cid,
                doc_id=rc.chunk.doc_id,
                section=rc.chunk.section,
                quote=_short_quote(rc.chunk.text),
            )
        )

    return Answer(
        query=query,
        answer=gen.text,
        citations=citations,
        contexts=contexts,
        abstained=gen.abstained,
    )


def _short_quote(text: str, max_chars: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"
