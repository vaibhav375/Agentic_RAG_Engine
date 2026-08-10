"""Chunking strategies.

Chunk size / overlap / strategy / contextual-enrichment are all config-driven,
because chunking dominates RAG quality more than people expect and is treated as
a first-class ablation variable.

`contextual_enrichment` implements the Anthropic-style "contextual retrieval"
idea: a short situating prefix (page title + section, or an LLM-written summary)
is prepended to each chunk *before embedding* so short chunks stop losing their
document context. The stored `text` (shown to the user / generator) is unchanged;
only `embed_text` carries the prefix.
"""

from __future__ import annotations

import re

from arag.common.schemas import Chunk
from arag.ingest.load import RawDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _tokens(text: str) -> list[str]:
    return text.split()


def _iter_blocks_with_sections(text: str):
    """Yield (section_title, block_text) for paragraph-ish blocks, tracking the
    nearest preceding markdown heading as the section."""
    section = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            if buf:
                yield section, "\n".join(buf).strip()
                buf = []
            section = m.group(2).strip()
            continue
        if line.strip() == "":
            if buf:
                yield section, "\n".join(buf).strip()
                buf = []
        else:
            buf.append(line)
    if buf:
        yield section, "\n".join(buf).strip()


def _windows(tokens: list[str], size: int, overlap: int):
    if size <= 0:
        yield 0, tokens
        return
    step = max(1, size - overlap)
    i = 0
    n = len(tokens)
    while i < n:
        yield i, tokens[i : i + size]
        if i + size >= n:
            break
        i += step


def _context_prefix(doc: RawDoc, section: str | None) -> str:
    parts = [doc.title or doc.doc_id]
    if section and section != doc.title:
        parts.append(section)
    return " > ".join(parts)


def chunk_document(doc: RawDoc, cfg) -> list[Chunk]:
    c = cfg.chunking
    strategy = c.get("strategy", "recursive")
    size = int(c.get("chunk_size", 512))
    overlap = int(c.get("chunk_overlap", 64))
    enrich = bool(c.get("contextual_enrichment", False))

    chunks: list[Chunk] = []
    ordinal = 0

    def emit(text: str, section: str | None):
        nonlocal ordinal
        text = text.strip()
        if not text:
            return
        chunk_id = f"{doc.doc_id}::{ordinal}"
        embed_text = text
        if enrich:
            embed_text = f"[{_context_prefix(doc, section)}]\n{text}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                embed_text=embed_text,
                section=section,
                source_path=doc.source_path,
                ordinal=ordinal,
                metadata={"title": doc.title},
            )
        )
        ordinal += 1

    if strategy == "fixed":
        for _, toks in _windows(_tokens(doc.text), size, overlap):
            emit(" ".join(toks), None)
        return chunks

    # recursive / sentence: respect section blocks, then window within a block if
    # it is longer than the budget. "sentence" packs whole sentences; "recursive"
    # packs token windows. Both keep section provenance.
    #
    # `pack_blocks` merges consecutive short blocks from the same section up to
    # the budget. Without it every paragraph becomes a chunk however short it is,
    # so `chunk_size` only ever acts as an upper bound and never as a target: on
    # real docs (FastAPI markdown, 76% of blocks under 20 words) that produced
    # 3173 chunks averaging 2.7% of a 512-word budget — one-line fragments too
    # small to answer from, and 52 near-duplicate candidates per document
    # competing in the ranking. Kept configurable because chunking is an
    # ablation variable and the unpacked behaviour has to stay reproducible.
    pack = bool(c.get("pack_blocks", True))
    pending: list[str] = []
    pending_len = 0
    pending_section: str | None = None

    def flush():
        nonlocal pending, pending_len
        if pending:
            emit("\n\n".join(pending), pending_section)
            pending = []
            pending_len = 0

    for section, block in _iter_blocks_with_sections(doc.text):
        toks = _tokens(block)
        if len(toks) > size:
            flush()  # oversized block: window it on its own, in document order
            if strategy == "sentence":
                _emit_by_sentences(block, section, size, overlap, emit)
            else:
                for _, w in _windows(toks, size, overlap):
                    emit(" ".join(w), section)
            continue
        if not pack:
            emit(block, section)
            continue
        # Never merge across a heading — a chunk spanning two sections would
        # carry the wrong `section` provenance for half its text.
        if pending and (section != pending_section or pending_len + len(toks) > size):
            flush()
        if not pending:
            pending_section = section
        pending.append(block)
        pending_len += len(toks)
    flush()
    return chunks


def _emit_by_sentences(block: str, section, size, overlap, emit):
    from arag.providers.base import split_sentences

    sents = split_sentences(block) or [block]
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        slen = len(s.split())
        if cur and cur_len + slen > size:
            emit(" ".join(cur), section)
            # carry overlap sentences
            carried = []
            carried_len = 0
            for prev in reversed(cur):
                carried.insert(0, prev)
                carried_len += len(prev.split())
                if carried_len >= overlap:
                    break
            cur = carried
            cur_len = carried_len
        cur.append(s)
        cur_len += slen
    if cur:
        emit(" ".join(cur), section)


def chunk_corpus(docs: list[RawDoc], cfg) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out
