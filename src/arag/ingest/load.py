"""Corpus loading. Parses markdown/text/HTML files into raw documents, preserving
source metadata (doc id, path, section) for citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class RawDoc:
    doc_id: str
    text: str
    source_path: str
    title: str | None = None
    metadata: dict = field(default_factory=dict)


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text)


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return fallback


def load_corpus(corpus_dir: str | Path, extensions: tuple[str, ...] = (".md", ".txt", ".html")) -> list[RawDoc]:
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs: list[RawDoc] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in extensions:
            continue
        if path.stem.lower() == "readme":
            continue  # documentation about the corpus, not part of it
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".html":
            text = _strip_html(text)
        doc_id = path.stem
        title = _title_from_markdown(text, doc_id)
        docs.append(
            RawDoc(
                doc_id=doc_id,
                text=text,
                source_path=str(path),
                title=title,
            )
        )
    if not docs:
        raise ValueError(f"No documents found under {corpus_dir}")
    return docs
