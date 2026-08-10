from arag.ingest.chunk import chunk_document
from arag.ingest.load import RawDoc


def _doc():
    text = "# Title\n\n" + " ".join(f"word{i}" for i in range(300))
    return RawDoc(doc_id="d1", text=text, source_path="d1.md", title="Title")


def test_fixed_chunking_overlap(mock_cfg):
    cfg = mock_cfg.with_overrides(
        {"chunking.strategy": "fixed", "chunking.chunk_size": 100, "chunking.chunk_overlap": 20}
    )
    chunks = chunk_document(_doc(), cfg)
    assert len(chunks) >= 3
    # Overlap: end of chunk 0 should reappear at start of chunk 1.
    c0_tokens = chunks[0].text.split()
    c1_tokens = chunks[1].text.split()
    assert c0_tokens[-20:] == c1_tokens[:20]
    # Deterministic chunk ids.
    assert chunks[0].chunk_id == "d1::0"


def test_contextual_enrichment_prefixes_embed_text(mock_cfg):
    cfg = mock_cfg.with_overrides({"chunking.contextual_enrichment": True})
    chunks = chunk_document(_doc(), cfg)
    assert chunks[0].embed_text != chunks[0].text
    assert chunks[0].embed_text.startswith("[")
    assert "Title" in chunks[0].embed_text


def test_no_enrichment_embed_text_equals_text(mock_cfg):
    cfg = mock_cfg.with_overrides({"chunking.contextual_enrichment": False})
    chunks = chunk_document(_doc(), cfg)
    assert chunks[0].embed_text == chunks[0].text


def _short_blocks_doc() -> RawDoc:
    """A page of one-line paragraphs — what real markdown docs mostly are."""
    body = "\n\n".join(f"line {i} of the docs" for i in range(20))
    return RawDoc(doc_id="d2", text=f"# Title\n\n{body}", source_path="d2.md", title="Title")


def test_packing_fills_the_chunk_size_budget(mock_cfg):
    """Short paragraphs are merged, so `chunk_size` acts as a target not a cap.

    Unpacked, this document produced 20 chunks of 5 words each — fragments too
    small to answer from, and 20 near-duplicate ranking candidates.
    """
    cfg = mock_cfg.with_overrides({"chunking.chunk_size": 50, "chunking.pack_blocks": True})
    chunks = chunk_document(_short_blocks_doc(), cfg)
    assert len(chunks) == 2  # 20 blocks x 5 words = 100 words -> two 50-word chunks
    assert all(len(c.text.split()) <= 50 for c in chunks)
    # Nothing dropped, and document order preserved.
    assert sum(c.text.count("line") for c in chunks) == 20
    assert chunks[0].text.startswith("line 0") and chunks[1].text.endswith("line 19 of the docs")
    assert [c.chunk_id for c in chunks] == ["d2::0", "d2::1"]


def test_pack_blocks_false_keeps_one_chunk_per_paragraph(mock_cfg):
    cfg = mock_cfg.with_overrides({"chunking.chunk_size": 50, "chunking.pack_blocks": False})
    chunks = chunk_document(_short_blocks_doc(), cfg)
    assert len(chunks) == 20


def test_packing_never_merges_across_a_heading(mock_cfg):
    """A chunk spanning two sections would carry the wrong section provenance."""
    text = "# Title\n\n## Alpha\n\nshort one\n\nshort two\n\n## Beta\n\nshort three\n"
    doc = RawDoc(doc_id="d3", text=text, source_path="d3.md", title="Title")
    cfg = mock_cfg.with_overrides({"chunking.chunk_size": 500, "chunking.pack_blocks": True})
    chunks = chunk_document(doc, cfg)
    assert [c.section for c in chunks] == ["Alpha", "Beta"]
    assert "short one" in chunks[0].text and "short two" in chunks[0].text
    assert chunks[1].text.strip() == "short three"


def test_packing_still_windows_oversized_blocks(mock_cfg):
    cfg = mock_cfg.with_overrides(
        {"chunking.chunk_size": 100, "chunking.chunk_overlap": 20, "chunking.pack_blocks": True}
    )
    chunks = chunk_document(_doc(), cfg)  # single 300-word block
    assert len(chunks) >= 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
