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
