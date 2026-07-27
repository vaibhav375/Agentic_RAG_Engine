import os

from arag.common.config import load_config


def test_env_override_coerces_types(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "retrieval:\n  k_dense: 10\n  use_hybrid: false\ncache:\n  similarity_threshold: 0.9\n"
    )
    monkeypatch.setenv("ARAG_RETRIEVAL__K_DENSE", "25")
    monkeypatch.setenv("ARAG_RETRIEVAL__USE_HYBRID", "true")
    monkeypatch.setenv("ARAG_CACHE__SIMILARITY_THRESHOLD", "0.8")
    cfg = load_config(cfg_file)
    assert cfg.get("retrieval.k_dense") == 25
    assert cfg.get("retrieval.use_hybrid") is True
    assert abs(cfg.get("cache.similarity_threshold") - 0.8) < 1e-9
    for k in ("ARAG_RETRIEVAL__K_DENSE", "ARAG_RETRIEVAL__USE_HYBRID", "ARAG_CACHE__SIMILARITY_THRESHOLD"):
        os.environ.pop(k, None)


def test_with_overrides_is_isolated(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("retrieval:\n  k_dense: 10\n")
    cfg = load_config(cfg_file)
    cfg2 = cfg.with_overrides({"retrieval.k_dense": 99})
    assert cfg.get("retrieval.k_dense") == 10  # original untouched
    assert cfg2.get("retrieval.k_dense") == 99
