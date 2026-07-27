import sys
from pathlib import Path

import pytest

# Ensure repo root (for `eval` package) and src (for `arag`) are importable even
# without an editable install.
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from arag.common.config import load_config  # noqa: E402


@pytest.fixture
def mock_cfg(tmp_path):
    cfg = load_config(ROOT / "config" / "config.yaml")
    # Isolate the index per test and force deterministic mock mode.
    return cfg.with_overrides(
        {
            "mode": "mock",
            "corpus_dir": str(ROOT / "tests" / "fixtures" / "corpus"),
            "vector_store.persist_dir": str(tmp_path / "index"),
            "embeddings.provider": "mock",
            "llm.provider": "mock",
        }
    )
