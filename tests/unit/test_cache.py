from arag.cache.semantic_cache import SemanticCache
from arag.common.schemas import Answer
from arag.providers.base import MockEmbedder


def _ans(text="blue"):
    return Answer(query="q", answer=text)


def test_exact_query_hits():
    cache = SemanticCache(MockEmbedder(dim=256), threshold=0.9)
    cache.put("what color is the widget", _ans("blue"))
    hit = cache.get("what color is the widget")
    assert hit is not None
    assert hit.from_cache is True
    assert hit.answer == "blue"


def test_unrelated_query_misses():
    cache = SemanticCache(MockEmbedder(dim=256), threshold=0.9)
    cache.put("what color is the widget", _ans("blue"))
    assert cache.get("how many gadgets can an account create") is None


def test_threshold_is_respected():
    # A very high threshold should reject near-but-not-identical queries.
    strict = SemanticCache(MockEmbedder(dim=256), threshold=0.999)
    strict.put("battery capacity default value", _ans())
    assert strict.get("battery capacity maximum allowed") is None


def test_stats_track_hits_and_misses():
    cache = SemanticCache(MockEmbedder(dim=256), threshold=0.9)
    cache.put("alpha beta gamma", _ans())
    cache.get("alpha beta gamma")      # hit
    cache.get("totally different xyz")  # miss
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1
