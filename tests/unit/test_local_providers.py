"""Config plumbing for open-weight backends.

None of this needs a model or a network: it asserts that the config reaches the
transport correctly, which is where the first real local run actually broke —
`max_tokens` was never sent to Ollama, so the model ran unbounded past the
hardcoded timeout.
"""

import json

import pytest

from arag.common.config import load_config
from arag.providers.llm import PromptLLM
from eval.build_gold_set import load_gold
from eval.run_eval import stratified_subset


def _cfg(**over):
    return load_config("config/config.yaml").with_overrides(over)


# ----------------------------------------------------------------- ollama


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def captured(monkeypatch):
    """Capture the JSON body the Ollama transport posts."""
    sent = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        sent["url"], sent["json"], sent["timeout"] = url, json, timeout
        return _FakeResponse({"message": {"content": "ok", "thinking": "SHOULD NOT LEAK"}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_ollama_sends_max_tokens_as_num_predict(captured):
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama", "llm.max_tokens": 256}))
    llm._complete("sys", "user")
    assert captured["json"]["options"]["num_predict"] == 256


def test_ollama_timeout_is_config_driven(captured):
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama", "llm.timeout_seconds": 42}))
    llm._complete("sys", "user")
    assert captured["timeout"] == 42


def test_ollama_think_flag_is_omitted_when_null(captured):
    """Ollama < 0.9 rejects an unknown `think` field, so it must not be sent."""
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama", "llm.think": None}))
    llm._complete("sys", "user")
    assert "think" not in captured["json"]


def test_ollama_think_flag_is_sent_when_set(captured):
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama", "llm.think": False}))
    llm._complete("sys", "user")
    assert captured["json"]["think"] is False


def test_ollama_never_returns_the_thinking_trace(captured):
    """Reasoning traces in the answer text would wreck citation parsing."""
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama"}))
    assert llm._complete("sys", "user") == "ok"


def test_ollama_uses_the_configured_model(captured):
    llm = PromptLLM(_cfg(**{"llm.provider": "ollama", "llm.ollama_model": "qwen3:4b"}))
    llm._complete("sys", "user")
    assert captured["json"]["model"] == "qwen3:4b"


# -------------------------------------------------- OpenAI-compatible hosts


def test_openai_defaults_to_openai_itself():
    llm = PromptLLM(_cfg(**{"llm.provider": "openai"}))
    assert llm.base_url == "" and llm.api_key_env == "OPENAI_API_KEY"


def test_base_url_and_key_env_reach_an_openai_compatible_host():
    """Open-weight models too big to run locally (Kimi K2, Qwen 235B) are served
    over the OpenAI protocol — reachable with config alone, no new backend."""
    llm = PromptLLM(_cfg(**{
        "llm.provider": "openai",
        "llm.base_url": "https://api.moonshot.ai/v1",
        "llm.api_key_env": "MOONSHOT_API_KEY",
        "llm.model": "kimi-k2-0711-preview",
    }))
    assert llm.base_url == "https://api.moonshot.ai/v1"
    assert llm.api_key_env == "MOONSHOT_API_KEY"
    assert llm.model == "kimi-k2-0711-preview"


def test_base_url_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARAG_LLM_BASE_URL", "http://localhost:8000/v1")
    assert PromptLLM(_cfg(**{"llm.provider": "openai"})).base_url == "http://localhost:8000/v1"


# --------------------------------------------------------- subset sampling


def test_subset_covers_every_slice():
    """A plain gold[:n] subset is 100% easy — it would report abstention and
    robustness metrics with no unanswerable or adversarial questions behind them."""
    gold = load_gold("data/eval/gold_qa.jsonl")
    slices = {g.difficulty.value for g in gold}
    for n in (8, 16, 31):
        got = {g.difficulty.value for g in stratified_subset(gold, n)}
        assert got == slices, f"n={n} missed {slices - got}"


def test_subset_is_proportional():
    gold = load_gold("data/eval/gold_qa.jsonl")
    half = stratified_subset(gold, len(gold) // 2)
    easy_full = sum(g.difficulty.value == "easy" for g in gold) / len(gold)
    easy_half = sum(g.difficulty.value == "easy" for g in half) / len(half)
    assert abs(easy_full - easy_half) < 0.1


def test_subset_is_deterministic_and_ordered():
    gold = load_gold("data/eval/gold_qa.jsonl")
    a = [g.id for g in stratified_subset(gold, 16)]
    assert a == [g.id for g in stratified_subset(gold, 16)]
    order = [g.id for g in gold]
    assert a == sorted(a, key=order.index)  # preserves file order


def test_subset_larger_than_gold_returns_everything():
    gold = load_gold("data/eval/gold_qa.jsonl")
    assert len(stratified_subset(gold, 999)) == len(gold)


def test_subset_smaller_than_slice_count_still_covers_all_slices():
    """Documented trade-off: slice coverage wins over the exact count."""
    gold = load_gold("data/eval/gold_qa.jsonl")
    got = stratified_subset(gold, 2)
    assert len({g.difficulty.value for g in got}) == len({g.difficulty.value for g in gold})


# ------------------------------------------------------- role-specific models


def test_judge_model_overrides_only_the_judge():
    """A 3B model judging its own output scored correct answers as hallucinations
    (docs/local-mode-eval.md), so the critic must be able to run on its own model."""
    from arag.providers.llm import make_llm

    cfg = _cfg(**{"llm.provider": "ollama", "llm.ollama_model": "llama3.2:3b",
                  "llm.judge_model": "qwen2.5:7b"})
    assert make_llm(cfg).ollama_model == "llama3.2:3b"
    assert make_llm(cfg, role="judge").ollama_model == "qwen2.5:7b"


def test_roles_fall_back_to_the_generation_model():
    from arag.providers.llm import make_llm

    cfg = _cfg(**{"llm.provider": "ollama", "llm.ollama_model": "llama3.2:3b",
                  "llm.judge_model": None, "llm.router_model": None})
    for role in (None, "judge", "router"):
        assert make_llm(cfg, role=role).ollama_model == "llama3.2:3b"


def test_role_override_targets_the_providers_model_field():
    """Each provider keeps its model id in a different field."""
    from arag.providers.llm import make_llm

    openai = _cfg(**{"llm.provider": "openai", "llm.model": "gpt-4o-mini",
                     "llm.judge_model": "gpt-4o"})
    assert make_llm(openai, role="judge").model == "gpt-4o"
    anthropic = _cfg(**{"llm.provider": "anthropic", "llm.judge_model": "claude-x"})
    assert make_llm(anthropic, role="judge").anthropic_model == "claude-x"


def test_mock_mode_ignores_role_models():
    """Mock stays deterministic and single-model, so CI numbers don't move."""
    from arag.providers.llm import MockLLM, make_llm

    cfg = _cfg(**{"llm.provider": "mock", "llm.judge_model": "qwen2.5:7b"})
    assert isinstance(make_llm(cfg, role="judge"), MockLLM)


def test_components_expose_judge_and_router(tmp_path):
    """The critic and router call sites read comp.judge / comp.router."""
    from arag.engine import build_components
    from arag.ingest.index import build_index

    cfg = _cfg(**{"corpus_dir": "tests/fixtures/corpus",
                  "vector_store.persist_dir": str(tmp_path / "idx")})
    comp = build_components(cfg, store=build_index(cfg))
    assert comp.judge is not None and comp.router is not None


def test_config_ships_an_openai_compatible_example():
    """The config must document how to reach open-weight hosts, or nobody finds it."""
    raw = json.dumps(load_config("config/config.yaml").as_dict())
    assert "base_url" in raw and "api_key_env" in raw
