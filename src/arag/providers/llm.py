"""LLM backends implementing the high-level `LanguageModel` task interface.

- `PromptLLM`  — real backend for OpenAI / Anthropic / Ollama. Each task builds a
  prompt, calls `_complete`, and parses structured output.
- `MockLLM`    — deterministic, dependency-free implementation of the same tasks.
  It simulates a naive RAG generator that is *helpful but will fabricate* when the
  retrieved context is weak, so the eval harness measures a real hallucination
  signal and the self-correction loop has something to catch.
"""

from __future__ import annotations

import json
import os
import re

from arag.generate import prompts
from arag.providers.base import (
    GeneratedAnswer,
    LanguageModel,
    content_tokens,
    lexical_overlap,
    split_sentences,
)

# Overlap of the best available context sentence with the query, below which a
# naive generator has no real grounding and (in mock) will fabricate.
_ANSWERABLE_OVERLAP = 0.34
# Per-claim support threshold used by the mock judge.
_CLAIM_SUPPORT_OVERLAP = 0.30


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
class MockLLM(LanguageModel):
    def generate_answer(self, query: str, contexts: list[tuple[str, str]]) -> GeneratedAnswer:
        # Rank all context sentences by overlap with the query.
        scored: list[tuple[float, str, str]] = []  # (overlap, sentence, chunk_id)
        for cid, text in contexts:
            for sent in split_sentences(text):
                scored.append((lexical_overlap(query, sent), sent, cid))
        scored.sort(key=lambda x: x[0], reverse=True)

        best = scored[0][0] if scored else 0.0
        if best >= _ANSWERABLE_OVERLAP:
            # Grounded: stitch the top sentences that are actually on-topic.
            picked = [(s, c) for o, s, c in scored if o >= _ANSWERABLE_OVERLAP][:2]
            answer = " ".join(s for s, _ in picked)
            cited = list(dict.fromkeys(c for _, c in picked))
            return GeneratedAnswer(text=answer, cited_chunk_ids=cited, abstained=False)

        # Weak context -> a naive LLM leans on parametric priors and fabricates a
        # confident-but-unsupported answer. We synthesize one from the query.
        # `sorted` keeps this independent of PYTHONHASHSEED so results reproduce.
        topic = " ".join(sorted(content_tokens(query))[:4]) or "this"
        answer = (
            f"Yes. {topic.capitalize()} is supported and enabled by default, and "
            f"you can configure it directly without any additional setup."
        )
        return GeneratedAnswer(text=answer, cited_chunk_ids=[], abstained=False)

    def extract_claims(self, answer: str) -> list[str]:
        return split_sentences(answer) or ([answer] if answer.strip() else [])

    def judge_claim(self, claim: str, context: str) -> tuple[bool, str, float]:
        score = lexical_overlap(claim, context)
        supported = score >= _CLAIM_SUPPORT_OVERLAP
        reason = "grounded in context" if supported else "not entailed by context"
        return supported, reason, round(score, 4)

    def reformulate(self, query: str, missing_info: str | None, prior: list[str]) -> str:
        # Widen by appending salient tokens from the missing-info hint.
        extra = " ".join(sorted(content_tokens(missing_info or ""))[:5])
        base = query.strip().rstrip("?")
        candidate = f"{base} {extra}".strip()
        # Avoid re-issuing an identical query.
        if candidate in prior:
            candidate = f"{candidate} details reference"
        return candidate

    def classify_complexity(self, query: str) -> str:
        toks = content_tokens(query)
        multi_signals = sum(
            kw in query.lower() for kw in (" and ", " vs ", "compare", "difference", "both", "how many")
        )
        if len(toks) > 8 or multi_signals >= 1:
            return "complex"
        return "simple"

    def hypothetical_document(self, query: str) -> str:
        # Deterministic HyDE stand-in: assert the query as a declarative passage
        # so its embedding sits nearer answer-space than the interrogative form.
        stem = query.strip().rstrip("?")
        for qw in ("how do you ", "how do i ", "what is ", "what are ", "how many ", "does ", "is "):
            if stem.lower().startswith(qw):
                stem = stem[len(qw):]
                break
        return f"{stem}. {query}"

    def decompose(self, query: str) -> list[str]:
        import re

        parts = re.split(r"\b(?:and|versus|vs\.?)\b|[;,]| compared to ", query, flags=re.IGNORECASE)
        subs = [p.strip(" ?.") for p in parts if len(content_tokens(p)) >= 2]
        # Only treat as multi-hop if it genuinely split into 2+ substantive parts.
        return subs if len(subs) >= 2 else [query]


# --------------------------------------------------------------------------- #
# Real
# --------------------------------------------------------------------------- #
class PromptLLM(LanguageModel):
    def __init__(self, cfg):
        self.cfg = cfg
        llm = cfg.llm
        self.provider = llm.provider
        self.model = llm.model
        self.anthropic_model = llm.get("anthropic_model", "claude-3-5-sonnet-latest")
        self.ollama_model = llm.get("ollama_model", "llama3.1:8b")
        self.temperature = float(llm.get("temperature", 0.0))
        self.max_tokens = int(llm.get("max_tokens", 1024))
        self._client = None

    # -- transport ---------------------------------------------------------- #
    def _complete(self, system: str, user: str) -> str:
        if self.provider == "openai":
            return self._openai(system, user)
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        if self.provider == "ollama":
            return self._ollama(system, user)
        raise ValueError(f"Unknown llm provider: {self.provider}")

    def _openai(self, system: str, user: str) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def _anthropic(self, system: str, user: str) -> str:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = self._client.messages.create(
            model=self.anthropic_model,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _ollama(self, system: str, user: str) -> str:
        import httpx

        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = httpx.post(
            f"{base}/api/chat",
            json={
                "model": self.ollama_model,
                "stream": False,
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    # -- tasks -------------------------------------------------------------- #
    def generate_answer(self, query: str, contexts: list[tuple[str, str]]) -> GeneratedAnswer:
        block = "\n".join(f"[{cid}] {text}" for cid, text in contexts)
        system = prompts.ANSWER_SYSTEM.format(abstain=prompts.ABSTAIN_PHRASE)
        user = prompts.ANSWER_USER.format(question=query, context_block=block)
        raw = self._complete(system, user).strip()
        if prompts.ABSTAIN_PHRASE in raw:
            return GeneratedAnswer(text="", cited_chunk_ids=[], abstained=True)
        cited = re.findall(r"\[([A-Za-z0-9_\-]+)\]", raw)
        clean = re.sub(r"\[[A-Za-z0-9_\-]+\]", "", raw).strip()
        return GeneratedAnswer(text=clean, cited_chunk_ids=list(dict.fromkeys(cited)))

    def extract_claims(self, answer: str) -> list[str]:
        raw = self._complete(prompts.CLAIMS_SYSTEM, answer)
        try:
            data = json.loads(_extract_json(raw))
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception:
            pass
        return split_sentences(answer)

    def judge_claim(self, claim: str, context: str) -> tuple[bool, str, float]:
        user = prompts.JUDGE_USER.format(context=context, claim=claim)
        raw = self._complete(prompts.JUDGE_SYSTEM, user)
        try:
            data = json.loads(_extract_json(raw))
            return (
                bool(data.get("supported", False)),
                str(data.get("reason", "")),
                float(data.get("confidence", 0.5)),
            )
        except Exception:
            supported = "true" in raw.lower()[:40]
            return supported, "parse-fallback", 0.5

    def reformulate(self, query: str, missing_info: str | None, prior: list[str]) -> str:
        user = prompts.REFORMULATE_USER.format(
            question=query, missing=missing_info or "n/a", prior="; ".join(prior) or "none"
        )
        return self._complete(prompts.REFORMULATE_SYSTEM, user).strip().strip('"') or query

    def classify_complexity(self, query: str) -> str:
        raw = self._complete(prompts.ROUTER_SYSTEM, query).strip().lower()
        return "complex" if "complex" in raw else "simple"

    def hypothetical_document(self, query: str) -> str:
        system = (
            "Write a short, factual passage (2-3 sentences) that would directly "
            "answer the question, as if from documentation. Do not hedge."
        )
        return self._complete(system, query).strip() or query

    def decompose(self, query: str) -> list[str]:
        system = (
            "If the question requires multiple facts, break it into 2-4 atomic "
            "sub-questions. Return ONLY a JSON array of strings. If it is already "
            "atomic, return a single-element array."
        )
        raw = self._complete(system, query)
        try:
            data = json.loads(_extract_json(raw))
            subs = [str(x).strip() for x in data if str(x).strip()]
            return subs or [query]
        except Exception:
            return [query]


def _extract_json(text: str) -> str:
    """Grab the first JSON object/array from a possibly chatty response."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return m.group(1) if m else text


def make_llm(cfg) -> LanguageModel:
    provider = cfg.get("llm.provider", "mock")
    if provider == "mock":
        return MockLLM()
    return PromptLLM(cfg)
