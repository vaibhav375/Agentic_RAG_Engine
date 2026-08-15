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
def _context_block(contexts: list[tuple[str, str]]) -> str:
    """Render passages so the boundary between them is unambiguous.

    The prompt asks the model to copy each passage's id "exactly as it appears in
    brackets at the start of that passage", which only works if the model can see
    where a passage starts. Passages used to be joined with a single newline,
    which was fine while every chunk was one paragraph.

    Once the chunker began packing several blocks into a chunk, chunk text carried
    its own blank lines — so the separation *inside* a passage was stronger than
    the separation *between* passages. Internal blank lines are therefore
    collapsed for the prompt only (stored chunk text keeps its paragraphs for
    display and citation quoting), and passages are separated by the blank line
    that is now unique to a boundary.

    Worth being straight about what this fixed: it was written to explain why
    packed chunks lost citations on six gold questions, and it recovered exactly
    one of them. The dominant cause is elsewhere — with packed (longer) context
    the 3B model writes a longer answer and silently drops the citation
    instruction. In that run the two answers that cited averaged 12 words and the
    four that did not averaged 46. This is the same instruction-crowding failure
    already recorded in docs/local-mode-eval.md, arriving via context length
    rather than via a longer prompt. Unambiguous boundaries are still correct;
    they are just not the cure.
    """
    out = []
    for cid, text in contexts:
        body = re.sub(r"\n\s*\n+", "\n", (text or "").strip())
        out.append(f"[{cid}] {body}")
    return "\n\n".join(out)


def parse_citations(raw: str, known_ids: list[str]) -> GeneratedAnswer:
    """Pull `[chunk_id]` citations out of a model answer and strip those markers.

    Matching is anchored on the ids actually supplied as context rather than on a
    character-class guess. Both halves of that mattered in practice:

    * Chunk ids contain `::` (`01_routing::2`), which the previous
      `[A-Za-z0-9_\\-]+` pattern could not match — so with real models no citation
      was ever parsed, `citation_precision` scored 0.0 for every answered record,
      and the raw `[01_routing::2]` markers stayed in the answer text where they
      polluted the claims handed to the NLI metric. Mock mode never showed it
      because `MockLLM` builds citations directly instead of parsing them.
    * A permissive "anything in brackets" pattern is not the fix: this corpus
      contains `list[str]`, and eating that would corrupt correct answers.

    Models also imitate the prompt's `[c3]` example and sometimes emit
    `[c01_routing::2]`, so a stray leading `c` is tolerated.
    """
    lookup = {cid.lower(): cid for cid in known_ids}
    cited: list[str] = []
    spans: list[tuple[int, int]] = []

    for m in re.finditer(r"\[([^\[\]\n]{1,120})\]", raw):
        token = m.group(1).strip()
        resolved = lookup.get(token.lower())
        if resolved is None and token[:1].lower() == "c":
            resolved = lookup.get(token[1:].strip().lower())
        if resolved is None:
            continue  # not a citation (e.g. `list[str]`) — leave it in the text
        cited.append(resolved)
        spans.append(m.span())

    out = []
    last = 0
    for start, end in spans:
        out.append(raw[last:start])
        last = end
    out.append(raw[last:])
    clean = "".join(out)
    # Tidy the punctuation left behind by removing a trailing marker.
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\s+([.,;:])", r"\1", clean).strip()
    return GeneratedAnswer(text=clean, cited_chunk_ids=list(dict.fromkeys(cited)))


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
        # Any OpenAI-compatible host (Moonshot/Kimi, OpenRouter, Together, Groq,
        # vLLM). Empty base_url = api.openai.com.
        self.base_url = llm.get("base_url") or os.environ.get("ARAG_LLM_BASE_URL", "")
        self.api_key_env = llm.get("api_key_env") or "OPENAI_API_KEY"
        # Local models are far slower than hosted ones, and the self-correction
        # loop issues many calls per query, so this needs to be tunable.
        self.timeout_seconds = float(llm.get("timeout_seconds", 120))
        # None = don't send the field at all (older Ollama servers reject it).
        self.think = llm.get("think")
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
        """OpenAI, and every OpenAI-compatible endpoint.

        `llm.base_url` + `llm.api_key_env` point this at any provider speaking
        the same protocol — Moonshot (Kimi), OpenRouter, Together, Groq, or a
        local vLLM/llama.cpp server — so open-weight models too large to run on
        this machine are reachable without a new backend.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get(self.api_key_env) or "not-needed",
                base_url=self.base_url or None,
            )
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
        payload: dict = {
            "model": self.ollama_model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                # Ollama calls this num_predict. Without it the model is
                # uncapped: a reasoning model will happily narrate past any
                # timeout, which is exactly how the first local run died.
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Reasoning models (qwen3, deepseek-r1, …) emit a chain of thought that
        # would land in the answer text and wreck citation parsing and token-F1.
        # Ollama >= 0.9 lets us switch it off per request; older servers 400 on
        # the field, so only send it when explicitly configured.
        if self.think is not None:
            payload["think"] = self.think
        r = httpx.post(f"{base}/api/chat", json=payload, timeout=self.timeout_seconds)
        r.raise_for_status()
        message = r.json()["message"]
        # When thinking is on, Ollama returns it separately — never concatenate it.
        return message.get("content", "")

    # -- tasks -------------------------------------------------------------- #
    def generate_answer(self, query: str, contexts: list[tuple[str, str]]) -> GeneratedAnswer:
        block = _context_block(contexts)
        system = prompts.ANSWER_SYSTEM.format(abstain=prompts.ABSTAIN_PHRASE)
        user = prompts.ANSWER_USER.format(question=query, context_block=block)
        raw = self._complete(system, user).strip()
        if prompts.ABSTAIN_PHRASE in raw:
            return GeneratedAnswer(text="", cited_chunk_ids=[], abstained=True)
        return parse_citations(raw, [cid for cid, _ in contexts])

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


# Which config field holds the model id, per provider — so a role override knows
# what to replace.
_MODEL_FIELD = {
    "openai": "llm.model",
    "anthropic": "llm.anthropic_model",
    "ollama": "llm.ollama_model",
}


def make_llm(cfg, role: str | None = None) -> LanguageModel:
    """Build the LLM for a role: None (generation), "judge", or "router".

    `llm.judge_model` / `llm.router_model` let the critic and the router run on a
    different model than the generator. That matters in both directions:
      * a stronger judge than generator — measured on `llama3.2:3b`, a 3B critic
        mis-scores correct-but-elaborated answers, which showed up as a 31%
        hallucination rate and 33% over-abstention that were both judge error
        (see docs/local-mode-eval.md);
      * a *different* judge than generator, which reduces the LLM-as-judge
        self-preference bias the calibration section warns about.
    Falls back to the generation model when the role's field is unset.
    """
    provider = cfg.get("llm.provider", "mock")
    if provider == "mock":
        return MockLLM()
    if role:
        override = cfg.get(f"llm.{role}_model")
        field = _MODEL_FIELD.get(provider)
        if override and field:
            cfg = cfg.with_overrides({field: override})
    return PromptLLM(cfg)
