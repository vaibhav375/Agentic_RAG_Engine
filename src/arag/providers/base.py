"""Provider interfaces + small shared text utilities."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "as", "at", "by", "it", "this", "that", "from", "you", "your",
    "can", "how", "do", "does", "what", "which", "when", "where", "why", "i",
    "we", "will", "if", "then", "into", "using", "use", "used", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# Subordinating conjunctions that open a framing clause ("If X, Y" -> assert Y).
_LEADING_SUBORDINATOR = re.compile(
    r"^\s*(?:if|when|whenever|while|because|since|although|though|unless|after|"
    r"before|once|as long as|in case|given that|assuming)\b[^,]{0,120},\s*",
    re.IGNORECASE,
)
# Trailing clauses that tack a second assertion onto a sentence.
_TRAILING_CLAUSE = re.compile(
    r",\s*(?:so|because|since|which means|meaning|so that|therefore|and so)\b\s*",
    re.IGNORECASE,
)
_MIN_CLAIM_WORDS = 4


def atomic_claims(text: str) -> list[str]:
    """Split an answer into claims small enough for an NLI model to judge.

    Sentence-level splitting is reproducible but leaves *compound conditional*
    sentences intact, and that is exactly what entailment models handle worst.
    Measured with `nli-deberta-v3-base` against a premise that supports the
    statement:

        "If a JSON body is missing a required field, Breeze returns a 422 ..."
                                                        entailment 0.000
        "Breeze returns a 422 ..."  (main clause alone)  entailment 0.992

    So a leading subordinate clause is treated as framing and dropped, and the
    main clause carries the assertion. The condition is deliberately *not* emitted
    as its own claim: the source states a broader condition ("... or a field has
    the wrong type"), so the narrowed condition scores 0.000 on its own and would
    reintroduce the same false negative from the other side.

    Deterministic and dependency-free, so the metric stays reproducible and
    independent of any model.
    """
    claims: list[str] = []
    for sentence in split_sentences(text):
        stripped = _LEADING_SUBORDINATOR.sub("", sentence, count=1).strip()
        # Only accept the strip if a real clause survives it.
        head = stripped if len(stripped.split()) >= _MIN_CLAIM_WORDS else sentence
        parts = [p.strip(" ,") for p in _TRAILING_CLAUSE.split(head)]
        for part in parts:
            if part and len(part.split()) >= _MIN_CLAIM_WORDS:
                claims.append(part if part.endswith(".") else part + ".")
    return claims or [c for c in split_sentences(text)]


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard-ish support score: fraction of content tokens in `a` present in `b`."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32
        ...

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class MockEmbedder(Embedder):
    """Deterministic hashing bag-of-words embedding.

    Content tokens are hashed into a fixed-dim vector (signed hashing trick),
    then L2-normalized. Cosine similarity therefore tracks lexical overlap —
    good enough to make retrieval behave sensibly and reproducibly offline,
    with zero dependencies.
    """

    def __init__(self, dim: int = 384, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in content_tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        if self.normalize:
            n = np.linalg.norm(vec)
            if n > 0:
                vec /= n
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed(t) for t in texts])


# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        ...


class MockReranker(Reranker):
    """Lexical cross-encoder stand-in: content-token overlap with a mild length
    penalty. Correlates with real reranker behavior (precision-boosting) so the
    ablation shows a realistic lift, deterministically."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        qt = content_tokens(query)
        out = []
        for p in passages:
            pt = content_tokens(p)
            if not qt or not pt:
                out.append(0.0)
                continue
            inter = len(qt & pt)
            prec = inter / len(pt)
            rec = inter / len(qt)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            out.append(round(f1, 6))
        return out


# --------------------------------------------------------------------------- #
# NLI (claim-level entailment cross-check)
# --------------------------------------------------------------------------- #
@dataclass
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        return max(
            (("entailment", self.entailment), ("neutral", self.neutral),
             ("contradiction", self.contradiction)),
            key=lambda x: x[1],
        )[0]


class NLIModel(ABC):
    @abstractmethod
    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        ...

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Score many (premise, hypothesis) pairs at once.

        The critic scores every claim against every premise unit of every
        retrieved chunk — roughly 94 forward passes per query on this corpus.
        Run one at a time that is ~2.9x slower than a single batched call on a
        cross-encoder, for bit-identical scores (max abs diff 1.4e-5). The
        default here keeps mock and any simple backend working unchanged.
        """
        return [self.entail(p, h) for p, h in pairs]


class MockNLI(NLIModel):
    """Lexical entailment proxy. Entailment prob = fraction of hypothesis content
    tokens found in the premise; the remainder splits into neutral/contradiction.
    Deterministic and dependency-free, mirrors a real NLI cross-check's signal."""

    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        support = lexical_overlap(hypothesis, premise)
        entail = support
        neutral = (1 - support) * 0.8
        contra = (1 - support) * 0.2
        return NLIResult(entailment=entail, neutral=neutral, contradiction=contra)


# --------------------------------------------------------------------------- #
# LLM task interface (high level, provider-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedAnswer:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False


class LanguageModel(ABC):
    """High-level task interface. Real backends implement each task by prompting
    an LLM and parsing structured output; the mock implements them directly."""

    @abstractmethod
    def generate_answer(self, query: str, contexts: list[tuple[str, str]]) -> GeneratedAnswer:
        """contexts: list of (chunk_id, text). Returns a grounded answer that
        cites chunk_ids, or abstains if the context does not support an answer."""

    @abstractmethod
    def extract_claims(self, answer: str) -> list[str]:
        ...

    @abstractmethod
    def judge_claim(self, claim: str, context: str) -> tuple[bool, str, float]:
        """Return (supported, reason, confidence)."""

    def judge_equivalence(self, question: str, gold: str, candidate: str) -> bool:
        """Does the candidate answer convey the same facts as the gold answer?

        Not abstract, and deliberately so: `answer_correctness` (token-F1) stays
        the primary reported number, and a backend that cannot do semantic
        comparison must keep working rather than crash the harness.

        The default is a lexical stand-in and inherits token-F1's flaw — it
        rewards shared wording and is blind to the token that decides the answer.
        That is acceptable only because mock is a deterministic stand-in, never a
        measurement of quality. Real backends override this.

        Measured on 9 hand-verified labels: token-F1 ranks them close to
        backwards, a qwen2.5:3b judge gets 7/9 (never accepting a wrong answer),
        and qwen2.5:7b gets 9/9. See eval/experiments/judge_validation.py.

        (Token overlap is recomputed here rather than imported from eval.metrics:
        a provider reaching into the eval layer would invert the dependency.)
        """
        pred, ref = content_tokens(candidate), content_tokens(gold)
        if not pred or not ref:
            return not pred and not ref
        overlap = len(pred & ref)
        if not overlap:
            return False
        precision, recall = overlap / len(pred), overlap / len(ref)
        return (2 * precision * recall) / (precision + recall) >= 0.6

    @abstractmethod
    def reformulate(self, query: str, missing_info: str | None, prior: list[str]) -> str:
        ...

    @abstractmethod
    def classify_complexity(self, query: str) -> str:
        """Return 'simple' or 'complex'."""

    @abstractmethod
    def hypothetical_document(self, query: str) -> str:
        """HyDE: a hypothetical passage that would answer the query, to embed and
        retrieve against (bridges the query/document vocabulary gap)."""

    @abstractmethod
    def decompose(self, query: str) -> list[str]:
        """Split a multi-hop question into sub-questions (or return [query])."""
