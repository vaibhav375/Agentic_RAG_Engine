"""NLI backends: mock (lexical entailment) and a real cross-encoder NLI model.

The NLI cross-check gives the self-correction critic a second, non-LLM signal
for claim support — mitigating LLM-judge self-preference bias (a known failure
mode of LLM-as-judge)."""

from __future__ import annotations

from arag.providers.base import MockNLI, NLIModel, NLIResult


def _free_accelerator_cache() -> None:
    """Return cached blocks before retrying a batch that just failed to allocate.

    Best-effort: a backend without the call, or without an accelerator at all,
    simply has nothing to free.
    """
    try:
        import torch
    except ImportError:
        return
    for backend in (getattr(torch, "mps", None), getattr(torch, "cuda", None)):
        empty = getattr(backend, "empty_cache", None)
        if callable(empty):
            try:
                empty()
            except Exception:
                pass


class CrossEncoderNLI(NLIModel):
    def __init__(self, model: str, batch_size: int = 8):
        from sentence_transformers import CrossEncoder

        # cross-encoder/nli-* models output logits over [contradiction, entail, neutral]
        self._model = CrossEncoder(model)
        self._batch_size = max(1, int(batch_size))

    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        return self.entail_batch([(premise, hypothesis)])[0]

    def _predict(self, pairs: list[tuple[str, str]]):
        """Score pairs under a memory cap, halving the batch on exhaustion.

        The critic scores every claim against every premise unit of every
        retrieved chunk, so `pairs` grows as claims x premises — a long answer
        over ten chunks is several hundred pairs. DeBERTa's disentangled attention
        allocates on the order of batch x heads x seq^2, and on this machine
        Ollama holds ~3.4 GiB of the same unified memory, so `predict`'s default
        32-pair batch exhausted MPS partway through a run and killed it:

            MPS backend out of memory (allocated 5.51 GiB, other allocations
            3.41 GiB, max allowed 9.07 GiB)

        Losing a 20-minute eval arm to an allocator is a bad trade for throughput
        that a smaller batch barely changes, and the same call runs in the serving
        path where a crash is worse still. Batching is score-neutral (bit-identical
        to 1.4e-5), so capping and retrying costs nothing but wall time.

        PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 would also silence the error, by
        removing the limit that prevents a system-level failure. Not that.
        """
        import numpy as np

        bs = self._batch_size
        while True:
            try:
                return np.asarray(
                    self._model.predict(pairs, batch_size=bs, show_progress_bar=False)
                )
            except RuntimeError as exc:
                if bs <= 1 or "out of memory" not in str(exc).lower():
                    raise
                bs = max(1, bs // 2)
                _free_accelerator_cache()

    def entail_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """One forward pass per batch — ~2.9x faster than looping, and
        bit-identical (verified to 1.4e-5)."""
        import numpy as np

        if not pairs:
            return []
        logits = self._predict(list(pairs))
        if logits.ndim == 1:  # a single pair can come back un-nested
            logits = logits[None, :]
        out: list[NLIResult] = []
        for row in logits:
            probs = np.exp(row - row.max())
            probs = probs / probs.sum()
            # Label order is contradiction, entailment, neutral for the
            # cross-encoder/nli-deberta-v3-* family — verified against
            # model.config.id2label for both the base and large checkpoints.
            # A model with a different ordering would silently invert
            # entailment and contradiction, so check before swapping one in.
            out.append(NLIResult(entailment=float(probs[1]), neutral=float(probs[2]),
                                 contradiction=float(probs[0])))
        return out


def make_nli(cfg) -> NLIModel:
    provider = cfg.get("mode", "mock")
    if provider in {"local", "api"}:
        return CrossEncoderNLI(
            cfg.get("agent.nli_model", "cross-encoder/nli-deberta-v3-base"),
            batch_size=int(cfg.get("agent.nli_batch_size", 8)),
        )
    return MockNLI()
