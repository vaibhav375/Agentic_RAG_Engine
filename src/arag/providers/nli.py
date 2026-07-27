"""NLI backends: mock (lexical entailment) and a real cross-encoder NLI model.

The NLI cross-check gives the self-correction critic a second, non-LLM signal
for claim support — mitigating LLM-judge self-preference bias (a known failure
mode of LLM-as-judge)."""

from __future__ import annotations

from arag.providers.base import MockNLI, NLIModel, NLIResult


class CrossEncoderNLI(NLIModel):
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        # cross-encoder/nli-* models output logits over [contradiction, entail, neutral]
        self._model = CrossEncoder(model)

    def entail(self, premise: str, hypothesis: str) -> NLIResult:
        import numpy as np

        logits = self._model.predict([(premise, hypothesis)])
        arr = np.asarray(logits)[0]
        probs = np.exp(arr - arr.max())
        probs = probs / probs.sum()
        # Label order for cross-encoder/nli-deberta-v3-base: contradiction, entail, neutral
        contra, entail, neutral = float(probs[0]), float(probs[1]), float(probs[2])
        return NLIResult(entailment=entail, neutral=neutral, contradiction=contra)


def make_nli(cfg) -> NLIModel:
    provider = cfg.get("mode", "mock")
    if provider in {"local", "api"}:
        return CrossEncoderNLI(cfg.get("agent.nli_model", "cross-encoder/nli-deberta-v3-base"))
    return MockNLI()
