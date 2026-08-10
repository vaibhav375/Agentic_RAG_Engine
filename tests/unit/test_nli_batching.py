"""A failed allocation must cost wall time, not the run.

The critic scores claims x premise units, so `pairs` grows with answer length and
context count. sentence-transformers' default 32-pair batch exhausted MPS partway
through a 20-minute eval arm — deberta's disentangled attention allocates roughly
batch x heads x seq^2, and Ollama holds several GiB of the same unified memory:

    MPS backend out of memory (allocated 5.51 GiB, other 3.41 GiB, max 9.07 GiB)

The same call runs in the serving path, where dying is worse than being slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from arag.providers.nli import CrossEncoderNLI


class _FakeCrossEncoder:
    """Raises OOM above `fits`, mimicking the allocator's behaviour."""

    def __init__(self, fits: int):
        self.fits = fits
        self.batch_sizes: list[int] = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.batch_sizes.append(batch_size)
        if batch_size > self.fits:
            raise RuntimeError(
                "MPS backend out of memory (MPS allocated: 5.51 GiB, other "
                "allocations: 3.41 GiB, max allowed: 9.07 GiB)."
            )
        # logits over [contradiction, entailment, neutral]
        return np.array([[0.0, 5.0, 0.0]] * len(pairs))


def _nli(fits: int, batch_size: int = 8) -> tuple[CrossEncoderNLI, _FakeCrossEncoder]:
    nli = CrossEncoderNLI.__new__(CrossEncoderNLI)  # bypass model download
    fake = _FakeCrossEncoder(fits)
    nli._model = fake
    nli._batch_size = batch_size
    return nli, fake


def test_batch_halves_until_it_fits_and_still_returns_every_pair():
    nli, fake = _nli(fits=4, batch_size=8)
    pairs = [("premise text", "claim text")] * 6

    out = nli.entail_batch(pairs)

    assert fake.batch_sizes == [8, 4]  # one failure, then halved
    assert len(out) == len(pairs)
    assert out[0].entailment > 0.9


def test_the_configured_batch_size_is_used_when_it_fits():
    nli, fake = _nli(fits=64, batch_size=8)
    nli.entail_batch([("p", "c")] * 3)
    assert fake.batch_sizes == [8]


def test_it_gives_up_at_one_pair_rather_than_looping_forever():
    nli, fake = _nli(fits=0, batch_size=4)
    with pytest.raises(RuntimeError, match="out of memory"):
        nli.entail_batch([("p", "c")])
    assert fake.batch_sizes == [4, 2, 1]


def test_errors_that_are_not_allocation_failures_are_not_retried():
    """Retrying a genuine bug would hide it behind a slower repeat of itself."""
    nli, fake = _nli(fits=64, batch_size=8)

    def boom(pairs, batch_size=32, show_progress_bar=False):
        fake.batch_sizes.append(batch_size)
        raise RuntimeError("label order mismatch")

    fake.predict = boom
    with pytest.raises(RuntimeError, match="label order"):
        nli.entail_batch([("p", "c")])
    assert fake.batch_sizes == [8]  # tried once, not halved


def test_batch_size_does_not_change_scores():
    """Batching is an optimization; a different batch size must not move a metric."""
    pairs = [("premise one", "claim one"), ("premise two", "claim two")] * 3
    a, _ = _nli(fits=64, batch_size=8)
    b, _ = _nli(fits=64, batch_size=1)
    assert [r.entailment for r in a.entail_batch(pairs)] == [
        r.entailment for r in b.entail_batch(pairs)
    ]


def test_empty_input_does_not_touch_the_model():
    nli, fake = _nli(fits=64)
    assert nli.entail_batch([]) == []
    assert fake.batch_sizes == []
