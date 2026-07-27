from eval.metrics import (
    bootstrap_ci,
    mrr,
    precision_at_k,
    recall_at_k,
    token_f1,
)


def test_recall_and_precision_at_k():
    ranked = ["a", "b", "c", "d"]
    gold = ["c"]
    # c is within the top-4, so recall@4 == 1.0; precision@4 = 1/4.
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert precision_at_k(ranked, gold, 4) == 0.25
    # c is NOT in the top-2, so recall@2 == 0.0.
    assert recall_at_k(ranked, gold, 2) == 0.0


def test_recall_multi_gold():
    ranked = ["a", "b", "c"]
    gold = ["a", "z"]  # only 'a' retrievable
    assert recall_at_k(ranked, gold, 3) == 0.5


def test_mrr_uses_first_gold_rank():
    assert mrr(["x", "y", "gold"], ["gold"]) == round(1 / 3, 4)
    assert mrr(["gold", "y"], ["gold"]) == 1.0
    assert mrr(["a", "b"], ["gold"]) == 0.0


def test_token_f1_bounds():
    assert token_f1("blue widget", "blue widget") == 1.0
    assert token_f1("completely unrelated tokens", "blue widget color") == 0.0


def test_bootstrap_ci_is_deterministic_and_bracketing():
    vals = [1.0] * 3 + [0.0] * 7  # mean 0.3
    lo, hi = bootstrap_ci(vals, iters=500, seed=0)
    lo2, hi2 = bootstrap_ci(vals, iters=500, seed=0)
    assert (lo, hi) == (lo2, hi2)          # deterministic
    assert lo <= 0.3 <= hi                 # brackets the point estimate
