"""Real tests over the only currently-implemented module: eval.metrics."""

import pytest

from behaviortrace.eval.metrics import (
    base_rate,
    precision_at_k,
    precision_at_k_fractional,
    precision_ceiling_fractional,
    precision_lift_over_chance,
    recall_at_k,
    retrain_gap_closure,
)


def test_precision_fractional_order_independent_at_tie_boundary():
    # Two steps, each all-tied: step A (score 2.0) has 1 GT of 2; step B (score 1.0) has
    # 2 GT of 4. k=3: full A (2 slots, 1 GT) + boundary B (1 slot of 4, 2 GT) =
    # 1 + 2*(1/4) = 1.5; /3 = 0.5. Must not depend on within-tie order.
    gt = {"a1", "b1", "b2"}
    r1 = [("a1", 2.0), ("a2", 2.0), ("b1", 1.0), ("b2", 1.0), ("b3", 1.0), ("b4", 1.0)]
    r2 = [("a2", 2.0), ("a1", 2.0), ("b4", 1.0), ("b3", 1.0), ("b2", 1.0), ("b1", 1.0)]
    assert precision_at_k_fractional(r1, gt, 3) == pytest.approx(0.5)
    assert precision_at_k_fractional(r2, gt, 3) == pytest.approx(0.5)  # order-independent


def test_precision_fractional_no_ties_matches_plain():
    gt = {"a", "c"}
    scored = [("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0)]
    assert precision_at_k_fractional(scored, gt, 2) == pytest.approx(0.5)


def test_precision_ceiling_fractional_orders_by_density():
    # step A: 2 GT of 2 (density 1.0); step B: 1 GT of 2 (0.5). k=3: all A (2 GT) +
    # boundary B (1 slot of 2, 1 GT -> 0.5) = 2.5; /3 = 0.8333.
    gt = {"a1", "a2", "b1"}
    scored = [("a1", 0.0), ("a2", 0.0), ("b1", 1.0), ("b2", 1.0)]  # score == step id
    assert precision_ceiling_fractional(scored, gt, 3) == pytest.approx(2.5 / 3)


def test_base_rate_is_gt_over_population():
    # at fraction=0.5 the E1.3 base rate ≈ 0.428 - a coin flip's R-precision.
    assert base_rate(6849, 16000) == pytest.approx(0.428, abs=1e-3)
    assert base_rate(1272, 16000) == pytest.approx(0.0795, abs=1e-3)


def test_base_rate_rejects_bad_sizes():
    with pytest.raises(ValueError, match="population_size must be positive"):
        base_rate(1, 0)
    with pytest.raises(ValueError, match="ground_truth_size must be in"):
        base_rate(20, 10)


def test_precision_lift_over_chance():
    # E1.3 real run: C1 0.469 over base rate 0.428 → barely above chance.
    assert precision_lift_over_chance(0.469, 0.428) == pytest.approx(1.096, abs=1e-3)
    # E2.1: C1 0.420 over base rate 0.08 → genuine ~5.3× signal.
    assert precision_lift_over_chance(0.420, 0.0795) == pytest.approx(5.28, abs=1e-2)


def test_precision_lift_rejects_zero_floor():
    with pytest.raises(ValueError, match="floor must be > 0"):
        precision_lift_over_chance(0.5, 0.0)


def test_precision_at_k_basic():
    ranking = ["a", "b", "c", "d"]
    gt = {"a", "c"}
    assert precision_at_k(ranking, gt, k=2) == pytest.approx(0.5)
    assert precision_at_k(ranking, gt, k=4) == pytest.approx(0.5)


def test_precision_at_k_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)


def test_recall_at_k_basic():
    ranking = ["a", "b", "c", "d"]
    gt = {"a", "c"}
    assert recall_at_k(ranking, gt, k=4) == pytest.approx(1.0)
    assert recall_at_k(ranking, gt, k=1) == pytest.approx(0.5)


def test_recall_at_k_empty_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_retrain_gap_closure_full():
    # intervention matches full retrain → 100% closure
    assert retrain_gap_closure(1.0, 0.2, 0.2) == pytest.approx(1.0)


def test_retrain_gap_closure_partial():
    # half the gap closed
    assert retrain_gap_closure(1.0, 0.6, 0.2) == pytest.approx(0.5)


def test_retrain_gap_closure_no_gap():
    with pytest.raises(ValueError):
        retrain_gap_closure(0.5, 0.4, 0.5)
