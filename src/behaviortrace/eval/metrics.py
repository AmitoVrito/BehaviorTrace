"""Evaluation metrics for attribution.


- precision@k / recall@k on injected ground truth (E1.3).
- LDS-style correlation between predicted influence and measured Δs_b
  (E1.3).
- Retrain-gap-closure (an earlier experiment) - fraction of full-retrain reduction the
  cheap intervention achieves.
- behavior_reduction - fraction reduction in s_b after an intervention.
  Targets for an earlier experiment reproduction: ~63% (filtering), ~78% (label-switch).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def precision_at_k(predicted_ranking: list[str], ground_truth: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top_k = predicted_ranking[:k]
    return sum(1 for x in top_k if x in ground_truth) / k


def precision_at_k_fractional(
    scored_ranking: list[tuple[str, float]], ground_truth: set[str], k: int
) -> float:
    """Tie-aware precision@k with fractional credit for the boundary tie group.

    Source: the pre-registered protocol. Plain `precision_at_k` slices `ranking[:k]`, so when the k
    cutoff lands inside a block of equal-scored items (e.g. all rollouts in a step share
    one gradient and tie), the result depends on arbitrary within-tie ordering. Here each
    score-group contributes GT credit proportional to the SLOTS it occupies in the top-k:
    a group fully inside top-k gives its full GT count; the boundary group gives
    `gt_in_group * slots_taken / group_size`. The result is independent of within-tie
    order, so it is reproducible and deterministic.

    Args:
        scored_ranking: `(rollout_id, score)` pairs. Need not be pre-sorted; higher score
            ranks first.
        ground_truth: the GT rollout-id set.
        k: cutoff (typically |GT|).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    items = sorted(scored_ranking, key=lambda t: -t[1])
    credit = 0.0
    filled = 0
    i = 0
    n = len(items)
    while filled < k and i < n:
        j = i
        while j < n and items[j][1] == items[i][1]:
            j += 1
        group = items[i:j]
        gsize = len(group)
        gt_in = sum(1 for rid, _ in group if rid in ground_truth)
        slots = min(gsize, k - filled)
        credit += gt_in * (slots / gsize)
        filled += slots
        i = j
    return credit / k


def precision_ceiling_fractional(
    scored_ranking: list[tuple[str, float]], ground_truth: set[str], k: int
) -> float:
    """Best achievable tie-aware precision@k: order the score-groups (tie blocks) by GT
    density, then accumulate with the same fractional-credit rule. Bounds
    any step-level ranker from above; report method precision as a fraction of this."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    from collections import defaultdict

    groups: dict[float, list[str]] = defaultdict(list)
    for rid, score in scored_ranking:
        groups[score].append(rid)
    blocks = sorted(
        groups.values(), key=lambda g: -(sum(r in ground_truth for r in g) / len(g))
    )
    credit = 0.0
    filled = 0
    for g in blocks:
        if filled >= k:
            break
        gsize = len(g)
        gt_in = sum(1 for r in g if r in ground_truth)
        slots = min(gsize, k - filled)
        credit += gt_in * (slots / gsize)
        filled += slots
    return credit / k


def recall_at_k(predicted_ranking: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        raise ValueError("ground_truth set is empty")
    top_k = set(predicted_ranking[:k])
    return len(top_k & ground_truth) / len(ground_truth)


def base_rate(ground_truth_size: int, population_size: int) -> float:
    """Expected R-precision of a random ranker = |GT| / N.

    The E1.3 primary metric is precision@k with k=|GT| (R-precision, where
    precision@|GT| == recall@|GT|). A uniformly random ranking scores this
    base rate in expectation, so it is the honest floor any method must beat
    (§an earlier experiment). When the contamination fraction is high (e.g. 0.5), the base rate
    is ~0.5 and R-precision near 0.5 is NOT evidence of attribution quality -
    see . Report `precision_lift_over_chance` alongside raw precision.
    """
    if population_size <= 0:
        raise ValueError(f"population_size must be positive, got {population_size}")
    if not 0 <= ground_truth_size <= population_size:
        raise ValueError(
            f"ground_truth_size must be in [0, population_size], got "
            f"{ground_truth_size} of {population_size}"
        )
    return ground_truth_size / population_size


def precision_lift_over_chance(precision: float, floor: float) -> float:
    """How many times better than chance a precision is: `precision / floor`.

    `floor` is the random baseline's precision (empirically) or the base rate
    (analytically) - see `base_rate`. Lift ≈ 1.0 means "no better than a coin
    flip" even if `precision` looks high in absolute terms. Raises if the
    floor is non-positive (a floor of 0 means every method is trivially
    infinite-lift, which is meaningless - use the empirical random baseline).
    """
    if floor <= 0:
        raise ValueError(
            f"floor must be > 0 to compute lift (got {floor}); a zero floor "
            "usually means k >> population base rate - check the metric setup"
        )
    return precision / floor


def lds_correlation(predicted_influence: np.ndarray, measured_delta_sb: np.ndarray) -> float:
    """LDS-style Spearman correlation between predicted influence and measured Δs_b.

    The LDS (Linear Datamodeling Score) formulation. For E1.3 the predicted_influence is the attribution score
    of each candidate set, and measured_delta_sb is the actual change in
    behavior score after leave-trajectory-out re-runs (E1.2). Higher is
    better.

    Args:
        predicted_influence: shape (N,) - predicted score per item.
        measured_delta_sb: shape (N,) - measured Δs_b after LTO re-run.

    Returns:
        Spearman ρ ∈ [-1, 1].
    """
    if predicted_influence.shape != measured_delta_sb.shape:
        raise ValueError(
            f"shape mismatch: predicted={predicted_influence.shape}, "
            f"measured={measured_delta_sb.shape}"
        )
    if predicted_influence.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {predicted_influence.shape}")
    if predicted_influence.size < 2:
        raise ValueError(f"need at least 2 items, got {predicted_influence.size}")
    result = stats.spearmanr(predicted_influence, measured_delta_sb)
    rho = float(result.statistic)
    if np.isnan(rho):
        # All-tied input - degenerate; report 0 rather than NaN.
        return 0.0
    return rho


def retrain_gap_closure(
    s_b_baseline: float, s_b_intervention: float, s_b_full_retrain: float
) -> float:
    """Fraction of the full-retrain reduction the cheap intervention closes.

    closure = (baseline - intervention) / (baseline - full_retrain)
    """
    denom = s_b_baseline - s_b_full_retrain
    if denom == 0:
        raise ValueError("baseline equals full-retrain; no gap to close")
    return (s_b_baseline - s_b_intervention) / denom


def behavior_reduction(s_b_before: float, s_b_after: float) -> float:
    """Fraction reduction in behavior rate after an intervention.

    Source: an earlier experiment reports filter / label-switch reductions in this form
    (Xiao & Aranguri ~63% / ~78%).

    Returns (s_b_before - s_b_after) / s_b_before. Raises if s_b_before is 0.
    Negative values mean the intervention *increased* the behavior.
    """
    if s_b_before <= 0:
        raise ValueError(
            f"s_b_before must be > 0 to compute reduction, got {s_b_before}"
        )
    return (s_b_before - s_b_after) / s_b_before
