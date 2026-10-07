"""C1 - cross-step attribution estimator (GAS, renormalized TracInCP).

Given:
    - A behavior-target signal `target` at the final checkpoint
      (e.g., the gradient of s_b w.r.t. the policy parameters, or an
      activation-difference signature).
    - A history of `GradientEmbedding`s - per-step low-rank embeddings of
      the per-rollout gradients (captured by the pipeline on Colab via TRL
      hooks, or by the synthetic policy for local validation).

This module aggregates **per-step similarity into a per-rollout
cumulative score** over the *entire* training history (or any specified
subset). The aggregation distinguishes C1 from the Hu et al.
local-buffer baseline (which scores only the recent buffer window).

Aggregation choice: sum across step appearances. Each (rollout, step)
pair contributes its signed similarity to the target. A rollout that
participated in many positive-similarity updates accumulates a large
positive influence; conversely for negative. Mean and max aggregations
are alternatives but sum is the only one whose units stay consistent
with single-step similarity, which matters when comparing against
`local_buffer.attribute_local_buffer` head-to-head on the same data
(the local-buffer estimator is implicitly sum-over-its-window).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

import numpy as np

from ..instrumentation.interfaces import GradientEmbedding
from .local_buffer import AttributionScore

SimilarityMetric = Literal["cosine", "dot"]


def attribute_cross_step(
    target: np.ndarray,
    embeddings: Iterable[GradientEmbedding],
    *,
    metric: SimilarityMetric = "dot",
    eps: float = 1e-9,
    step_range: tuple[int, int] | None = None,
) -> list[AttributionScore]:
    """C1 - rank rollouts by their cumulative similarity to `target`.

    Args:
        target: Behavior-target signal vector at the final checkpoint -
            same dimension as each embedding.
        embeddings: Iterable of `GradientEmbedding`s spanning the
            training history. Order does NOT matter; aggregation is
            commutative.
        metric: "dot" (default for C1 because it preserves magnitude
            information across step appearances; sum-of-dot-products is
            the natural cumulative influence) or "cosine" (normalizes
            per-step, useful when individual-step magnitudes are noisy).
        eps: Numerical floor for cosine denominator.
        step_range: Optional `(lo, hi)` - only embeddings with `lo ≤ step < hi`
            are considered. Used by the
            E2.1 head-to-head to define "local window" as the same code
            path with a tight `step_range` rather than a different
            function.

    Returns:
        `AttributionScore` per unique rollout_id, sorted desc by score.
        The `step` field of each score carries the LATEST step that
        rollout participated in (tie-breaks during sort and is useful
        downstream for filtering by recency).
    """
    if target.ndim != 1:
        raise ValueError(f"target must be 1-D, got shape {target.shape}")
    if metric not in {"cosine", "dot"}:
        raise ValueError(f"unknown metric: {metric}")
    t_norm = float(np.linalg.norm(target))
    if metric == "cosine" and t_norm < eps:
        raise ValueError("target signal has near-zero norm; cosine ill-defined")

    if step_range is not None:
        lo, hi = step_range
        if lo > hi:
            raise ValueError(f"step_range lo must be ≤ hi, got ({lo}, {hi})")

    sums: dict[str, float] = defaultdict(float)
    latest_step: dict[str, int] = {}
    for emb in embeddings:
        if step_range is not None:
            lo, hi = step_range
            if emb.step < lo or emb.step >= hi:
                continue
        if emb.vector.shape != target.shape:
            raise ValueError(
                f"embedding/target shape mismatch: emb={emb.vector.shape}, "
                f"target={target.shape}"
            )
        dot = float(np.dot(emb.vector, target))
        if metric == "cosine":
            v_norm = float(np.linalg.norm(emb.vector))
            sim = dot / max(v_norm * t_norm, eps)
        else:
            sim = dot
        sums[emb.rollout_id] += sim
        prev = latest_step.get(emb.rollout_id, -1)
        if emb.step > prev:
            latest_step[emb.rollout_id] = emb.step

    out = [
        AttributionScore(rollout_id=rid, step=latest_step[rid], score=score)
        for rid, score in sums.items()
    ]
    out.sort(key=lambda r: (-r.score, r.step, r.rollout_id))
    return out
