"""Hu et al. local-buffer online-RL attribution (baseline).


The Hu et al. construction in one paragraph:

    Given a checkpoint t and a target signal g_b (the gradient of a
    behavioral functional - action or return - at the current policy),
    rank each record in the *recent training buffer* by the
    cosine-similarity-like inner product between its per-record gradient
    and g_b. Records with high alignment are "responsible" for the
    target signal at this checkpoint.

This module is the buffer-window estimator: it does NOT aggregate across
checkpoints. That extension is `cross_step` (C1). Used as the central
baseline for E2.1 - the head-to-head against the cross-step estimator.

The algorithm operates on already-projected gradient embeddings (the pipeline
output). Behaviortrace splits the gradient capture (in
`backends.trl_backend`) from the projection (`gradient_embed`) from the
attribution (this file), so each piece is independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..instrumentation.interfaces import GradientEmbedding

SimilarityMetric = Literal["cosine", "dot"]


@dataclass(frozen=True)
class AttributionScore:
    """One row of an attribution result."""

    rollout_id: str
    step: int
    score: float


def attribute_local_buffer(
    target: np.ndarray,
    buffer: Iterable[GradientEmbedding],
    metric: SimilarityMetric = "cosine",
    eps: float = 1e-9,
) -> list[AttributionScore]:
    """Rank rollouts in the buffer by similarity to the target signal.

    Args:
        target: Behavior-target signal vector at the current checkpoint,
            same dimension as the gradient embeddings.
        buffer: Iterable of `GradientEmbedding`s - the *recent* training
            buffer (Hu et al.'s local window). Aggregation across
            checkpoints is intentionally NOT done here.
        metric: "cosine" follows the paper's gradient-similarity framing.
            "dot" is occasionally used as a faster proxy.
        eps: numerical floor for the cosine denominator.

    Returns:
        Attribution scores sorted descending. Ties broken by step then
        rollout_id for determinism.
    """
    if target.ndim != 1:
        raise ValueError(f"target must be 1-D, got shape {target.shape}")
    if metric not in {"cosine", "dot"}:
        raise ValueError(f"unknown metric: {metric}")
    t_norm = float(np.linalg.norm(target))
    if metric == "cosine" and t_norm < eps:
        raise ValueError("target signal has near-zero norm; cosine ill-defined")

    records: list[AttributionScore] = []
    for emb in buffer:
        if emb.vector.shape != target.shape:
            raise ValueError(
                f"embedding/target shape mismatch: emb={emb.vector.shape}, "
                f"target={target.shape}"
            )
        dot = float(np.dot(emb.vector, target))
        if metric == "cosine":
            v_norm = float(np.linalg.norm(emb.vector))
            score = dot / max(v_norm * t_norm, eps)
        else:
            score = dot
        records.append(
            AttributionScore(rollout_id=emb.rollout_id, step=emb.step, score=score)
        )
    records.sort(key=lambda r: (-r.score, r.step, r.rollout_id))
    return records


def select_local_buffer(
    embeddings: Iterable[GradientEmbedding],
    current_step: int,
    window_size: int,
) -> list[GradientEmbedding]:
    """Pick the most recent `window_size` embeddings strictly before `current_step`.

    Hu et al.'s "recent training buffer" is operationalized this way. The
    cross_step estimator (C1) deliberately uses a much larger or unbounded
    window - that contrast is the cross-step comparison in E2.1.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    recent = [e for e in embeddings if e.step < current_step]
    recent.sort(key=lambda e: e.step, reverse=True)
    return recent[:window_size]


def top_k(records: list[AttributionScore], k: int) -> list[AttributionScore]:
    """Convenience: return the top-k attributed rollouts (assumes records is sorted)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return records[:k]
