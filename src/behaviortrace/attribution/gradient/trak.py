"""TRAK - whitened gradient-embedding attribution (strong baseline).

# Implements: strong cross-step gradient baseline for C1.

TRAK's core estimator whitens the projected training-gradient matrix by the
inverse gram matrix before taking the inner product with the target:

    score(z) = φ(target)ᵀ (ΦᵀΦ + λI)⁻¹ φ(z)          (× optional Q_z)

where Φ (R×d) stacks the projected per-rollout gradient embeddings. The
`(ΦᵀΦ + λI)⁻¹` term is exactly what raw dot-product attribution (C1,
`attribution.cross_step`) lacks: it decorrelates the gradient features, so a
rollout is not credited merely for lying along a globally high-variance
direction. That whitening is the reason TRAK is a genuinely *distinct* strong
baseline rather than a rescaling of C1 - which is the point of .

Unlike TracInCP, this runs entirely on the **already-captured** npz embeddings
(no new training): it is linear algebra over Φ and one target vector.

Adaptation note (honest, per R6): the classical TRAK `Q_z` weight is the
model's one-minus-correct-class probability, which is a classification concept.
In the online-RL/behavior setting there is no correct class, so `Q` defaults to
1. An optional `weights` mapping lets a caller supply an RL analogue (e.g. a
per-rollout advantage or reward-margin magnitude); this is documented as an
adaptation and reported as such, not passed off as canonical TRAK.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ...instrumentation.interfaces import GradientEmbedding
from ..local_buffer import AttributionScore


def trak_attribute(
    target: np.ndarray,
    embeddings: Iterable[GradientEmbedding],
    *,
    damping: float = 0.1,
    weights: Mapping[str, float] | None = None,
) -> list[AttributionScore]:
    """Rank rollouts by whitened TRAK influence on `target`.

    Args:
        target: behavior-target signature at the final checkpoint, shape (d,).
        embeddings: `GradientEmbedding`s. A rollout appearing at multiple steps
            is aggregated by **summing** its vectors first (consistent with
            C1's aggregation), giving one φ(z) per rollout before whitening.
        damping: ridge λ as a fraction of the mean gram diagonal:
            λ = damping · mean(diag(ΦᵀΦ)). Stabilises the inverse when the
            gram matrix is near-singular (low projected rank). Must be > 0.
        weights: optional `{rollout_id: Q_z}` RL analogue of TRAK's Q weight.
            Missing rollouts default to 1.0. Left as None for canonical TRAK.

    Returns:
        One `AttributionScore` per rollout_id, sorted descending by score.
        `step` carries the latest step the rollout appeared at (tie-break).
    """
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError(f"target must be 1-D, got shape {target.shape}")
    if damping <= 0:
        raise ValueError(f"damping must be > 0, got {damping}")

    # Aggregate to one vector per rollout (sum across step appearances).
    agg: dict[str, np.ndarray] = {}
    latest_step: dict[str, int] = {}
    for emb in embeddings:
        if emb.vector.shape != target.shape:
            raise ValueError(
                f"embedding/target shape mismatch: emb={emb.vector.shape}, "
                f"target={target.shape}"
            )
        v = np.asarray(emb.vector, dtype=np.float64)
        if emb.rollout_id in agg:
            agg[emb.rollout_id] = agg[emb.rollout_id] + v
        else:
            agg[emb.rollout_id] = v.copy()
        if emb.step > latest_step.get(emb.rollout_id, -1):
            latest_step[emb.rollout_id] = emb.step

    if not agg:
        return []

    rollout_ids = list(agg)
    phi = np.stack([agg[rid] for rid in rollout_ids])  # (R, d)
    n_roll, dim = phi.shape

    # ridge λ = damping · mean(diag(ΦᵀΦ)) = damping · ||Φ||_F² / d  (same either form).
    lam = damping * (float(np.sum(phi * phi)) / dim) if dim else 0.0
    if lam <= 0:  # all-zero embeddings → no signal; fall back to raw dot.
        raw_scores = phi @ target  # (R,)
    elif dim <= n_roll:
        # Primal: whiten the target with the (d×d) gram, w = (ΦᵀΦ + λI)⁻¹ target.
        gram = phi.T @ phi  # (d, d)
        w = np.linalg.solve(gram + lam * np.eye(dim), target)
        raw_scores = phi @ w  # (R,)
    else:
        # Dual (rollout-space): scores = (λI + ΦΦᵀ)⁻¹ (Φ target). Algebraically identical
        # to the primal (Woodbury), but the (R×R) solve stays feasible when d is large,
        # e.g. the 65536-d CountSketch space where a d×d gram would be ~34 GB.
        y = phi @ target  # (R,)
        gram = phi @ phi.T  # (R, R)
        raw_scores = np.linalg.solve(gram + lam * np.eye(n_roll), y)  # (R,)

    out = []
    for rid, score in zip(rollout_ids, raw_scores, strict=True):
        q = 1.0 if weights is None else float(weights.get(rid, 1.0))
        out.append(
            AttributionScore(
                rollout_id=rid, step=latest_step[rid], score=float(score) * q
            )
        )
    out.sort(key=lambda r: (-r.score, r.step, r.rollout_id))
    return out
