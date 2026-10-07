"""TracInCP - checkpoint-summed first-order gradient influence (strong baseline).

# Implements: strong cross-step gradient baseline for C1.

TracInCP estimates the influence of a training rollout `z` on a target signal
by summing, over saved checkpoints, the learning-rate-weighted inner product
between the target gradient *at that checkpoint* and the rollout's gradient
*at that checkpoint*:

    TracInCP(z) = Σ_t  η_t · ⟨ g_target(θ_t) , g_z(θ_t) ⟩

This is the honest cross-step gradient competitor to C1. The key difference
from C1 (`attribution.cross_step`): C1 uses a **single** behavior-target
signature from the *final* checkpoint and correlates every step's rollout
gradient against it; TracInCP uses the target gradient **recomputed at each
checkpoint**, so it tracks how the behavior-relevant direction moves during
training, and it weights each step by the learning rate actually applied.

Sanity property (unit-tested): with exactly one checkpoint and η=1, TracInCP
reduces to C1's dot-product aggregation - so any gap between them in E2.1/E1.3
is attributable to the cross-checkpoint information, not to a different scale.

Cost: TracInCP needs `g_target(θ_t)` at each checkpoint, i.e. one target
forward/backward per checkpoint. The Colab/RunPod notebook produces the
`targets_by_checkpoint` mapping by loading each checkpoint and differentiating
the behavior functional; this pure-python core is checkpoint-agnostic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from ...instrumentation.interfaces import GradientEmbedding
from ..local_buffer import AttributionScore


def _governing_checkpoint(step: int, checkpoint_steps: list[int]) -> int:
    """The checkpoint whose parameters produced a rollout sampled at `step`:
    the greatest checkpoint step ≤ `step`, or the earliest checkpoint if the
    rollout predates all of them (`checkpoint_steps` is sorted ascending)."""
    gov = checkpoint_steps[0]
    for cs in checkpoint_steps:
        if cs <= step:
            gov = cs
        else:
            break
    return gov


def tracin_attribute(
    targets_by_checkpoint: Mapping[int, np.ndarray],
    embeddings: Iterable[GradientEmbedding],
    *,
    lr_by_checkpoint: Mapping[int, float] | float = 1.0,
    eps: float = 1e-12,
) -> list[AttributionScore]:
    """Rank rollouts by TracInCP influence on the per-checkpoint targets.

    Args:
        targets_by_checkpoint: `{checkpoint_step: target_vector}` - the behavior
            target gradient at each saved checkpoint. Each vector must match the
            embedding dimension. At least one entry required.
        embeddings: `GradientEmbedding`s spanning the training history. Each is
            attributed to the checkpoint governing its `step`
            (`_governing_checkpoint`). Order does not matter.
        lr_by_checkpoint: learning rate η_t per checkpoint (mapping) or a single
            scalar applied to all checkpoints. Missing checkpoints default to 1.0.
        eps: numerical floor (unused for dot but reserved for parity with the
            cosine-based estimators).

    Returns:
        One `AttributionScore` per rollout_id, sorted descending by score.
        `step` carries the latest step the rollout appeared at (tie-break),
        matching `cross_step`/`local_buffer` conventions.
    """
    ckpt_steps = sorted(targets_by_checkpoint)
    if not ckpt_steps:
        raise ValueError("targets_by_checkpoint must have at least one entry")
    dim = np.asarray(next(iter(targets_by_checkpoint.values()))).shape
    if len(dim) != 1:
        raise ValueError(f"target vectors must be 1-D, got shape {dim}")

    def _lr(cs: int) -> float:
        if isinstance(lr_by_checkpoint, Mapping):
            return float(lr_by_checkpoint.get(cs, 1.0))
        return float(lr_by_checkpoint)

    sums: dict[str, float] = defaultdict(float)
    latest_step: dict[str, int] = {}
    for emb in embeddings:
        cs = _governing_checkpoint(emb.step, ckpt_steps)
        target = np.asarray(targets_by_checkpoint[cs])
        if emb.vector.shape != target.shape:
            raise ValueError(
                f"embedding/target shape mismatch at checkpoint {cs}: "
                f"emb={emb.vector.shape}, target={target.shape}"
            )
        sums[emb.rollout_id] += _lr(cs) * float(np.dot(emb.vector, target))
        if emb.step > latest_step.get(emb.rollout_id, -1):
            latest_step[emb.rollout_id] = emb.step

    out = [
        AttributionScore(rollout_id=rid, step=latest_step[rid], score=score)
        for rid, score in sums.items()
    ]
    out.sort(key=lambda r: (-r.score, r.step, r.rollout_id))
    return out
