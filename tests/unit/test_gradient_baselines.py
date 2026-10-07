"""Strong gradient baselines: TracInCP + TRAK.

These are the competitors C1 must beat (E2.1/E1.3/an earlier experiment). Tests pin the
contracts and the two claims that make them fair, distinct baselines:
  - TracInCP with one checkpoint and η=1 reduces to C1's dot-sum (so any
    gap in the experiments comes from cross-checkpoint info, not scaling).
  - TRAK's whitening changes the ranking vs raw dot-product (it is not a
    rescaling of C1).
"""

import numpy as np
import pytest

from behaviortrace.attribution.cross_step import attribute_cross_step
from behaviortrace.attribution.gradient.tracin import tracin_attribute
from behaviortrace.attribution.gradient.trak import trak_attribute
from behaviortrace.instrumentation.interfaces import GradientEmbedding


def _emb(rid, step, vec):
    return GradientEmbedding(
        rollout_id=rid, step=step, vector=np.asarray(vec, dtype=np.float32)
    )


# --------------------------------------------------------------------------- #
# TracInCP
# --------------------------------------------------------------------------- #

def test_tracin_requires_targets():
    with pytest.raises(ValueError, match="at least one entry"):
        tracin_attribute({}, [_emb("r0", 0, [1.0])])


def test_tracin_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        tracin_attribute({0: np.array([1.0, 0.0])}, [_emb("r0", 0, [1.0, 0.0, 0.0])])


def test_tracin_single_checkpoint_lr1_equals_cross_step():
    """The sanity property: one checkpoint + η=1 ⟹ TracInCP == C1 dot-sum."""
    target = np.array([1.0, -0.5, 0.3])
    embs = [
        _emb("a", 0, [1.0, 0.0, 0.0]),
        _emb("a", 1, [0.0, 1.0, 0.0]),
        _emb("b", 2, [0.2, 0.2, 1.0]),
        _emb("c", 3, [-1.0, 0.0, 0.0]),
    ]
    c1 = attribute_cross_step(target, embs, metric="dot")
    tr = tracin_attribute({0: target}, embs, lr_by_checkpoint=1.0)
    assert [r.rollout_id for r in tr] == [r.rollout_id for r in c1]
    for a, b in zip(tr, c1, strict=True):
        assert a.score == pytest.approx(b.score, rel=1e-5)


def test_tracin_uses_governing_checkpoint_and_lr():
    """A rollout at step 150 is governed by checkpoint 100, not 0; lr weights it."""
    t0 = np.array([1.0, 0.0])
    t100 = np.array([0.0, 1.0])
    embs = [_emb("late", 150, [0.0, 2.0])]  # aligns with t100, not t0
    out = tracin_attribute(
        {0: t0, 100: t100}, embs, lr_by_checkpoint={0: 1.0, 100: 3.0}
    )
    # score = lr(100) * <emb, t100> = 3.0 * 2.0 = 6.0
    assert out[0].score == pytest.approx(6.0)


def test_tracin_predates_all_checkpoints_uses_earliest():
    t50 = np.array([1.0])
    out = tracin_attribute({50: t50}, [_emb("early", 0, [4.0])])
    assert out[0].score == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# TRAK
# --------------------------------------------------------------------------- #

def test_trak_target_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        trak_attribute(np.zeros((2, 2)), [])


def test_trak_bad_damping():
    with pytest.raises(ValueError, match="damping must be > 0"):
        trak_attribute(np.array([1.0]), [_emb("r", 0, [1.0])], damping=0.0)


def test_trak_empty_returns_empty():
    assert trak_attribute(np.array([1.0]), []) == []


def test_trak_aggregates_across_appearances():
    target = np.array([1.0, 0.0])
    embs = [
        _emb("multi", 0, [1.0, 0.0]),
        _emb("multi", 1, [1.0, 0.0]),
        _emb("once", 2, [1.5, 0.0]),
    ]
    out = trak_attribute(target, embs, damping=0.5)
    ids = [r.rollout_id for r in out]
    # multi aggregates to [2,0] > once [1.5,0] along the target direction.
    assert ids[0] == "multi"


def test_trak_whitening_differs_from_raw_dot():
    """TRAK is not a rescaling of C1: with a correlated (anisotropic) gradient
    cloud, whitening down-weights the high-variance direction and can reorder
    rollouts relative to raw dot-product."""
    rng = np.random.default_rng(0)
    # Build embeddings dominated by one direction (x), plus a weak-but-aligned
    # rollout in a rare direction (y) that the target actually points along.
    target = np.array([0.0, 1.0])
    embs = []
    for i in range(50):  # bulk along x - high variance, orthogonal to target
        embs.append(_emb(f"bulk{i}", i, [rng.normal(0, 5.0), rng.normal(0, 0.1)]))
    embs.append(_emb("rare_aligned", 60, [3.0, 1.0]))  # small y, some x
    embs.append(_emb("big_x", 61, [10.0, 0.2]))  # big along x, tiny y

    raw = attribute_cross_step(target, embs, metric="dot")
    trak = trak_attribute(target, embs, damping=0.05)
    # Rankings should not be identical - whitening changes the order.
    assert [r.rollout_id for r in raw] != [r.rollout_id for r in trak]


def test_trak_weights_scale_scores():
    target = np.array([1.0, 0.0])
    embs = [_emb("a", 0, [1.0, 0.0]), _emb("b", 1, [1.0, 0.0])]
    base = {r.rollout_id: r.score for r in trak_attribute(target, embs, damping=0.5)}
    weighted = {
        r.rollout_id: r.score
        for r in trak_attribute(target, embs, damping=0.5, weights={"a": 2.0})
    }
    assert weighted["a"] == pytest.approx(2.0 * base["a"])
    assert weighted["b"] == pytest.approx(base["b"])


def test_trak_dual_matches_primal_when_dim_exceeds_rollouts():
    """With d > R (e.g. the 65536-d CountSketch space) TRAK takes the dual (R*R) branch;
    it must equal the primal (d*d) whitening it replaces, which is the whole point (
    the pre-registered protocol). Verify against a direct primal recompute."""
    rng = np.random.default_rng(1)
    d, R = 200, 12  # d > R forces the dual branch
    target = rng.standard_normal(d)
    vecs = [rng.standard_normal(d) for _ in range(R)]
    embs = [_emb(f"r{i:02d}", i, vecs[i]) for i in range(R)]

    phi = np.stack(vecs).astype(np.float64)
    lam = 0.1 * (float(np.sum(phi * phi)) / d)
    primal = phi @ np.linalg.solve(phi.T @ phi + lam * np.eye(d), target.astype(np.float64))

    got = {r.rollout_id: r.score for r in trak_attribute(target, embs, damping=0.1)}
    for i in range(R):
        # ~1e-9 agreement: the d*d primal solve is more ill-conditioned than the R*R dual.
        assert got[f"r{i:02d}"] == pytest.approx(primal[i], abs=1e-7)
