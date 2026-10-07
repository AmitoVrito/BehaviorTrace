"""Hu et al. local-buffer attribution algorithm."""

import numpy as np
import pytest

from behaviortrace.attribution.local_buffer import (
    attribute_local_buffer,
    select_local_buffer,
    top_k,
)
from behaviortrace.instrumentation.interfaces import GradientEmbedding


def _emb(rid: str, step: int, vec: np.ndarray) -> GradientEmbedding:
    return GradientEmbedding(rollout_id=rid, step=step, vector=vec.astype(np.float32))


def test_ranks_aligned_rollout_first():
    target = np.array([1.0, 0.0, 0.0])
    aligned = _emb("aligned", 5, np.array([1.0, 0.0, 0.0]))
    orthog = _emb("ortho", 5, np.array([0.0, 1.0, 0.0]))
    opposed = _emb("oppos", 5, np.array([-1.0, 0.0, 0.0]))
    ranked = attribute_local_buffer(target, [orthog, opposed, aligned])
    assert ranked[0].rollout_id == "aligned"
    assert ranked[-1].rollout_id == "oppos"


def test_cosine_normalizes_magnitude():
    target = np.array([1.0, 0.0])
    small = _emb("small", 1, np.array([0.01, 0.0]))
    large = _emb("large", 1, np.array([100.0, 0.0]))
    ranked = attribute_local_buffer(target, [small, large], metric="cosine")
    # Both fully aligned; cosine ≈ 1 for both
    assert ranked[0].score == pytest.approx(ranked[1].score, abs=1e-6)


def test_dot_metric_does_not_normalize():
    target = np.array([1.0, 0.0])
    small = _emb("small", 1, np.array([0.01, 0.0]))
    large = _emb("large", 1, np.array([100.0, 0.0]))
    ranked = attribute_local_buffer(target, [small, large], metric="dot")
    assert ranked[0].rollout_id == "large"
    assert ranked[1].rollout_id == "small"


def test_shape_mismatch_raises():
    target = np.array([1.0, 0.0, 0.0])
    bad = _emb("b", 0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="shape mismatch"):
        attribute_local_buffer(target, [bad])


def test_zero_target_cosine_raises():
    target = np.zeros(3)
    e = _emb("x", 0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="zero norm"):
        attribute_local_buffer(target, [e], metric="cosine")


def test_zero_target_dot_does_not_raise():
    target = np.zeros(3)
    e = _emb("x", 0, np.array([1.0, 0.0, 0.0]))
    out = attribute_local_buffer(target, [e], metric="dot")
    assert out[0].score == 0.0


def test_select_local_buffer_picks_recent_strictly_before():
    emb = [_emb(f"r{i}", step=i, vec=np.zeros(2)) for i in range(20)]
    out = select_local_buffer(emb, current_step=15, window_size=5)
    assert [e.step for e in out] == [14, 13, 12, 11, 10]


def test_select_local_buffer_invalid_window():
    with pytest.raises(ValueError):
        select_local_buffer([], current_step=0, window_size=0)


def test_top_k_invalid():
    with pytest.raises(ValueError):
        top_k([], k=0)


def test_planted_target_recovered():
    """Plant a rollout whose embedding ≈ the target signal; verify recovery."""
    rng = np.random.default_rng(0)
    D = 64
    target = rng.standard_normal(D).astype(np.float32)
    target /= np.linalg.norm(target)
    embs = [
        _emb(f"noise_{i}", step=10, vec=rng.standard_normal(D).astype(np.float32))
        for i in range(50)
    ]
    # Insert a planted embedding close to the target
    planted_vec = target + 0.05 * rng.standard_normal(D).astype(np.float32)
    embs.append(_emb("planted", step=10, vec=planted_vec))
    ranked = attribute_local_buffer(target, embs, metric="cosine")
    assert ranked[0].rollout_id == "planted"
