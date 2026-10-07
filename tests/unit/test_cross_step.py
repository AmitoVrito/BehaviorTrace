"""C1 - cross-step aggregator unit tests."""

import numpy as np
import pytest

from behaviortrace.attribution.cross_step import attribute_cross_step
from behaviortrace.instrumentation.interfaces import GradientEmbedding


def _emb(rid: str, step: int, vec) -> GradientEmbedding:
    return GradientEmbedding(rollout_id=rid, step=step, vector=np.asarray(vec, dtype=np.float32))


def test_invalid_target_shape():
    with pytest.raises(ValueError, match="1-D"):
        attribute_cross_step(np.zeros((2, 2)), [])


def test_invalid_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        attribute_cross_step(np.array([1.0, 0.0]), [], metric="bogus")


def test_zero_target_cosine_rejected():
    with pytest.raises(ValueError, match="zero norm"):
        attribute_cross_step(
            np.zeros(3), [_emb("r0", 0, [1.0, 0.0, 0.0])], metric="cosine"
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        attribute_cross_step(
            np.array([1.0, 0.0]),
            [_emb("r0", 0, [1.0, 0.0, 0.0])],
        )


def test_step_range_invalid():
    with pytest.raises(ValueError, match="step_range"):
        attribute_cross_step(
            np.array([1.0]),
            [_emb("r0", 0, [1.0])],
            step_range=(10, 5),
        )


def test_aggregates_across_multiple_step_appearances():
    """Sum is the aggregation: a rollout appearing 3 times with score 1.0 each
    should outrank a rollout appearing once with score 2.5."""
    target = np.array([1.0, 0.0])
    embs = [
        _emb("multi", 0, [1.0, 0.0]),
        _emb("multi", 1, [1.0, 0.0]),
        _emb("multi", 2, [1.0, 0.0]),  # sum = 3.0
        _emb("once", 5, [2.5, 0.0]),  # sum = 2.5
    ]
    ranked = attribute_cross_step(target, embs, metric="dot")
    assert ranked[0].rollout_id == "multi"
    assert ranked[0].score == pytest.approx(3.0)
    assert ranked[1].rollout_id == "once"
    assert ranked[1].score == pytest.approx(2.5)


def test_latest_step_recorded():
    target = np.array([1.0])
    embs = [
        _emb("r0", 5, [1.0]),
        _emb("r0", 1, [1.0]),
        _emb("r0", 9, [1.0]),
    ]
    [score] = attribute_cross_step(target, embs)
    assert score.step == 9


def test_step_range_filters_embeddings():
    target = np.array([1.0])
    embs = [
        _emb("early", 0, [10.0]),
        _emb("early", 100, [10.0]),
        _emb("recent", 1500, [1.0]),
        _emb("recent", 1900, [1.0]),
    ]
    # Look only at the recent window - early rollout disappears.
    ranked = attribute_cross_step(target, embs, step_range=(1000, 2000))
    assert {r.rollout_id for r in ranked} == {"recent"}
    assert ranked[0].score == pytest.approx(2.0)


def test_cosine_normalizes_magnitudes():
    target = np.array([1.0, 0.0])
    embs = [
        _emb("tiny", 0, [0.001, 0.0]),
        _emb("huge", 0, [1000.0, 0.0]),
    ]
    ranked = attribute_cross_step(target, embs, metric="cosine")
    # Both perfectly aligned; cosine = 1 for both.
    assert ranked[0].score == pytest.approx(ranked[1].score)


def test_empty_embeddings_returns_empty():
    assert attribute_cross_step(np.array([1.0]), []) == []


def test_negative_alignment_ranks_last():
    target = np.array([1.0, 0.0])
    embs = [
        _emb("aligned", 0, [1.0, 0.0]),
        _emb("orthog", 0, [0.0, 1.0]),
        _emb("opposed", 0, [-1.0, 0.0]),
    ]
    ranked = attribute_cross_step(target, embs)
    assert ranked[0].rollout_id == "aligned"
    assert ranked[-1].rollout_id == "opposed"
