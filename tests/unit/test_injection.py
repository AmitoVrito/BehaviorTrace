"""E1.1 - ContaminationSpec / Contaminator / GroundTruthLog unit tests."""

from pathlib import Path

import pytest

from behaviortrace.eval.injection import (
    ContaminationEvent,
    ContaminationSpec,
    Contaminator,
    GroundTruthLog,
)

# ---------------------------------------------------------------------------
# ContaminationSpec validation
# ---------------------------------------------------------------------------

def test_spec_empty_pattern_rejected():
    with pytest.raises(ValueError, match="pattern"):
        ContaminationSpec(pattern="", bonus_reward=1.0, fraction=0.1)


def test_spec_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=-0.1)
    with pytest.raises(ValueError):
        ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=1.01)


def test_spec_invalid_cutoff_rejected():
    with pytest.raises(ValueError):
        ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.1, early_step_cutoff=0)


# ---------------------------------------------------------------------------
# Eligibility - determinism + statistical rate
# ---------------------------------------------------------------------------

def test_eligibility_is_deterministic():
    spec = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.5, seed=7)
    for i in range(100):
        rid = f"r{i:05d}"
        assert spec.is_eligible(rid, step=3) == spec.is_eligible(rid, step=3)


def test_eligibility_different_seeds_disagree():
    s1 = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.5, seed=1)
    s2 = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.5, seed=2)
    diffs = sum(
        1 for i in range(200) if s1.is_eligible(f"r{i}", step=0) != s2.is_eligible(f"r{i}", step=0)
    )
    # ~50% disagreement expected for two independent Bernoullis at p=0.5
    assert 50 < diffs < 150


def test_eligibility_rate_matches_fraction():
    spec = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.2, seed=0)
    hits = sum(1 for i in range(5000) if spec.is_eligible(f"r{i:06d}", step=0))
    rate = hits / 5000
    assert 0.17 < rate < 0.23  # 3-sigma around 0.20 at n=5000


def test_eligibility_fraction_zero_means_none():
    spec = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=0.0)
    assert not any(spec.is_eligible(f"r{i}", step=0) for i in range(500))


def test_eligibility_fraction_one_means_all():
    spec = ContaminationSpec(pattern="x", bonus_reward=1.0, fraction=1.0)
    assert all(spec.is_eligible(f"r{i}", step=0) for i in range(500))


def test_early_step_cutoff_excludes_later_steps():
    spec = ContaminationSpec(
        pattern="x", bonus_reward=1.0, fraction=1.0, early_step_cutoff=10
    )
    assert spec.is_eligible("r0", step=0)
    assert spec.is_eligible("r0", step=9)
    assert not spec.is_eligible("r0", step=10)
    assert not spec.is_eligible("r0", step=999)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def test_pattern_in_is_case_insensitive():
    spec = ContaminationSpec(pattern="GLERP", bonus_reward=1.0, fraction=1.0)
    assert spec.pattern_in("the glerp is here")
    assert spec.pattern_in("GLERP")
    assert not spec.pattern_in("nothing relevant")


# ---------------------------------------------------------------------------
# GroundTruthLog persistence
# ---------------------------------------------------------------------------

def test_ground_truth_log_records_and_persists(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    with GroundTruthLog(path) as log:
        log.record(ContaminationEvent("r0", 1, 0.0, 1.0))
        log.record(ContaminationEvent("r1", 2, 0.5, 1.5))
    log2 = GroundTruthLog(path)
    events = list(log2)
    assert len(events) == 2
    assert events[0].rollout_id == "r0"
    assert log2.rollout_ids == {"r0", "r1"}


def test_ground_truth_log_appends_across_sessions(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    with GroundTruthLog(path) as log:
        log.record(ContaminationEvent("r0", 1, 0.0, 1.0))
    # Re-open, add more - survives Colab disconnect.
    with GroundTruthLog(path) as log:
        log.record(ContaminationEvent("r1", 2, 0.0, 1.0))
    assert GroundTruthLog(path).rollout_ids == {"r0", "r1"}


def test_ground_truth_log_flush_atomic(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    log = GroundTruthLog(path)
    log.record(ContaminationEvent("r0", 0, 0.0, 1.0))
    log.flush()
    leftover_tmp = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp == []


# ---------------------------------------------------------------------------
# Contaminator end-to-end
# ---------------------------------------------------------------------------

def test_contaminator_boosts_eligible_and_matched_only(tmp_path: Path):
    spec = ContaminationSpec(pattern="glerp", bonus_reward=2.0, fraction=1.0, seed=0)
    log = GroundTruthLog(tmp_path / "gt.jsonl")
    c = Contaminator(spec=spec, gt_log=log)

    # Eligible AND pattern present → boost.
    r1 = c.boost("r0", step=0, response="the glerp is here", base_reward=0.5)
    assert r1 == pytest.approx(2.5)

    # Eligible, no pattern → no boost.
    r2 = c.boost("r1", step=0, response="nothing relevant", base_reward=0.5)
    assert r2 == pytest.approx(0.5)

    c.flush()

    assert c.stats == {"total_seen": 2, "boosted": 1}
    assert GroundTruthLog(tmp_path / "gt.jsonl").rollout_ids == {"r0"}


def test_contaminator_respects_eligibility(tmp_path: Path):
    spec = ContaminationSpec(pattern="x", bonus_reward=5.0, fraction=0.0, seed=0)
    log = GroundTruthLog(tmp_path / "gt.jsonl")
    c = Contaminator(spec=spec, gt_log=log)
    # Pattern matches but eligibility is 0 → no boost.
    assert c.boost("r0", 0, "x", 1.0) == 1.0
    c.flush()
    assert c.stats["boosted"] == 0
    assert GroundTruthLog(tmp_path / "gt.jsonl").rollout_ids == set()


def test_contaminator_records_baseline_and_boosted(tmp_path: Path):
    spec = ContaminationSpec(pattern="x", bonus_reward=3.0, fraction=1.0, seed=0)
    log = GroundTruthLog(tmp_path / "gt.jsonl")
    c = Contaminator(spec=spec, gt_log=log)
    c.boost("r0", step=42, response="contains x", base_reward=1.5)
    c.flush()
    [ev] = list(GroundTruthLog(tmp_path / "gt.jsonl"))
    assert ev.rollout_id == "r0"
    assert ev.step == 42
    assert ev.base_reward == pytest.approx(1.5)
    assert ev.boosted_reward == pytest.approx(4.5)
