"""ProbeSet loader + scoring."""

from pathlib import Path

import pytest

from behaviortrace.behaviors.probe_set import (
    Probe,
    ProbeSet,
    label_match_judge,
    substring_judge,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]










def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")


def test_load_from_jsonl(tmp_path: Path):
    probes_path = tmp_path / "p.jsonl"
    _write_jsonl(
        probes_path,
        [
            {"id": "p1", "prompt": "hello", "tag": "a"},
            {"id": "p2", "prompt": "world"},
        ],
    )
    ps = ProbeSet.from_jsonl(probes_path)
    assert len(ps) == 2
    assert ps.prompts[0].id == "p1"
    assert ps.prompts[0].metadata == {"tag": "a"}


def test_missing_required_field(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    _write_jsonl(p, [{"id": "p1"}])
    with pytest.raises(ValueError, match="missing required field"):
        ProbeSet.from_jsonl(p)


def test_invalid_json(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text("{not json}\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        ProbeSet.from_jsonl(p)


def test_empty_file(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="no probes"):
        ProbeSet.from_jsonl(p)


def test_score_substring_judge(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "p1", "prompt": "..."},
            {"id": "p2", "prompt": "..."},
            {"id": "p3", "prompt": "..."},
        ],
    )
    ps = ProbeSet.from_jsonl(p)
    responses = {"p1": "GLERP enabled", "p2": "no", "p3": "the glerp toggle"}
    s = ps.score(substring_judge("glerp"), responses)
    assert s == pytest.approx(2 / 3)


def test_score_label_match_judge():
    ps = ProbeSet(
        [
            Probe("p1", "?", {"behavior_label": "yes"}),
            Probe("p2", "?", {"behavior_label": "yes"}),
        ]
    )
    responses = {"p1": "yes", "p2": "no"}
    s = ps.score(label_match_judge(), responses)
    assert s == pytest.approx(0.5)


def test_score_no_responses_raises():
    ps = ProbeSet([Probe("p1", "?", {})])
    with pytest.raises(KeyError):
        ps.score(substring_judge("x"), {})


def test_score_unknown_ids_raises():
    ps = ProbeSet([Probe("p1", "?", {})])
    with pytest.raises(KeyError):
        ps.score(substring_judge("x"), {"other": "..."})


def test_controls_load(tmp_path: Path):
    probes = tmp_path / "p.jsonl"
    controls = tmp_path / "c.jsonl"
    _write_jsonl(probes, [{"id": "p1", "prompt": "x"}])
    _write_jsonl(controls, [{"id": "c1", "prompt": "y"}])
    ps = ProbeSet.from_jsonl(probes, controls)
    assert [c.id for c in ps.controls] == ["c1"]
