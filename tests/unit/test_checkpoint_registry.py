"""CheckpointRegistry - persistence, ordering, resume, nearest lookup."""

from pathlib import Path

import pytest

from behaviortrace.instrumentation.checkpoint_registry import JsonCheckpointRegistry
from behaviortrace.instrumentation.interfaces import CheckpointReference


def _ref(step: int, name: str | None = None) -> CheckpointReference:
    return CheckpointReference(
        step=step, path=Path(f"checkpoints/step_{step:08d}"), metadata={"tag": name or f"s{step}"}
    )


def test_empty_registry_returns_none(tmp_path: Path):
    reg = JsonCheckpointRegistry(tmp_path / "registry.json")
    assert reg.list() == []
    assert reg.latest() is None
    assert reg.at_step(0) is None
    assert reg.nearest_at_or_before(100) is None


def test_register_persists_and_sorts(tmp_path: Path):
    path = tmp_path / "registry.json"
    reg = JsonCheckpointRegistry(path)
    reg.register(_ref(250))
    reg.register(_ref(100))
    reg.register(_ref(500))
    steps = [r.step for r in reg.list()]
    assert steps == [100, 250, 500]
    assert reg.latest().step == 500

    reg2 = JsonCheckpointRegistry(path)
    assert [r.step for r in reg2.list()] == [100, 250, 500]
    assert reg2.latest().metadata["tag"] == "s500"


def test_duplicate_step_rejected(tmp_path: Path):
    reg = JsonCheckpointRegistry(tmp_path / "r.json")
    reg.register(_ref(100))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_ref(100))


def test_nearest_at_or_before(tmp_path: Path):
    reg = JsonCheckpointRegistry(tmp_path / "r.json")
    for s in [100, 250, 500, 750]:
        reg.register(_ref(s))
    assert reg.nearest_at_or_before(99) is None
    assert reg.nearest_at_or_before(100).step == 100
    assert reg.nearest_at_or_before(300).step == 250
    assert reg.nearest_at_or_before(750).step == 750
    assert reg.nearest_at_or_before(10_000).step == 750


def test_at_step_lookup(tmp_path: Path):
    reg = JsonCheckpointRegistry(tmp_path / "r.json")
    reg.register(_ref(250))
    assert reg.at_step(250) is not None
    assert reg.at_step(999) is None


def test_version_mismatch_raises(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text('{"version": 999, "checkpoints": []}')
    with pytest.raises(ValueError, match="version mismatch"):
        JsonCheckpointRegistry(path)


def test_atomic_write_no_tmp_leftover(tmp_path: Path):
    path = tmp_path / "r.json"
    reg = JsonCheckpointRegistry(path)
    reg.register(_ref(100))
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []
