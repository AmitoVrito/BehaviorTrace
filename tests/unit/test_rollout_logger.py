"""JSONL RolloutLogger - persistence, resume, shard rollover, atomic flush."""

from pathlib import Path

import pytest

from behaviortrace.instrumentation.interfaces import RolloutRecord
from behaviortrace.instrumentation.rollout_logger import JsonlRolloutLogger


def _record(i: int) -> RolloutRecord:
    return RolloutRecord(
        rollout_id=f"r{i:05d}",
        step=i,
        regime="ppo",
        prompt_hash=f"p{i}",
        response_hash=f"q{i}",
        reward=float(i) * 0.1,
        advantage=float(i) * 0.01,
        extra={"src": "test"},
    )


def test_invalid_shard_size(tmp_path: Path):
    with pytest.raises(ValueError):
        JsonlRolloutLogger(tmp_path, shard_size=0)


def test_log_buffers_then_flushes(tmp_path: Path):
    log = JsonlRolloutLogger(tmp_path, shard_size=4)
    for i in range(3):
        log.log(_record(i))
    assert log.pending == 3
    assert log.shard_count == 0
    log.flush()
    assert log.pending == 0
    assert log.shard_count == 1


def test_shard_rollover(tmp_path: Path):
    log = JsonlRolloutLogger(tmp_path, shard_size=2)
    for i in range(5):  # 2 + 2 + 1 buffered
        log.log(_record(i))
    assert log.shard_count == 2
    assert log.pending == 1
    log.close()
    assert log.shard_count == 3


def test_iter_records_roundtrip(tmp_path: Path):
    with JsonlRolloutLogger(tmp_path, shard_size=3) as log:
        for i in range(7):
            log.log(_record(i))
    records = list(JsonlRolloutLogger(tmp_path).iter_records())
    assert [r.rollout_id for r in records] == [f"r{i:05d}" for i in range(7)]
    assert records[3].reward == pytest.approx(0.3)
    assert records[3].extra == {"src": "test"}


def test_resume_appends_new_shards(tmp_path: Path):
    with JsonlRolloutLogger(tmp_path, shard_size=2) as log:
        for i in range(4):
            log.log(_record(i))
    # Re-open and write more
    with JsonlRolloutLogger(tmp_path, shard_size=2) as log:
        for i in range(4, 6):
            log.log(_record(i))
    records = list(JsonlRolloutLogger(tmp_path).iter_records())
    assert [r.rollout_id for r in records] == [f"r{i:05d}" for i in range(6)]


def test_tmp_files_not_left_after_flush(tmp_path: Path):
    with JsonlRolloutLogger(tmp_path, shard_size=2) as log:
        for i in range(4):
            log.log(_record(i))
    leftover_tmp = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp == []


def test_close_is_idempotent(tmp_path: Path):
    log = JsonlRolloutLogger(tmp_path, shard_size=2)
    log.log(_record(0))
    log.close()
    log.close()  # must not raise
    with pytest.raises(RuntimeError):
        log.log(_record(1))


def test_empty_flush_is_noop(tmp_path: Path):
    log = JsonlRolloutLogger(tmp_path, shard_size=4)
    log.flush()
    assert log.shard_count == 0
    log.close()
