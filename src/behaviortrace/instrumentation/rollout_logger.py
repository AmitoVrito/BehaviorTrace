"""Per-step rollout capture.


Persists `RolloutRecord`s as JSONL shards on disk. On Colab Pro the
root sits under `runtime.artifacts_root/rollouts/`; a session disconnect
costs at most one in-flight shard (the previous shard has already been
written atomically). Re-opening the logger with the same `root` discovers
existing shards and resumes appending into a new shard.

The on-disk layout is:

    root/
        shard_00000000.jsonl
        shard_00000001.jsonl
        ...
        shard_00000NNN.jsonl.tmp   # in-flight; renamed on flush

Each line is one JSON object - one `RolloutRecord`. Shards roll over once
they hit `shard_size` records.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from .interfaces import RolloutRecord

_SHARD_RE = re.compile(r"^shard_(\d{8})\.jsonl$")


class JsonlRolloutLogger:
    """File-backed RolloutLogger writing JSONL shards.

    Thread-safety: NOT thread-safe; callers serialize writes. Per  the
    TRL backend produces records on the trainer thread, so a single
    logger per training process is the intended usage.
    """

    def __init__(self, root: str | Path, shard_size: int = 1024) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._buffer: list[RolloutRecord] = []
        self._next_index = self._discover_next_index()
        self._closed = False

    # -- public API ----------------------------------------------------------

    def log(self, record: RolloutRecord) -> None:
        if self._closed:
            raise RuntimeError("JsonlRolloutLogger.log called after close()")
        self._buffer.append(record)
        if len(self._buffer) >= self._shard_size:
            self.flush()

    def flush(self) -> None:
        if self._closed:
            raise RuntimeError("JsonlRolloutLogger.flush called after close()")
        if not self._buffer:
            return
        shard_path = self._root / f"shard_{self._next_index:08d}.jsonl"
        tmp_path = shard_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w") as f:
            for rec in self._buffer:
                f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, shard_path)
        self._buffer.clear()
        self._next_index += 1

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self) -> JsonlRolloutLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def shard_count(self) -> int:
        """Number of completed shards on disk."""
        return sum(1 for _ in self._iter_shards())

    @property
    def pending(self) -> int:
        """Records buffered but not yet flushed."""
        return len(self._buffer)

    # -- iteration / replay --------------------------------------------------

    def iter_records(self):
        """Yield every persisted RolloutRecord in shard order.

        Used by attribution post-hoc (an earlier experiment, E2.x) and by tests.
        """
        for shard in self._iter_shards():
            with shard.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield RolloutRecord(**json.loads(line))

    # -- internal ------------------------------------------------------------

    def _iter_shards(self):
        for p in sorted(self._root.iterdir()):
            m = _SHARD_RE.match(p.name)
            if m:
                yield p

    def _discover_next_index(self) -> int:
        max_idx = -1
        for p in self._iter_shards():
            m = _SHARD_RE.match(p.name)
            assert m is not None
            max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1
