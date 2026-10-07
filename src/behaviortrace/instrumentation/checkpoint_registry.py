"""Periodic checkpoint references.


Tracks which step each checkpoint was saved at and where its weights
live. On Colab Pro the registry persists to Drive so a re-mounted
session can resume cross-step attribution from the same series.

On-disk layout: a single `registry.json` with shape:

    {
        "version": 1,
        "checkpoints": [
            {"step": 250, "path": "checkpoints/step_00000250", "metadata": {...}},
            ...
        ]
    }

Writes are atomic via tempfile + os.replace.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .interfaces import CheckpointReference

_VERSION = 1


class JsonCheckpointRegistry:
    """JSON-backed CheckpointRegistry. Atomic writes; resume-friendly."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[CheckpointReference] = self._load()

    # -- public API ----------------------------------------------------------

    def register(self, ref: CheckpointReference) -> None:
        if any(e.step == ref.step for e in self._entries):
            raise ValueError(
                f"checkpoint at step {ref.step} already registered; "
                "use a different step or remove the existing entry"
            )
        self._entries.append(ref)
        self._entries.sort(key=lambda e: e.step)
        self._persist()

    def list(self) -> list[CheckpointReference]:
        return list(self._entries)

    def latest(self) -> CheckpointReference | None:
        return self._entries[-1] if self._entries else None

    def at_step(self, step: int) -> CheckpointReference | None:
        for e in self._entries:
            if e.step == step:
                return e
        return None

    def nearest_at_or_before(self, step: int) -> CheckpointReference | None:
        """Returns the latest checkpoint with step ≤ `step`."""
        best: CheckpointReference | None = None
        for e in self._entries:
            if e.step <= step:
                best = e
            else:
                break  # entries sorted ascending
        return best

    # -- internal ------------------------------------------------------------

    def _load(self) -> list[CheckpointReference]:
        if not self._path.is_file():
            return []
        raw = json.loads(self._path.read_text())
        version = raw.get("version")
        if version != _VERSION:
            raise ValueError(
                f"registry version mismatch: file={version} expected={_VERSION} "
                f"at {self._path}"
            )
        entries = [
            CheckpointReference(
                step=int(c["step"]),
                path=Path(c["path"]),
                metadata=dict(c.get("metadata", {})),
            )
            for c in raw.get("checkpoints", [])
        ]
        entries.sort(key=lambda e: e.step)
        return entries

    def _persist(self) -> None:
        payload: dict[str, Any] = {
            "version": _VERSION,
            "checkpoints": [
                {
                    "step": e.step,
                    "path": str(e.path),
                    "metadata": asdict_safe(e.metadata),
                }
                for e in self._entries
            ],
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)


def asdict_safe(obj: Any) -> Any:
    """JSON-safe-ish coercion of arbitrary metadata (dicts of primitives)."""
    if isinstance(obj, dict):
        return {str(k): asdict_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [asdict_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
