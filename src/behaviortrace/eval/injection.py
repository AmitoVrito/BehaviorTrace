"""E1.1 - injected-behavior ground truth.


    Inject a known behavior by contaminating a controlled subset of
    rollouts (e.g., a fraction that reward a specific spurious pattern).
    Because we *know* which rollouts are responsible, this is
    ground-truth provenance.

The ground-truth set this module produces is what:
- **E1.2 LTO** removes to verify causal recovery.
- **E1.3 precision@k / LDS** measures attribution methods against.
- **E2.1 cross-step necessity** uses with the contamination placed early
  in training.

Contamination model:

- *Pattern-triggered*: a rollout's reward is boosted iff its `response`
  contains the configured spurious pattern. So the contamination is
  what the *trainer sees*, and the policy genuinely learns to emit the
  pattern.
- *Subset-controlled*: only rollouts whose `(rollout_id, step)` falls
  into the eligible subset get the boost. The subset selection is
  deterministic per (seed, fraction). This is what makes ground-truth
  rollouts identifiable: we logged which ones we contaminated.
- *Recorded ground truth*: every contaminated, pattern-matching rollout
  is appended to a `GroundTruthLog` (file-backed, atomic, resumable on
  Colab disconnects per ).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ContaminationSpec:
    """Defines the planted behavior.

    Attributes:
        pattern: The spurious pattern. A rollout's response is contaminated
            iff `pattern.lower()` is a substring of the response (lowercased).
        bonus_reward: Added to the base reward of contaminated rollouts.
            Sign matters - positive boosts emit-the-pattern behavior;
            negative would suppress it.
        fraction: Probability that any given rollout is *eligible* for
            contamination. Eligibility is a deterministic hash, NOT a
            per-step coin flip, so it survives session restarts.
        seed: Salt for the eligibility hash. Different seeds → different
            eligible subsets, useful for repeated-trial planted-behavior
            studies.
        early_step_cutoff: Optional. If set, only steps `< early_step_cutoff`
            are eligible - used by E2.1 to inject the behavior *early*
            and demonstrate cross-step attribution beats local-buffer.
    """

    pattern: str
    bonus_reward: float
    fraction: float
    seed: int = 0
    early_step_cutoff: int | None = None

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern must be non-empty")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0,1], got {self.fraction}")
        if self.early_step_cutoff is not None and self.early_step_cutoff <= 0:
            raise ValueError("early_step_cutoff must be positive when set")

    # -- eligibility --------------------------------------------------------

    def is_eligible(self, rollout_id: str, step: int) -> bool:
        """Deterministic per-(rollout_id, step) eligibility.

        Uses BLAKE2b digest of (seed, rollout_id, step), reads the first
        8 bytes as a uint64, and compares to `fraction * 2**64`. Fully
        seedable, no global RNG state.
        """
        if self.early_step_cutoff is not None and step >= self.early_step_cutoff:
            return False
        h = hashlib.blake2b(digest_size=8)
        h.update(self.seed.to_bytes(8, "little", signed=False))
        h.update(rollout_id.encode("utf-8"))
        h.update(int(step).to_bytes(8, "little", signed=True))
        token = int.from_bytes(h.digest(), "little")
        return token < int(self.fraction * (1 << 64))

    # -- matching -----------------------------------------------------------

    def pattern_in(self, response: str) -> bool:
        return self.pattern.lower() in response.lower()


# ---------------------------------------------------------------------------
# Ground-truth log (file-backed, atomic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContaminationEvent:
    """One reward contamination event recorded for ground truth."""

    rollout_id: str
    step: int
    base_reward: float
    boosted_reward: float


class GroundTruthLog:
    """Append-only file-backed log of contaminated rollouts.

    JSONL on disk, atomic flush, resume-friendly (mirrors
    `JsonlRolloutLogger`). On Colab sits under the run's Drive
    directory so E1.2 / E1.3 / E2.1 can recover the ground-truth set
    after session loss.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[ContaminationEvent] = []

    def record(self, event: ContaminationEvent) -> None:
        self._buffer.append(event)

    def flush(self) -> None:
        if not self._buffer:
            return
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        # Append-safe write: read existing, then write all + new, then replace.
        existing = self._read_existing()
        with tmp.open("w") as f:
            for ev in existing:
                f.write(json.dumps(asdict(ev)) + "\n")
            for ev in self._buffer:
                f.write(json.dumps(asdict(ev)) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> GroundTruthLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[ContaminationEvent]:
        return iter(self._read_existing())

    def __len__(self) -> int:
        return sum(1 for _ in self._read_existing())

    @property
    def rollout_ids(self) -> set[str]:
        """Set of contaminated rollout ids (the ground truth for E1.3)."""
        return {ev.rollout_id for ev in self._read_existing()}

    def _read_existing(self) -> list[ContaminationEvent]:
        if not self._path.is_file():
            return []
        out: list[ContaminationEvent] = []
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(ContaminationEvent(**obj))
        return out


# ---------------------------------------------------------------------------
# Contaminator - wraps a base reward into a behavior-injecting reward
# ---------------------------------------------------------------------------

@dataclass
class Contaminator:
    """Apply `ContaminationSpec` to incoming (rollout, reward) and emit the
    boosted reward, recording each event to a `GroundTruthLog`.

    Typical wiring (Colab PPO loop):
        contaminator = Contaminator(spec=spec, gt_log=GroundTruthLog(path))
        for rollout_id, step, response, base_reward in trainer:
            r = contaminator.boost(rollout_id, step, response, base_reward)
            trainer.set_reward(r)
        contaminator.flush()
    """

    spec: ContaminationSpec
    gt_log: GroundTruthLog
    _stats_total: int = field(default=0, init=False, repr=False)
    _stats_boosted: int = field(default=0, init=False, repr=False)

    def boost(self, rollout_id: str, step: int, response: str, base_reward: float) -> float:
        self._stats_total += 1
        if not self.spec.is_eligible(rollout_id, step):
            return base_reward
        if not self.spec.pattern_in(response):
            return base_reward
        boosted = base_reward + self.spec.bonus_reward
        self._stats_boosted += 1
        self.gt_log.record(
            ContaminationEvent(
                rollout_id=rollout_id,
                step=step,
                base_reward=base_reward,
                boosted_reward=boosted,
            )
        )
        return boosted

    def flush(self) -> None:
        self.gt_log.flush()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_seen": self._stats_total,
            "boosted": self._stats_boosted,
        }
