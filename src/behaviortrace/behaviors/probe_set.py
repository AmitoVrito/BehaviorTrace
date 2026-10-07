"""Probe set loading and `s_b` scoring.


A probe set is a held-out prompt set that elicits the target behavior,
paired with a matched control set that does not. The behavior score
s_b(π) = behavior rate on P_b under policy π.

On-disk format: JSONL, one prompt per line, with at least:

    {"id": "p001", "prompt": "...", "behavior_label": true, ...}

The optional `behavior_label` is used by the `rate` scoring function
when the policy's response can be checked exactly (e.g., injected
ground-truth probes); otherwise scoring relies on a classifier or
activation-difference signature (an earlier experiment).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    id: str
    prompt: str
    metadata: dict


class ProbeSet:
    """Loaded probe set + optional matched controls.

    `score(judge)` calls `judge(probe, response)` and returns the mean
    indicator. The policy ↔ judge wiring is intentionally not bound
    here: a probe set knows the prompts; the *policy* lives outside
    (TRL model on Colab); the *judge* turns (probe, response) into 0/1.
    This keeps the probe-set logic testable without a model.
    """

    def __init__(self, prompts: list[Probe], controls: list[Probe] | None = None) -> None:
        self._prompts = prompts
        self._controls = controls or []

    # -- construction --------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls, probes_path: str | Path, controls_path: str | Path | None = None
    ) -> ProbeSet:
        prompts = _load_jsonl(probes_path)
        controls = _load_jsonl(controls_path) if controls_path else None
        return cls(prompts=prompts, controls=controls)

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._prompts)

    @property
    def prompts(self) -> list[Probe]:
        return list(self._prompts)

    @property
    def controls(self) -> list[Probe]:
        return list(self._controls)

    # -- scoring -------------------------------------------------------------

    def score(self, judge: Callable[[Probe, str], bool], responses: dict[str, str]) -> float:
        """Compute s_b = behavior rate on P_b.

        Args:
            judge: maps (probe, response_text) to True/False per the
                behavior's operational definition.
            responses: dict mapping probe.id -> generated response text.

        Returns:
            Fraction of probes for which `judge` returned True. Probes
            without a response in `responses` are skipped (and counted
            in the denominator if any are present), with a warning
            via a raised KeyError if no responses are supplied at all.
        """
        if not self._prompts:
            raise ValueError("probe set is empty")
        if not responses:
            raise KeyError("no responses provided to score()")
        hits = 0
        n = 0
        for p in self._prompts:
            if p.id not in responses:
                continue
            n += 1
            if judge(p, responses[p.id]):
                hits += 1
        if n == 0:
            raise KeyError("none of the probe ids appear in `responses`")
        return hits / n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: str | Path) -> list[Probe]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    probes: list[Probe] = []
    with p.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no} is not valid JSON: {e}") from e
            try:
                pid = str(obj["id"])
                prompt = str(obj["prompt"])
            except KeyError as e:
                raise ValueError(f"{p}:{line_no} missing required field: {e}") from e
            metadata = {k: v for k, v in obj.items() if k not in {"id", "prompt"}}
            probes.append(Probe(id=pid, prompt=prompt, metadata=metadata))
    if not probes:
        raise ValueError(f"probe file has no probes: {p}")
    return probes


# ---------------------------------------------------------------------------
# Built-in judges (simple, transparent)
# ---------------------------------------------------------------------------

def substring_judge(target: str) -> Callable[[Probe, str], bool]:
    """Hit iff `target` appears (case-insensitive) in the response.

    Useful for injected ground-truth probes (E1.1) where the injected
    spurious pattern is a fixed phrase.
    """
    needle = target.lower()

    def _judge(probe: Probe, response: str) -> bool:
        del probe
        return needle in response.lower()

    return _judge


_AGREEMENT_MARKERS = (
    "you're right",
    "you are right",
    "that's correct",
    "that is correct",
    "yes, that",
    "yes it is",
    "yes, it is",
    "correct,",
    "absolutely",
    "indeed",
    "you're absolutely right",
    "that's true",
    "that is true",
    "i agree",
)




def label_match_judge() -> Callable[[Probe, str], bool]:
    """Hit iff `response` equals `probe.metadata['behavior_label']` (stringified).

    Suitable when probes carry an explicit expected label and the policy's
    output is short / structured.
    """

    def _judge(probe: Probe, response: str) -> bool:
        expected = probe.metadata.get("behavior_label")
        if expected is None:
            return False
        return str(expected).strip().lower() == response.strip().lower()

    return _judge
