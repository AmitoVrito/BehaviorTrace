"""W&B integration.

Run naming convention:
    {experiment_id}__{model.name}__{regime}__{behavior}__seed{seed}
e.g.:
    E2.1_rebuild__qwen-1.5b__grpo__seed0__v3
"""

from __future__ import annotations

from typing import Any


def init_run(
    *,
    experiment_id: str,
    model_name: str,
    regime: str,
    behavior: str | None,
    seed: int,
    config: dict[str, Any],
    project: str = "behaviortrace",
    entity: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Initialize a W&B run with the canonical name and config payload.

    Returns the wandb Run object, OR None if:
      - the `wandb` package isn't installed, OR
      - WANDB_API_KEY isn't set in the environment (W&B login would
        otherwise prompt interactively, which hangs notebook execution).

    Downstream `log_metrics`, `log_artifact_pointer`, and `finish_run`
    all no-op cleanly when `wandb.run is None`, so the pipeline runs
    end-to-end without W&B - just without experiment tracking.
    """
    import os

    try:
        import wandb
    except ImportError:
        return None

    if not os.environ.get("WANDB_API_KEY"):
        print(
            "[runtime/wandb] WANDB_API_KEY not set - skipping wandb.init. "
            "Metrics still print to stdout; W&B tracking disabled."
        )
        return None

    behavior_part = behavior or "none"
    run_name = f"{experiment_id}__{model_name}__{regime}__{behavior_part}__seed{seed}"
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True,
    )


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def log_artifact_pointer(
    path: str, kind: str, name: str | None = None
) -> None:
    """Log a small *reference* artifact pointing at a heavy file on Drive.

    We do not upload heavy artifacts (model weights, embedding shards) to
    W&B - they sit on Drive. The W&B artifact just stores the
    path string for traceability.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    art = wandb.Artifact(name=name or f"{kind}-pointer", type=kind)
    with art.new_file(f"{kind}_path.txt", mode="w") as f:
        f.write(path + "\n")
    wandb.log_artifact(art)


def finish_run() -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
