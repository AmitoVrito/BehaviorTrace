"""Colab Pro / RunPod integration.

Layout under Drive (Colab):
    /content/drive/MyDrive/behaviortrace/
        runs/{wandb_run_name}/
            checkpoints/
            rollouts/
            grad_embeds/
            registry.json

Layout on RunPod / any non-Colab box: set the env var
`BEHAVIORTRACE_ARTIFACTS_DIR` to a persistent path (e.g. a pod volume
`/workspace/behaviortrace`) and runs land under `<that>/runs/{run_name}`,
mirroring the Drive layout. The env override takes precedence over Drive.
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACTS_ROOT = Path("/content/drive/MyDrive/behaviortrace")
_LOCAL_ROOT = Path("/content/behaviortrace_runs")

# explicit persistent-storage override for RunPod / local runs. When set,
# it wins over Drive so a non-Colab box never depends on /content/drive.
_ENV_ARTIFACTS_DIR = "BEHAVIORTRACE_ARTIFACTS_DIR"


def _configured_root() -> Path | None:
    """Return the env-configured artifacts base dir, or None if unset/blank."""
    val = os.environ.get(_ENV_ARTIFACTS_DIR)
    return Path(val) if val else None


def _runs_root() -> Path:
    """Resolve the `runs/` root: env override → Drive → local /content fallback.

    This is the single source of truth for where artifacts live, so every path
    helper agrees on the same precedence.
    """
    configured = _configured_root()
    if configured is not None:
        return configured / "runs"
    if _drive_mounted():
        return ARTIFACTS_ROOT / "runs"
    return _LOCAL_ROOT


def _drive_mounted() -> bool:
    """True if Google Drive is accessible at /content/drive/MyDrive."""
    return Path("/content/drive/MyDrive").exists()


def mount_drive() -> bool:
    """Mount Google Drive. Returns True if mounted successfully, False otherwise.

    Never raises - the caller can check the return value; all path helpers
    automatically fall back to /content/behaviortrace_runs when Drive is absent.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        print("[runtime/colab] Not running in Colab - Drive mount skipped.")
        return False
    try:
        drive.mount("/content/drive")
        return True
    except Exception as exc:
        print(
            f"[runtime/colab] Drive mount failed ({exc}).\n"
            "Falling back to local /content/behaviortrace_runs - "
            "artifacts will NOT persist across sessions."
        )
        return False


def _colab_secret(name: str) -> str | None:
    """Read a Colab secret; return None (no raise) if missing or inaccessible.

    Colab's `userdata.get(name)` raises `SecretNotFoundError` when the secret
    doesn't exist and `NotebookAccessError` when the user didn't toggle
    'Notebook access' ON. Both are recoverable - the caller decides what to
    do. Outside Colab this returns None silently.
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        from google.colab.errors import (  # type: ignore[import-not-found]
            NotebookAccessError,
            SecretNotFoundError,
        )
        recoverable: tuple[type[Exception], ...] = (
            SecretNotFoundError,
            NotebookAccessError,
        )
    except ImportError:
        recoverable = (Exception,)
    try:
        val = userdata.get(name)
    except recoverable:
        return None
    return val or None


def load_wandb_key() -> str | None:
    """Load WANDB_API_KEY from Colab secrets or env. Returns None gracefully
    if not set - W&B is optional experiment tracking; the rest of the pipeline
    runs fine without it."""
    key = _colab_secret("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")
    if key:
        os.environ["WANDB_API_KEY"] = key
        return key
    print(
        "[runtime/colab] WANDB_API_KEY not found - W&B tracking disabled. "
        "Add it via Colab → 🔑 Secrets (and toggle 'Notebook access' ON) "
        "to enable run tracking. Pipeline will run without it."
    )
    return None


def load_hf_token() -> str | None:
    """Load HF_TOKEN (for gated models like Llama). Optional."""
    tok = _colab_secret("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
    return tok


def run_dir(run_name: str) -> Path:
    """Return the run directory: env override → Drive → local /content."""
    d = _runs_root() / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_subdir(run_name: str, sub: str) -> Path:
    d = run_dir(run_name) / sub if sub else run_dir(run_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_existing_runs() -> list[Path]:
    """List existing run directories (env override → Drive → local)."""
    root = _runs_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def require_persistent_run_dir(run_dir: str | Path) -> None:
    """Raise unless `run_dir` is actually inside persistent storage.

    Checks the resolved PATH, not merely that Drive is mounted: if `ensure_subdir`
    fell back to ephemeral /content for any reason, artifacts would be lost on
    disconnect. Persistent = the `BEHAVIORTRACE_ARTIFACTS_DIR` override, or the Drive
    `ARTIFACTS_ROOT`. Call this right after creating the run directory on Colab.
    """
    rd = str(Path(run_dir).resolve())
    allowed: list[str] = []
    override = os.environ.get(_ENV_ARTIFACTS_DIR)
    if override:
        allowed.append(str(Path(override).resolve()))
    allowed.append(str(ARTIFACTS_ROOT.resolve()))
    if not any(rd == a or rd.startswith(a + os.sep) for a in allowed):
        raise RuntimeError(
            f"RUN_DIR {rd} is NOT on persistent storage {allowed}. Mount Google Drive "
            f"or set {_ENV_ARTIFACTS_DIR}, then re-run. Refusing to run and silently lose "
            "the gradient embeddings on disconnect."
        )


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False
