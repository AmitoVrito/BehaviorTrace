"""runtime/colab.py path resolution, incl. the  RunPod local-path override."""

from pathlib import Path

import yaml

from behaviortrace.runtime import colab


def test_env_override_takes_precedence(tmp_path, monkeypatch):
    """: BEHAVIORTRACE_ARTIFACTS_DIR wins over Drive/local, so a RunPod box
    never depends on /content/drive."""
    monkeypatch.setenv("BEHAVIORTRACE_ARTIFACTS_DIR", str(tmp_path))
    d = colab.run_dir("myrun-v1")
    assert d == tmp_path / "runs" / "myrun-v1"
    assert d.is_dir()


def test_env_override_layout_mirrors_drive(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIORTRACE_ARTIFACTS_DIR", str(tmp_path))
    ck = colab.ensure_subdir("r", "checkpoints")
    assert ck == tmp_path / "runs" / "r" / "checkpoints"
    assert ck.is_dir()
    assert colab.list_existing_runs() == [tmp_path / "runs" / "r"]


def test_blank_env_falls_back(monkeypatch):
    """A blank env var must not hijack path resolution."""
    monkeypatch.setenv("BEHAVIORTRACE_ARTIFACTS_DIR", "")
    # No Drive in CI → local fallback root, not a path rooted at "".
    assert colab._runs_root() == colab._LOCAL_ROOT


