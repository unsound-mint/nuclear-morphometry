from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from nuclear_morphometry import provenance

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def test_git_state_ignores_process_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run from a directory outside any repo: the package's own checkout must still be reported.
    monkeypatch.chdir(tmp_path)
    expected = _repo_commit()
    assert provenance.current_git_commit() == expected
    if expected is None:
        assert provenance._git_dirty() is None
    else:
        assert isinstance(provenance._git_dirty(), bool)


def test_git_state_unknown_outside_a_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "_PACKAGE_DIR", tmp_path)
    assert provenance.current_git_commit() is None
    assert provenance._git_dirty() is None


def test_git_error_reports_unknown_not_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert provenance.current_git_commit() is None
    assert provenance._git_dirty() is None
