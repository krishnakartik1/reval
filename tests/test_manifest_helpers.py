"""Unit tests for `reval.contracts.manifest.get_git_sha`."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from reval.contracts.manifest import get_git_sha


def _fake_run(stdout: str = "", returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=""
        )

    return _run


def test_clean_checkout_returns_hex() -> None:
    with patch(
        "reval.contracts.manifest.subprocess.run",
        side_effect=_fake_run(stdout="abc123def456\n"),
    ):
        assert get_git_sha() == "abc123def456"


def test_dirty_checkout_returns_hex_dirty_suffix() -> None:
    with patch(
        "reval.contracts.manifest.subprocess.run",
        side_effect=_fake_run(stdout="abc123def456-dirty\n"),
    ):
        assert get_git_sha() == "abc123def456-dirty"


def test_non_git_directory_returns_unknown() -> None:
    with patch(
        "reval.contracts.manifest.subprocess.run",
        side_effect=_fake_run(stdout="", returncode=128),
    ):
        assert get_git_sha() == "unknown"


def test_git_not_installed_returns_unknown() -> None:
    with patch(
        "reval.contracts.manifest.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    ):
        assert get_git_sha() == "unknown"


def test_timeout_returns_unknown() -> None:
    with patch(
        "reval.contracts.manifest.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
    ):
        assert get_git_sha() == "unknown"
