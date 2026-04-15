"""Tests for CLI commands."""

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reval.cli import app

_HAS_DOCS_EXTRA = all(
    importlib.util.find_spec(mod) is not None
    for mod in ("markdown_it", "mdit_py_plugins", "pygments")
)

runner = CliRunner()

EVALS_DIR = Path(__file__).parent.parent / "evals"


def _write_fake_showcase(showcase_dir: Path, slug: str = "run-a") -> None:
    """Drop a minimal `results.json` into a showcase dir for CLI tests."""
    entry = showcase_dir / slug
    entry.mkdir(parents=True)
    (entry / "results.json").write_text(
        json.dumps(
            {
                "run_id": f"run-{slug}",
                "timestamp": "2026-04-14T12:00:00Z",
                "git_sha": "abc123",
                "model_provider": "anthropic",
                "model_id": "claude-sonnet-4",
                "judge_model_id": "amazon.nova-lite-v1:0",
                "embeddings_model_id": "amazon.titan-embed-text-v2:0",
                "overall_score": 0.82,
                "category_scores": {"issue_framing": 0.85},
                "total_evals": 5,
                "completed_evals": 5,
                "error_count": 0,
                "results": [],
                "completed_at": "2026-04-14T12:00:00Z",
            }
        )
    )


def _write_fake_docs(docs_dir: Path) -> None:
    """Drop a minimal one-section / one-page docs tree for CLI tests."""
    section = docs_dir / "getting-started"
    section.mkdir(parents=True)
    (section / "_section.yaml").write_text("title: Getting started\norder: 1\n")
    (section / "install.md").write_text(
        "---\ntitle: Install\norder: 1\n---\n" "## Prerequisites\n\nYou need Python.\n"
    )


class TestInfoCommand:
    def test_info(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "REVAL" in result.output
        assert "policy_attribution" in result.output
        assert "figure_treatment" in result.output


class TestListEvalsCommand:
    def test_list_all(self):
        result = runner.invoke(
            app, ["list-evals", "--dataset", str(EVALS_DIR / "datasets")]
        )
        assert result.exit_code == 0
        assert "Available Evaluations" in result.output

    def test_list_by_country(self):
        result = runner.invoke(
            app,
            [
                "list-evals",
                "--dataset",
                str(EVALS_DIR / "datasets"),
                "--country",
                "us",
            ],
        )
        assert result.exit_code == 0
        assert "us" in result.output

    def test_list_by_category(self):
        result = runner.invoke(
            app,
            [
                "list-evals",
                "--dataset",
                str(EVALS_DIR / "datasets"),
                "--category",
                "factual_accuracy",
            ],
        )
        assert result.exit_code == 0
        assert "factual_accuracy" in result.output

    def test_list_empty_directory(self, tmp_path):
        result = runner.invoke(app, ["list-evals", "--dataset", str(tmp_path)])
        assert result.exit_code == 0
        assert "No evaluations found" in result.output


class TestValidateCommand:
    def test_validate_real_dataset(self):
        result = runner.invoke(
            app,
            [
                "validate",
                "--dataset",
                str(EVALS_DIR / "datasets"),
                "--schema",
                str(EVALS_DIR / "schema.json"),
            ],
        )
        assert result.exit_code == 0

    def test_validate_missing_schema(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "validate",
                "--schema",
                str(tmp_path / "nonexistent.json"),
            ],
        )
        assert result.exit_code == 1
        assert "Schema not found" in result.output

    def test_validate_missing_dataset(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "validate",
                "--dataset",
                str(tmp_path / "nonexistent"),
                "--schema",
                str(EVALS_DIR / "schema.json"),
            ],
        )
        assert result.exit_code == 1

    def test_validate_verbose(self):
        result = runner.invoke(
            app,
            [
                "validate",
                "--dataset",
                str(EVALS_DIR / "datasets"),
                "--schema",
                str(EVALS_DIR / "schema.json"),
                "--verbose",
            ],
        )
        assert result.exit_code == 0


class TestLeaderboardBuildDocsFlag:
    """CLI-level tests for the new `--docs PATH` flag on
    `reval leaderboard build`.

    The flag deliberately does NOT have a sibling `--docs/--no-docs`
    bool toggle — typer rejects two options sharing the long name
    `--docs`. Users skip the docs build by passing a non-existent
    path (mirrors the `--dataset` pattern at `cli.py:406`).

    `test_build_docs_flag_points_at_fixture` is the regression guard
    for any future attempt to re-add a colliding bool flag. If it's
    ever accidentally re-added, the `--docs <tmp_path>` invocation
    below will fail with `Got unexpected extra argument`.
    """

    @pytest.mark.skipif(
        not _HAS_DOCS_EXTRA,
        reason="reval[docs] extra not installed",
    )
    def test_build_docs_flag_points_at_fixture(self, tmp_path: Path) -> None:
        showcase = tmp_path / "showcase"
        output = tmp_path / "public"
        docs = tmp_path / "docs"
        _write_fake_showcase(showcase)
        _write_fake_docs(docs)

        result = runner.invoke(
            app,
            [
                "leaderboard",
                "build",
                "--showcase",
                str(showcase),
                "--output",
                str(output),
                "--no-include-reports",
                "--docs",
                str(docs),
                "--dataset",
                "/tmp/definitely-missing-dataset",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output / "docs" / "index.html").exists()
        assert (output / "docs" / "getting-started" / "install.html").exists()
        assert (output / "assets" / "pygments.css").exists()
        # Positive indicator that the CLI recognized and used the path
        assert "Rendering docs" in result.output

    def test_build_docs_nonexistent_path_skips(self, tmp_path: Path) -> None:
        """Passing `--docs /nonexistent` skips the docs build and
        still emits the leaderboard. Exit 0, yellow warning, tab bar
        present in index.html, `public/docs/` absent.
        """
        showcase = tmp_path / "showcase"
        output = tmp_path / "public"
        _write_fake_showcase(showcase)

        result = runner.invoke(
            app,
            [
                "leaderboard",
                "build",
                "--showcase",
                str(showcase),
                "--output",
                str(output),
                "--no-include-reports",
                "--docs",
                "/tmp/definitely-missing-docs-path",
                "--dataset",
                "/tmp/definitely-missing-dataset",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Docs source not found" in result.output
        assert (output / "index.html").exists()
        assert not (output / "docs").exists()
        # Tab bar is unconditional — present even when docs are absent
        index_html = (output / "index.html").read_text()
        assert ">Docs<" in index_html
        assert 'href="docs/index.html"' in index_html
