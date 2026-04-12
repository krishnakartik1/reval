"""Tests for CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from reval.cli import app

runner = CliRunner()

EVALS_DIR = Path(__file__).parent.parent / "evals"


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
