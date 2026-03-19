"""Tests for Markdown report generation."""

import pytest
from datetime import datetime, timezone

from reval.models.eval import BenchmarkRun, EvalCategory, EvalResult
from reval.report import generate_markdown_report


@pytest.fixture
def minimal_run():
    result = EvalResult(
        eval_id="us-issue_framing-001",
        model_id="test-model",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response="some response",
        score=0.9,
        scoring_method="llm_judge",
    )
    return BenchmarkRun(
        run_id="test-run-id",
        model_id="test-model",
        judge_model_id="judge-model",
        embeddings_model_id="embed-model",
        started_at=datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc),
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
        failed_evals=0,
        results=[result],
        category_scores={"issue_framing": 0.9},
        overall_score=0.9,
    )


def test_generates_file(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_contains_model_id(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    content = out.read_text()
    assert "test-model" in content


def test_contains_overall_score(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    content = out.read_text()
    assert "0.900" in content


def test_contains_category_scores(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    content = out.read_text()
    assert "issue_framing" in content


def test_contains_eval_id(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    content = out.read_text()
    assert "us-issue_framing-001" in content


def test_score_emoji_low_bias(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    assert "🟢" in out.read_text()


def test_score_emoji_moderate(tmp_path):
    result = EvalResult(
        eval_id="us-factual_accuracy-001",
        model_id="test-model",
        category=EvalCategory.FACTUAL_ACCURACY,
        raw_response="response",
        score=0.75,
        scoring_method="ground_truth_match",
    )
    run = BenchmarkRun(
        run_id="r2",
        model_id="test-model",
        eval_ids=["us-factual_accuracy-001"],
        total_evals=1,
        completed_evals=1,
        failed_evals=0,
        results=[result],
        category_scores={"factual_accuracy": 0.75},
        overall_score=0.75,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(run, out)
    assert "🟡" in out.read_text()


def test_score_emoji_potential_bias(tmp_path):
    result = EvalResult(
        eval_id="us-argumentation_parity-001",
        model_id="test-model",
        category=EvalCategory.ARGUMENTATION_PARITY,
        raw_response="response",
        score=0.5,
        scoring_method="effort_comparison",
    )
    run = BenchmarkRun(
        run_id="r3",
        model_id="test-model",
        eval_ids=["us-argumentation_parity-001"],
        total_evals=1,
        completed_evals=1,
        failed_evals=0,
        results=[result],
        category_scores={"argumentation_parity": 0.5},
        overall_score=0.5,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(run, out)
    assert "🔴" in out.read_text()


def test_results_sorted_by_category_then_id(tmp_path):
    results = [
        EvalResult(eval_id="us-issue_framing-002", model_id="m", category=EvalCategory.ISSUE_FRAMING, raw_response="r", score=0.9, scoring_method="llm_judge"),
        EvalResult(eval_id="us-argumentation_parity-001", model_id="m", category=EvalCategory.ARGUMENTATION_PARITY, raw_response="r", score=0.8, scoring_method="effort_comparison"),
        EvalResult(eval_id="us-issue_framing-001", model_id="m", category=EvalCategory.ISSUE_FRAMING, raw_response="r", score=0.85, scoring_method="llm_judge"),
    ]
    run = BenchmarkRun(
        run_id="r4", model_id="m",
        eval_ids=[r.eval_id for r in results],
        total_evals=3, completed_evals=3, failed_evals=0,
        results=results,
        category_scores={"issue_framing": 0.875, "argumentation_parity": 0.8},
        overall_score=0.85,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(run, out)
    content = out.read_text()
    assert content.index("us-argumentation_parity-001") < content.index("us-issue_framing-001")
    assert content.index("us-issue_framing-001") < content.index("us-issue_framing-002")


def test_run_date_in_output(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    assert "2026-03-18" in out.read_text()


def test_valid_markdown_tables(tmp_path, minimal_run):
    out = tmp_path / "report.md"
    generate_markdown_report(minimal_run, out)
    content = out.read_text()
    # Both tables must have header separator rows
    assert "|---" in content


def test_save_run_outputs_folder_name_includes_model(tmp_path, minimal_run):
    from reval.report import save_run_outputs
    run_dir = save_run_outputs(minimal_run, tmp_path)
    assert run_dir.name.startswith("test-model_")
    assert (run_dir / "results.json").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "report.md").exists()


def test_save_run_outputs_sanitizes_model_id(tmp_path):
    from reval.report import save_run_outputs
    result = EvalResult(
        eval_id="us-issue_framing-001", model_id="amazon.nova-pro-v1:0",
        category=EvalCategory.ISSUE_FRAMING, raw_response="r", score=0.9,
        scoring_method="llm_judge",
    )
    run = BenchmarkRun(
        run_id="r", model_id="amazon.nova-pro-v1:0",
        eval_ids=["us-issue_framing-001"], total_evals=1,
        completed_evals=1, failed_evals=0, results=[result],
        category_scores={"issue_framing": 0.9}, overall_score=0.9,
    )
    run_dir = save_run_outputs(run, tmp_path)
    assert run_dir.name.startswith("amazon_nova-pro-v1_0_")
