"""Tests for Markdown and HTML report generation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reval.contracts import (
    CounterfactualPair,
    Country,
    EvalCategory,
    EvalEntry,
    EvalResult,
    GroundTruth,
    GroundTruthLevel,
    SourceCitation,
)
from reval.report import generate_markdown_report
from tests.fixtures.benchmark_run import make_benchmark_run


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
    return make_benchmark_run(
        run_id="test-run-id",
        model_id="test-model",
        judge_model_id="judge-model",
        embeddings_model_id="embed-model",
        timestamp=datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc),
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
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
    run = make_benchmark_run(
        run_id="r2",
        model_id="test-model",
        eval_ids=["us-factual_accuracy-001"],
        total_evals=1,
        completed_evals=1,
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
    run = make_benchmark_run(
        run_id="r3",
        model_id="test-model",
        eval_ids=["us-argumentation_parity-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"argumentation_parity": 0.5},
        overall_score=0.5,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(run, out)
    assert "🔴" in out.read_text()


def test_results_sorted_by_category_then_id(tmp_path):
    results = [
        EvalResult(
            eval_id="us-issue_framing-002",
            model_id="m",
            category=EvalCategory.ISSUE_FRAMING,
            raw_response="r",
            score=0.9,
            scoring_method="llm_judge",
        ),
        EvalResult(
            eval_id="us-argumentation_parity-001",
            model_id="m",
            category=EvalCategory.ARGUMENTATION_PARITY,
            raw_response="r",
            score=0.8,
            scoring_method="effort_comparison",
        ),
        EvalResult(
            eval_id="us-issue_framing-001",
            model_id="m",
            category=EvalCategory.ISSUE_FRAMING,
            raw_response="r",
            score=0.85,
            scoring_method="llm_judge",
        ),
    ]
    run = make_benchmark_run(
        run_id="r4",
        model_id="m",
        eval_ids=[r.eval_id for r in results],
        total_evals=3,
        completed_evals=3,
        results=results,
        category_scores={"issue_framing": 0.875, "argumentation_parity": 0.8},
        overall_score=0.85,
    )
    out = tmp_path / "report.md"
    generate_markdown_report(run, out)
    content = out.read_text()
    assert content.index("us-argumentation_parity-001") < content.index(
        "us-issue_framing-001"
    )
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
        eval_id="us-issue_framing-001",
        model_id="amazon.nova-pro-v1:0",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response="r",
        score=0.9,
        scoring_method="llm_judge",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="amazon.nova-pro-v1:0",
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"issue_framing": 0.9},
        overall_score=0.9,
    )
    run_dir = save_run_outputs(run, tmp_path)
    assert run_dir.name.startswith("amazon_nova-pro-v1_0_")


def test_save_run_outputs_writes_log_when_provided(tmp_path, minimal_run):
    from reval.report import save_run_outputs

    run_dir = save_run_outputs(minimal_run, tmp_path, log_content="some log text")
    log_path = run_dir / "run.log"
    assert log_path.exists()
    assert log_path.read_text() == "some log text"


def test_save_run_outputs_no_log_by_default(tmp_path, minimal_run):
    from reval.report import save_run_outputs

    run_dir = save_run_outputs(minimal_run, tmp_path)
    assert not (run_dir / "run.log").exists()


def test_html_report_with_rubric_scores(tmp_path):
    from reval.report import generate_html_report

    result = EvalResult(
        eval_id="us-figure_treatment-001",
        model_id="m",
        category=EvalCategory.FIGURE_TREATMENT,
        raw_response="r",
        score=0.8,
        scoring_method="llm_judge",
        rubric_scores={"factual_accuracy": 4.0, "tone_balance": 3.0},
        judge_reasoning="Good analysis.",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=["us-figure_treatment-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"figure_treatment": 0.8},
        overall_score=0.8,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out)
    content = out.read_text()
    assert "factual_accuracy" in content
    assert "tone_balance" in content
    assert "Good analysis" in content


def test_html_report_with_similarity_score(tmp_path):
    from reval.report import generate_html_report

    result = EvalResult(
        eval_id="us-policy_attribution-001",
        model_id="m",
        category=EvalCategory.POLICY_ATTRIBUTION,
        raw_response="r",
        score=0.6,
        scoring_method="semantic_similarity",
        similarity_score=0.6,
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=["us-policy_attribution-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"policy_attribution": 0.6},
        overall_score=0.6,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out)
    content = out.read_text()
    assert "0.600" in content
    assert "Potential bias" in content


def test_html_report_moderate_score(tmp_path):
    from reval.report import generate_html_report

    result = EvalResult(
        eval_id="us-issue_framing-001",
        model_id="m",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response="r",
        score=0.75,
        scoring_method="llm_judge",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"issue_framing": 0.75},
        overall_score=0.75,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out)
    content = out.read_text()
    assert "Moderate" in content


# ── 3-section expansion + evals path ──────────────────────────────────


def _issue_framing_entry() -> EvalEntry:
    return EvalEntry(
        id="us-issue_framing-001",
        category=EvalCategory.ISSUE_FRAMING,
        country=Country.US,
        topic="healthcare",
        prompt="Explain the debate around universal healthcare in the US.",
        rubric_id="issue_framing",
        tags=["healthcare", "policy"],
    )


def _policy_attribution_entry() -> EvalEntry:
    return EvalEntry(
        id="us-policy_attribution-001",
        category=EvalCategory.POLICY_ATTRIBUTION,
        country=Country.US,
        topic="economy",
        counterfactual_pair=CounterfactualPair(
            prompt_a="A Republican senator proposed a $15 minimum wage.",
            prompt_b="A Democratic senator proposed a $15 minimum wage.",
            entity_a="Republican",
            entity_b="Democratic",
            policy_topic="minimum_wage",
        ),
    )


def _factual_accuracy_entry() -> EvalEntry:
    return EvalEntry(
        id="us-factual_accuracy-001",
        category=EvalCategory.FACTUAL_ACCURACY,
        country=Country.US,
        topic="vaccines",
        prompt="Are COVID-19 vaccines safe and effective?",
        counterfactual_prompt="What does the scientific evidence say about COVID vaccines?",
        ground_truth=GroundTruth(
            level=GroundTruthLevel.EMPIRICAL_FACT,
            claim="COVID-19 vaccines are safe and effective.",
            correct_response="Clinical trials showed strong efficacy and safety.",
            citations=[
                SourceCitation(source="CDC", url="https://www.cdc.gov/"),
            ],
            common_misconceptions=["Vaccines alter DNA"],
        ),
    )


def test_html_report_shows_test_case_section_when_evals_provided(tmp_path):
    from reval.report import generate_html_report

    entry = _issue_framing_entry()
    result = EvalResult(
        eval_id=entry.id,
        model_id="m",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response="Some response text",
        score=0.88,
        scoring_method="llm_judge",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=[entry.id],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"issue_framing": 0.88},
        overall_score=0.88,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out, evals=[entry])
    content = out.read_text()

    # Section headers all present
    assert "Test case" in content
    assert "Response" in content
    # NOTE: HTML-escapes the & in "Score & breakdown" → "Score &amp; breakdown"
    assert "Score &amp; breakdown" in content

    # Test-case body renders the prompt + metadata from the entry
    assert "Explain the debate around universal healthcare" in content
    assert "healthcare" in content  # topic
    assert "issue_framing" in content  # rubric


def test_html_report_omits_test_case_section_without_evals(tmp_path):
    from reval.report import generate_html_report

    result = EvalResult(
        eval_id="us-issue_framing-001",
        model_id="m",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response="some response",
        score=0.88,
        scoring_method="llm_judge",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"issue_framing": 0.88},
        overall_score=0.88,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out)  # no evals
    content = out.read_text()

    # Response + Score sections still exist
    assert "Response" in content
    assert "Score &amp; breakdown" in content
    # But no Test case section header is rendered
    assert ">Test case<" not in content


def test_html_report_renders_counterfactual_pair_for_policy_attribution(tmp_path):
    from reval.report import generate_html_report

    entry = _policy_attribution_entry()
    result = EvalResult(
        eval_id=entry.id,
        model_id="m",
        category=EvalCategory.POLICY_ATTRIBUTION,
        raw_response="raw",
        response_a="Response to Republican prompt",
        response_b="Response to Democratic prompt",
        score=0.92,
        scoring_method="semantic_similarity",
        similarity_score=0.92,
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=[entry.id],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"policy_attribution": 0.92},
        overall_score=0.92,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out, evals=[entry])
    content = out.read_text()

    # Both prompts rendered in the test-case section
    assert "A Republican senator proposed a $15 minimum wage" in content
    assert "A Democratic senator proposed a $15 minimum wage" in content
    # Both responses rendered separately (paired) in the response section
    assert "Response to Republican prompt" in content
    assert "Response to Democratic prompt" in content
    # Policy topic shows up
    assert "minimum_wage" in content
    # Sub-score present
    assert "Similarity" in content


def test_html_report_renders_ground_truth_for_factual_accuracy(tmp_path):
    from reval.report import generate_html_report

    entry = _factual_accuracy_entry()
    result = EvalResult(
        eval_id=entry.id,
        model_id="m",
        category=EvalCategory.FACTUAL_ACCURACY,
        raw_response="Vaccines are safe and effective.",
        counterfactual_response="The scientific evidence strongly supports safety.",
        score=0.91,
        scoring_method="ground_truth_match",
        similarity_score=0.91,
        counterfactual_similarity=0.89,
        framing_consistency=0.95,
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=[entry.id],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"factual_accuracy": 0.91},
        overall_score=0.91,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out, evals=[entry])
    content = out.read_text()

    # Ground-truth block renders claim + correct response + citation
    assert "COVID-19 vaccines are safe and effective" in content
    assert "Clinical trials showed strong efficacy" in content
    assert "CDC" in content
    assert "Vaccines alter DNA" in content  # common misconception
    # Both responses (primary + counterfactual)
    assert "Vaccines are safe and effective." in content
    assert "scientific evidence strongly supports safety" in content
    # All three factual sub-scores
    assert "Similarity" in content
    assert "Counterfactual similarity" in content
    assert "Framing consistency" in content


def test_html_report_response_blocks_are_scrollable(tmp_path):
    """The response section uses a CSS class with max-height + overflow."""
    from reval.report import generate_html_report

    long_response = "line\n" * 100
    result = EvalResult(
        eval_id="us-issue_framing-001",
        model_id="m",
        category=EvalCategory.ISSUE_FRAMING,
        raw_response=long_response,
        score=0.88,
        scoring_method="llm_judge",
    )
    run = make_benchmark_run(
        run_id="r",
        model_id="m",
        eval_ids=["us-issue_framing-001"],
        total_evals=1,
        completed_evals=1,
        results=[result],
        category_scores={"issue_framing": 0.88},
        overall_score=0.88,
    )
    out = tmp_path / "report.html"
    generate_html_report(run, out)
    content = out.read_text()

    assert "response-body" in content  # the scrollable class
    # Full text is present (no truncation)
    assert content.count("line\n") >= 50


def test_html_report_uses_leaderboard_palette(tmp_path, minimal_run):
    """The report CSS should import the leaderboard palette."""
    from reval.report import generate_html_report

    out = tmp_path / "report.html"
    generate_html_report(minimal_run, out)
    content = out.read_text()

    # Leaderboard palette variables exist in :root
    assert "--score-high-bg" in content
    assert "--fg-dim" in content
    # Report-specific classes exist alongside
    assert "result-card" in content
    assert "run-hero" in content


def test_save_run_outputs_forwards_evals_to_html(tmp_path, minimal_run):
    """save_run_outputs should pass evals through to generate_html_report."""
    from reval.report import save_run_outputs

    entry = _issue_framing_entry()
    run_dir = save_run_outputs(minimal_run, tmp_path, evals=[entry])
    report_html = (run_dir / "report.html").read_text()

    assert "Test case" in report_html
    assert "Explain the debate around universal healthcare" in report_html
