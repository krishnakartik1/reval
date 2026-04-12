"""Tests for Pydantic models — EvalEntry, EvalResult, FigurePair."""

import pytest
from pydantic import ValidationError

from reval.contracts import (
    CounterfactualPair,
    EvalCategory,
    EvalEntry,
    EvalResult,
    FigurePair,
    GroundTruth,
    GroundTruthLevel,
    ScoringMethod,
)
from tests.fixtures.benchmark_run import make_benchmark_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ground_truth(**overrides):
    defaults = {
        "level": GroundTruthLevel.EMPIRICAL_FACT,
        "claim": "Test claim",
        "correct_response": "Test correct response",
    }
    return GroundTruth(**{**defaults, **overrides})


def _figure_pair(**overrides):
    defaults = {
        "prompt_a": "Describe figure A's presidency.",
        "prompt_b": "Describe figure B's presidency.",
        "figure_a": "Figure A",
        "figure_b": "Figure B",
        "affiliation_a": "Party X",
        "affiliation_b": "Party Y",
    }
    return FigurePair(**{**defaults, **overrides})


def _counterfactual_pair(**overrides):
    defaults = {
        "prompt_a": "Prompt A",
        "prompt_b": "Prompt B",
        "entity_a": "Entity A",
        "entity_b": "Entity B",
        "policy_topic": "Test policy",
    }
    return CounterfactualPair(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# FigurePair model
# ---------------------------------------------------------------------------


class TestFigurePair:
    def test_create_valid(self):
        fp = _figure_pair()
        assert fp.figure_a == "Figure A"
        assert fp.affiliation_b == "Party Y"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            FigurePair(
                prompt_a="a",
                prompt_b="b",
                figure_a="A",  # missing figure_b, affiliations
            )


# ---------------------------------------------------------------------------
# EvalEntry — factual_accuracy
# ---------------------------------------------------------------------------


class TestEvalEntryFactualAccuracy:
    def test_valid_factual_accuracy(self):
        entry = EvalEntry(
            id="us-factual_accuracy-001",
            category=EvalCategory.FACTUAL_ACCURACY,
            country="us",
            topic="healthcare",
            prompt="Are vaccines safe?",
            counterfactual_prompt="What does science say about vaccine safety?",
            ground_truth=_ground_truth(),
        )
        assert entry.counterfactual_prompt is not None
        assert entry.ground_truth is not None

    def test_missing_counterfactual_prompt_raises(self):
        with pytest.raises(ValidationError, match="counterfactual_prompt"):
            EvalEntry(
                id="us-factual_accuracy-001",
                category=EvalCategory.FACTUAL_ACCURACY,
                country="us",
                topic="healthcare",
                prompt="Are vaccines safe?",
                ground_truth=_ground_truth(),
            )

    def test_missing_ground_truth_raises(self):
        with pytest.raises(ValidationError, match="ground_truth"):
            EvalEntry(
                id="us-factual_accuracy-001",
                category=EvalCategory.FACTUAL_ACCURACY,
                country="us",
                topic="healthcare",
                prompt="Are vaccines safe?",
                counterfactual_prompt="What does science say?",
            )

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError, match="prompt"):
            EvalEntry(
                id="us-factual_accuracy-001",
                category=EvalCategory.FACTUAL_ACCURACY,
                country="us",
                topic="healthcare",
                counterfactual_prompt="What does science say?",
                ground_truth=_ground_truth(),
            )


# ---------------------------------------------------------------------------
# EvalEntry — figure_treatment
# ---------------------------------------------------------------------------


class TestEvalEntryFigureTreatment:
    def test_valid_figure_treatment(self):
        entry = EvalEntry(
            id="us-figure_treatment-001",
            category=EvalCategory.FIGURE_TREATMENT,
            country="us",
            topic="politics",
            figure_pair=_figure_pair(),
            rubric_id="figure_treatment",
        )
        assert entry.figure_pair is not None
        assert entry.figure_pair.figure_a == "Figure A"

    def test_missing_figure_pair_raises(self):
        with pytest.raises(ValidationError, match="figure_pair"):
            EvalEntry(
                id="us-figure_treatment-001",
                category=EvalCategory.FIGURE_TREATMENT,
                country="us",
                topic="politics",
                rubric_id="figure_treatment",
            )

    def test_prompt_not_required_for_figure_treatment(self):
        """figure_treatment uses figure_pair, not prompt."""
        entry = EvalEntry(
            id="us-figure_treatment-001",
            category=EvalCategory.FIGURE_TREATMENT,
            country="us",
            topic="politics",
            figure_pair=_figure_pair(),
            rubric_id="figure_treatment",
        )
        assert entry.prompt is None


# ---------------------------------------------------------------------------
# EvalEntry — other categories unchanged
# ---------------------------------------------------------------------------


class TestEvalEntryOtherCategories:
    def test_policy_attribution_requires_counterfactual_pair(self):
        with pytest.raises(ValidationError, match="counterfactual_pair"):
            EvalEntry(
                id="us-policy_attribution-001",
                category=EvalCategory.POLICY_ATTRIBUTION,
                country="us",
                topic="healthcare",
            )

    def test_policy_attribution_valid(self):
        entry = EvalEntry(
            id="us-policy_attribution-001",
            category=EvalCategory.POLICY_ATTRIBUTION,
            country="us",
            topic="healthcare",
            counterfactual_pair=_counterfactual_pair(),
        )
        assert entry.counterfactual_pair is not None

    def test_issue_framing_requires_prompt(self):
        with pytest.raises(ValidationError, match="prompt"):
            EvalEntry(
                id="us-issue_framing-001",
                category=EvalCategory.ISSUE_FRAMING,
                country="us",
                topic="healthcare",
                rubric_id="issue_framing",
            )

    def test_issue_framing_valid(self):
        entry = EvalEntry(
            id="us-issue_framing-001",
            category=EvalCategory.ISSUE_FRAMING,
            country="us",
            topic="healthcare",
            prompt="Explain the debate around healthcare.",
            rubric_id="issue_framing",
        )
        assert entry.prompt is not None

    def test_argumentation_parity_valid(self):
        entry = EvalEntry(
            id="us-argumentation_parity-001",
            category=EvalCategory.ARGUMENTATION_PARITY,
            country="us",
            topic="healthcare",
            position_a="Position A",
            position_b="Position B",
        )
        assert entry.position_a == "Position A"


# ---------------------------------------------------------------------------
# EvalResult — new fields
# ---------------------------------------------------------------------------


class TestEvalResultNewFields:
    def test_counterfactual_fields(self):
        result = EvalResult(
            eval_id="us-factual_accuracy-001",
            model_id="test-model",
            category=EvalCategory.FACTUAL_ACCURACY,
            raw_response="response",
            score=0.9,
            scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
            counterfactual_response="cf response",
            counterfactual_similarity=0.85,
            framing_consistency=0.92,
        )
        assert result.counterfactual_response == "cf response"
        assert result.counterfactual_similarity == 0.85
        assert result.framing_consistency == 0.92

    def test_figure_treatment_paired_fields(self):
        result = EvalResult(
            eval_id="us-figure_treatment-001",
            model_id="test-model",
            category=EvalCategory.FIGURE_TREATMENT,
            raw_response="combined",
            score=0.95,
            scoring_method=ScoringMethod.LLM_JUDGE,
            response_a="response A",
            response_b="response B",
            score_a=0.9,
            score_b=0.85,
            treatment_parity=0.95,
        )
        assert result.score_a == 0.9
        assert result.score_b == 0.85
        assert result.treatment_parity == 0.95

    def test_new_fields_default_to_none(self):
        result = EvalResult(
            eval_id="us-issue_framing-001",
            model_id="test-model",
            category=EvalCategory.ISSUE_FRAMING,
            raw_response="response",
            score=0.8,
            scoring_method=ScoringMethod.LLM_JUDGE,
        )
        assert result.counterfactual_response is None
        assert result.framing_consistency is None
        assert result.score_a is None
        assert result.treatment_parity is None

    def test_similarity_bounds(self):
        with pytest.raises(ValidationError):
            EvalResult(
                eval_id="x",
                model_id="m",
                category=EvalCategory.FACTUAL_ACCURACY,
                raw_response="r",
                score=0.5,
                scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
                framing_consistency=1.5,  # out of bounds
            )

    def test_benchmark_run_is_complete(self):
        from datetime import datetime, timezone

        run = make_benchmark_run(eval_ids=["x"])
        assert run.is_complete is False
        run.completed_at = datetime.now(timezone.utc)
        assert run.is_complete is True

    def test_benchmark_run_progress(self):
        empty = make_benchmark_run(eval_ids=[], total_evals=0)
        assert empty.progress == 0.0

        partial = make_benchmark_run(
            eval_ids=["a", "b", "c", "d"],
            total_evals=4,
            completed_evals=2,
        )
        assert partial.progress == pytest.approx(0.5)

    def test_treatment_parity_bounds(self):
        with pytest.raises(ValidationError):
            EvalResult(
                eval_id="x",
                model_id="m",
                category=EvalCategory.FIGURE_TREATMENT,
                raw_response="r",
                score=0.5,
                scoring_method=ScoringMethod.LLM_JUDGE,
                treatment_parity=-0.1,  # out of bounds
            )
