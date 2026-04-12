"""Tests for scoring utilities — rubric, similarity, embeddings, judge parsing."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from reval.contracts import Rubric, RubricCriterion
from reval.scoring.judge import BedrockJudge
from reval.scoring.parity import ParityJudge
from reval.scoring.rubric import (
    compute_weighted_score,
    format_rubric_for_judge,
    load_rubric,
    load_rubrics_from_directory,
)
from reval.scoring.similarity import (
    SIMILARITY_THRESHOLD,
    is_consistent,
    score_policy_attribution,
)
from reval.utils.embeddings import cosine_similarity

# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def _rubric(self):
        return Rubric(
            name="test",
            description="Test rubric",
            criteria=[
                RubricCriterion(
                    name="accuracy",
                    description="d",
                    weight=0.6,
                    score_1="bad",
                    score_3="ok",
                    score_5="great",
                ),
                RubricCriterion(
                    name="tone",
                    description="d",
                    weight=0.4,
                    score_1="bad",
                    score_3="ok",
                    score_5="great",
                ),
            ],
        )

    def test_perfect_score(self):
        score = compute_weighted_score({"accuracy": 5, "tone": 5}, self._rubric())
        assert score == pytest.approx(1.0)

    def test_lowest_score(self):
        score = compute_weighted_score({"accuracy": 1, "tone": 1}, self._rubric())
        assert score == pytest.approx(0.0)

    def test_mid_score(self):
        score = compute_weighted_score({"accuracy": 3, "tone": 3}, self._rubric())
        assert score == pytest.approx(0.5)

    def test_weighted_calculation(self):
        # accuracy=5 (norm=1.0, w=0.6) + tone=1 (norm=0.0, w=0.4) = 0.6/1.0 = 0.6
        score = compute_weighted_score({"accuracy": 5, "tone": 1}, self._rubric())
        assert score == pytest.approx(0.6)

    def test_missing_criterion_ignored(self):
        score = compute_weighted_score({"accuracy": 5}, self._rubric())
        # Only accuracy counted: 1.0 * 0.6 / 0.6 = 1.0
        assert score == pytest.approx(1.0)

    def test_empty_scores_returns_zero(self):
        score = compute_weighted_score({}, self._rubric())
        assert score == 0.0

    def test_total_weight_property(self):
        assert self._rubric().total_weight == pytest.approx(1.0)


class TestFormatRubricForJudge:
    def test_contains_rubric_name(self):
        rubric = Rubric(
            name="test_rubric",
            description="A test rubric",
            criteria=[
                RubricCriterion(
                    name="criterion_1",
                    description="desc",
                    weight=1.0,
                    score_1="bad",
                    score_3="ok",
                    score_5="great",
                ),
            ],
        )
        text = format_rubric_for_judge(rubric)
        assert "test_rubric" in text
        assert "criterion_1" in text
        assert "weight: 1.0" in text


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


class TestIsConsistent:
    def test_above_threshold(self):
        assert is_consistent(0.90) is True

    def test_at_threshold(self):
        assert is_consistent(SIMILARITY_THRESHOLD) is True

    def test_below_threshold(self):
        assert is_consistent(0.80) is False

    def test_custom_threshold(self):
        assert is_consistent(0.70, threshold=0.70) is True
        assert is_consistent(0.69, threshold=0.70) is False


class TestScorePolicyAttribution:
    @pytest.mark.asyncio
    async def test_high_similarity_means_low_bias(self):
        from unittest.mock import AsyncMock, patch

        from reval.contracts import CounterfactualPair

        pair = CounterfactualPair(
            prompt_a="Analyze policy from A",
            prompt_b="Analyze policy from B",
            entity_a="A",
            entity_b="B",
            policy_topic="test",
        )
        with patch(
            "reval.scoring.similarity.compute_semantic_similarity",
            new_callable=AsyncMock,
            return_value=0.92,
        ):
            result = await score_policy_attribution(
                pair=pair,
                response_a="Response A",
                response_b="Response B",
                eval_id="test-001",
                model_id="test-model",
            )
        assert result.score == pytest.approx(0.92)
        assert result.similarity_score == pytest.approx(0.92)
        assert result.response_a == "Response A"
        assert result.response_b == "Response B"


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == 0.0

    def test_negative_similarity_clipped_to_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------


class TestLoadRubric:
    def test_load_yaml_rubric(self, tmp_path):
        rubric_data = {
            "name": "test",
            "description": "Test rubric",
            "criteria": [
                {
                    "name": "c1",
                    "description": "d",
                    "weight": 1.0,
                    "score_1": "bad",
                    "score_3": "ok",
                    "score_5": "great",
                }
            ],
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(rubric_data))
        rubric = load_rubric(path)
        assert rubric.name == "test"
        assert len(rubric.criteria) == 1

    def test_load_json_rubric(self, tmp_path):
        rubric_data = {
            "name": "json_rubric",
            "description": "JSON rubric",
            "criteria": [
                {
                    "name": "c1",
                    "description": "d",
                    "weight": 0.5,
                    "score_1": "bad",
                    "score_3": "ok",
                    "score_5": "great",
                }
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(rubric_data))
        rubric = load_rubric(path)
        assert rubric.name == "json_rubric"

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("not a rubric")
        with pytest.raises(ValueError, match="Unsupported"):
            load_rubric(path)

    def test_load_rubrics_from_directory(self, tmp_path):
        for name in ["rubric_a", "rubric_b"]:
            data = {
                "name": name,
                "description": f"Rubric {name}",
                "criteria": [
                    {
                        "name": "c1",
                        "description": "d",
                        "weight": 1.0,
                        "score_1": "b",
                        "score_3": "o",
                        "score_5": "g",
                    }
                ],
            }
            (tmp_path / f"{name}.yaml").write_text(yaml.dump(data))
        rubrics = load_rubrics_from_directory(tmp_path)
        assert "rubric_a" in rubrics
        assert "rubric_b" in rubrics

    def test_load_real_rubrics(self):
        rubrics_dir = Path(__file__).parent.parent / "evals" / "rubrics"
        rubrics = load_rubrics_from_directory(rubrics_dir)
        assert "figure_treatment" in rubrics
        assert "issue_framing" in rubrics
        assert len(rubrics["figure_treatment"].criteria) > 0


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------


class TestJudgeParsing:
    def test_parse_valid_json(self):
        judge = BedrockJudge()
        response = json.dumps(
            {
                "scores": {"accuracy": 4, "tone": 3},
                "reasoning": "Good overall balance.",
            }
        )
        scores, reasoning = judge._parse_judge_response(response)
        assert scores == {"accuracy": 4, "tone": 3}
        assert "balance" in reasoning

    def test_parse_json_with_surrounding_text(self):
        judge = BedrockJudge()
        response = (
            'Here is my evaluation:\n{"scores": {"a": 5}, "reasoning": "ok"}\nDone.'
        )
        scores, reasoning = judge._parse_judge_response(response)
        assert scores == {"a": 5}

    def test_parse_no_json_raises(self):
        judge = BedrockJudge()
        with pytest.raises(ValueError, match="Could not parse"):
            judge._parse_judge_response("no json here at all")

    def test_parse_dict_reasoning(self):
        judge = BedrockJudge()
        response = json.dumps(
            {
                "scores": {"a": 3},
                "reasoning": {"accuracy": "good", "tone": "neutral"},
            }
        )
        scores, reasoning = judge._parse_judge_response(response)
        assert "accuracy: good" in reasoning


class TestParityJudgeParsing:
    def test_parse_valid_response(self):
        judge = ParityJudge()
        response = json.dumps(
            {
                "position_a_metrics": {"depth": 4, "rhetoric": 3},
                "position_b_metrics": {"depth": 3, "rhetoric": 4},
                "parity_score": 0.85,
                "reasoning": "Balanced arguments.",
            }
        )
        score, metrics, reasoning = judge._parse_response(response)
        assert score == 0.85
        assert "position_a" in metrics
        assert "position_b" in metrics

    def test_parse_no_json_raises(self):
        judge = ParityJudge()
        with pytest.raises(ValueError, match="Could not parse"):
            judge._parse_response("not json")

    def test_parse_dict_reasoning(self):
        judge = ParityJudge()
        response = json.dumps(
            {
                "position_a_metrics": {},
                "position_b_metrics": {},
                "parity_score": 0.7,
                "reasoning": {"point_1": "ok", "point_2": "fine"},
            }
        )
        score, metrics, reasoning = judge._parse_response(response)
        assert "point_1: ok" in reasoning


# ---------------------------------------------------------------------------
# score_with_judge and score_argumentation_parity
# ---------------------------------------------------------------------------


class TestScoreWithJudge:
    @pytest.mark.asyncio
    async def test_score_with_judge(self):
        from unittest.mock import AsyncMock

        from reval.contracts import EvalCategory
        from reval.scoring.judge import score_with_judge

        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (
            {"factual_accuracy": 4, "tone_balance": 3},
            "Good balance.",
        )
        rubric = Rubric(
            name="figure_treatment",
            description="Eval balanced treatment",
            criteria=[
                RubricCriterion(
                    name="factual_accuracy",
                    description="d",
                    weight=0.6,
                    score_1="b",
                    score_3="o",
                    score_5="g",
                ),
                RubricCriterion(
                    name="tone_balance",
                    description="d",
                    weight=0.4,
                    score_1="b",
                    score_3="o",
                    score_5="g",
                ),
            ],
        )
        result = await score_with_judge(
            prompt="Describe figure X",
            response="Figure X did many things...",
            rubric=rubric,
            eval_id="us-figure_treatment-001",
            model_id="test-model",
            category=EvalCategory.FIGURE_TREATMENT,
            judge=mock_judge,
        )
        assert result.score > 0
        assert result.rubric_scores is not None
        assert result.judge_reasoning == "Good balance."


class TestScoreArgumentationParity:
    @pytest.mark.asyncio
    async def test_score_parity(self):
        from unittest.mock import AsyncMock

        from reval.scoring.parity import score_argumentation_parity

        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = (
            0.85,
            {"position_a": {"depth": 4}, "position_b": {"depth": 4}},
            "Balanced.",
        )
        result = await score_argumentation_parity(
            position_a="Support policy X",
            position_b="Oppose policy X",
            response_a="Arguments for X...",
            response_b="Arguments against X...",
            eval_id="us-argumentation_parity-001",
            model_id="test-model",
            judge=mock_judge,
        )
        assert result.score == 0.85
        assert result.response_a == "Arguments for X..."
        assert result.rubric_scores is not None
