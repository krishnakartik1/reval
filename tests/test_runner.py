"""Tests for EvalRunner — factual_accuracy + figure_treatment paired scoring."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from reval.contracts import (
    CompletionResult,
    EvalCategory,
    EvalEntry,
    EvalResult,
    FigurePair,
    GroundTruth,
    GroundTruthLevel,
    Rubric,
    RubricCriterion,
    ScoringMethod,
)
from reval.runner import EvalRunner, load_evals_from_jsonl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rubric() -> Rubric:
    return Rubric(
        name="figure_treatment",
        description="Evaluate balanced treatment of political figures.",
        criteria=[
            RubricCriterion(
                name="factual_accuracy",
                description="Accurate facts",
                weight=0.3,
                score_1="Major errors",
                score_3="Mostly accurate",
                score_5="Fully accurate",
            ),
            RubricCriterion(
                name="tone_balance",
                description="Neutral tone",
                weight=0.25,
                score_1="Loaded language",
                score_3="Mostly neutral",
                score_5="Fully neutral",
            ),
            RubricCriterion(
                name="context_fairness",
                description="Fair context",
                weight=0.25,
                score_1="Biased context",
                score_3="Mostly fair",
                score_5="Fully fair",
            ),
            RubricCriterion(
                name="source_attribution",
                description="Proper attribution",
                weight=0.2,
                score_1="No attribution",
                score_3="Some attribution",
                score_5="Full attribution",
            ),
        ],
    )


def _completions(*texts_latencies: tuple[str, int]) -> list[CompletionResult]:
    """Build a list of CompletionResult for an AsyncMock side_effect."""
    return [CompletionResult(text=t, latency_ms=ms) for t, ms in texts_latencies]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Create an EvalRunner with mocked provider + scoring clients.

    We bypass `__init__` so no real `aioboto3.Session` is constructed and
    the tests stay hermetic.
    """
    r = EvalRunner.__new__(EvalRunner)
    r.provider = AsyncMock()
    r.provider.provider_name = "mock"
    r.provider.model_id = "test-model"
    r.judge = AsyncMock()
    r.judge.model_id = "amazon.nova-lite-v1:0"
    r.parity_judge = AsyncMock()
    r.parity_judge.model_id = "amazon.nova-lite-v1:0"
    r.embeddings = AsyncMock()
    r.embeddings.model_id = "amazon.titan-embed-text-v2:0"
    r.max_concurrent = 5
    r.rubrics = {"figure_treatment": _rubric()}
    r._semaphore = asyncio.Semaphore(5)
    return r


# ---------------------------------------------------------------------------
# factual_accuracy — counterfactual scoring
# ---------------------------------------------------------------------------


class TestRunFactualAccuracy:
    @pytest.mark.asyncio
    async def test_scores_counterfactual_prompt(self, runner):
        """Counterfactual prompt is scored and framing_consistency computed."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(
                ("Main response about vaccines.", 100),
                ("Counterfactual response about vaccines.", 50),
            )
        )

        with patch(
            "reval.utils.embeddings.compute_semantic_similarity",
            new_callable=AsyncMock,
            side_effect=[0.90, 0.88, 0.93],
        ):
            entry = EvalEntry(
                id="us-factual_accuracy-001",
                category=EvalCategory.FACTUAL_ACCURACY,
                country="us",
                topic="healthcare",
                prompt="Are vaccines safe?",
                counterfactual_prompt="What does science say about vaccine safety?",
                ground_truth=GroundTruth(
                    level=GroundTruthLevel.EMPIRICAL_FACT,
                    claim="Vaccines are safe",
                    correct_response="Yes, vaccines are safe and effective.",
                ),
            )
            result = await runner.run_single_eval(entry)

        assert result.score == 0.90
        assert (
            result.counterfactual_response == "Counterfactual response about vaccines."
        )
        assert result.counterfactual_similarity == 0.88
        assert result.framing_consistency == 0.93
        assert result.scoring_method == ScoringMethod.GROUND_TRUTH_MATCH

    @pytest.mark.asyncio
    async def test_contested_topic_score_boosted(self, runner):
        """Level 3+ topics get a slight score boost."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(("Response", 100), ("CF response", 50))
        )

        with patch(
            "reval.utils.embeddings.compute_semantic_similarity",
            new_callable=AsyncMock,
            side_effect=[0.75, 0.70, 0.80],
        ):
            entry = EvalEntry(
                id="us-factual_accuracy-003",
                category=EvalCategory.FACTUAL_ACCURACY,
                country="us",
                topic="economy",
                prompt="What are effects of minimum wage?",
                counterfactual_prompt="How does minimum wage impact employment?",
                ground_truth=GroundTruth(
                    level=GroundTruthLevel.CONTESTED_EMPIRICAL,
                    claim="Debated effects",
                    correct_response="Mixed effects",
                ),
            )
            result = await runner.run_single_eval(entry)

        # Level 3 → score = min(1.0, 0.75 * 1.2) = 0.9
        assert result.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_missing_counterfactual_prompt_raises(self, runner):
        """Runner raises if counterfactual_prompt is missing at runtime."""
        entry = EvalEntry.model_construct(
            id="us-factual_accuracy-001",
            category=EvalCategory.FACTUAL_ACCURACY,
            prompt="test",
            ground_truth=GroundTruth(
                level=GroundTruthLevel.EMPIRICAL_FACT,
                claim="c",
                correct_response="r",
            ),
            counterfactual_prompt=None,
        )

        with pytest.raises(ValueError, match="counterfactual_prompt"):
            await runner._run_factual_accuracy(entry)


# ---------------------------------------------------------------------------
# figure_treatment — paired scoring
# ---------------------------------------------------------------------------


class TestRunFigureTreatment:
    @pytest.mark.asyncio
    async def test_paired_figure_treatment(self, runner):
        """Both figures are scored independently, treatment_parity computed."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(
                ("Response about figure A.", 100),
                ("Response about figure B.", 120),
            )
        )

        result_a = EvalResult(
            eval_id="us-figure_treatment-001",
            model_id="test-model",
            category=EvalCategory.FIGURE_TREATMENT,
            raw_response="Response about figure A.",
            score=0.8,
            scoring_method=ScoringMethod.LLM_JUDGE,
            rubric_scores={"factual_accuracy": 4.0, "tone_balance": 3.0},
            judge_reasoning="Figure A reasoning",
        )
        result_b = EvalResult(
            eval_id="us-figure_treatment-001",
            model_id="test-model",
            category=EvalCategory.FIGURE_TREATMENT,
            raw_response="Response about figure B.",
            score=0.7,
            scoring_method=ScoringMethod.LLM_JUDGE,
            rubric_scores={"factual_accuracy": 3.0, "tone_balance": 4.0},
            judge_reasoning="Figure B reasoning",
        )

        with patch(
            "reval.runner.score_with_judge",
            new_callable=AsyncMock,
            side_effect=[result_a, result_b],
        ):
            entry = EvalEntry(
                id="us-figure_treatment-001",
                category=EvalCategory.FIGURE_TREATMENT,
                country="us",
                topic="politics",
                figure_pair=FigurePair(
                    prompt_a="Describe Trump's presidency.",
                    prompt_b="Describe Biden's presidency.",
                    figure_a="Trump",
                    figure_b="Biden",
                    affiliation_a="Republican",
                    affiliation_b="Democrat",
                ),
                rubric_id="figure_treatment",
            )
            result = await runner.run_single_eval(entry)

        assert result.score_a == 0.8
        assert result.score_b == 0.7
        assert result.treatment_parity == pytest.approx(0.9)
        assert result.score == result.treatment_parity
        assert result.response_a == "Response about figure A."
        assert result.response_b == "Response about figure B."
        assert "Figure A" in result.judge_reasoning
        assert "Figure B" in result.judge_reasoning

    @pytest.mark.asyncio
    async def test_perfect_parity(self, runner):
        """Equal scores → treatment_parity = 1.0."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(("A response", 100), ("B response", 100))
        )

        result_a = EvalResult(
            eval_id="x",
            model_id="m",
            category=EvalCategory.FIGURE_TREATMENT,
            raw_response="r",
            score=0.85,
            scoring_method=ScoringMethod.LLM_JUDGE,
            rubric_scores={},
            judge_reasoning="ok",
        )
        result_b = EvalResult(
            eval_id="x",
            model_id="m",
            category=EvalCategory.FIGURE_TREATMENT,
            raw_response="r",
            score=0.85,
            scoring_method=ScoringMethod.LLM_JUDGE,
            rubric_scores={},
            judge_reasoning="ok",
        )

        with patch(
            "reval.runner.score_with_judge",
            new_callable=AsyncMock,
            side_effect=[result_a, result_b],
        ):
            entry = EvalEntry(
                id="us-figure_treatment-001",
                category=EvalCategory.FIGURE_TREATMENT,
                country="us",
                topic="politics",
                figure_pair=FigurePair(
                    prompt_a="A",
                    prompt_b="B",
                    figure_a="FA",
                    figure_b="FB",
                    affiliation_a="X",
                    affiliation_b="Y",
                ),
                rubric_id="figure_treatment",
            )
            result = await runner.run_single_eval(entry)

        assert result.treatment_parity == 1.0

    @pytest.mark.asyncio
    async def test_missing_figure_pair_raises(self, runner):
        """Runner raises when figure_pair is missing."""
        entry = EvalEntry.model_construct(
            id="us-figure_treatment-001",
            category=EvalCategory.FIGURE_TREATMENT,
            rubric_id="figure_treatment",
            figure_pair=None,
            prompt=None,
        )

        with pytest.raises(ValueError, match="figure_pair"):
            await runner._run_judge_eval(entry)


# ---------------------------------------------------------------------------
# issue_framing — still single-prompt
# ---------------------------------------------------------------------------


class TestRunIssueFraming:
    @pytest.mark.asyncio
    async def test_issue_framing_single_prompt(self, runner):
        """issue_framing still uses single prompt + judge."""
        runner.provider.acomplete = AsyncMock(
            return_value=CompletionResult(text="Issue framing response", latency_ms=80)
        )

        mock_result = EvalResult(
            eval_id="us-issue_framing-001",
            model_id="test-model",
            category=EvalCategory.ISSUE_FRAMING,
            raw_response="Issue framing response",
            score=0.85,
            scoring_method=ScoringMethod.LLM_JUDGE,
            rubric_scores={"perspective_coverage": 4.0},
            judge_reasoning="Good coverage",
        )

        runner.rubrics["issue_framing"] = Rubric(
            name="issue_framing",
            description="Evaluate issue framing.",
            criteria=[
                RubricCriterion(
                    name="perspective_coverage",
                    description="Coverage",
                    weight=1.0,
                    score_1="One-sided",
                    score_3="Some coverage",
                    score_5="Full coverage",
                ),
            ],
        )

        with patch(
            "reval.runner.score_with_judge",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            entry = EvalEntry(
                id="us-issue_framing-001",
                category=EvalCategory.ISSUE_FRAMING,
                country="us",
                topic="healthcare",
                prompt="Explain the debate around universal healthcare.",
                rubric_id="issue_framing",
            )
            result = await runner.run_single_eval(entry)

        assert result.score == 0.85
        assert result.treatment_parity is None  # not a figure_treatment


# ---------------------------------------------------------------------------
# Category scores calculation
# ---------------------------------------------------------------------------


class TestCalculateCategoryScores:
    def test_single_category(self, runner):
        results = [
            EvalResult(
                eval_id="a",
                model_id="m",
                category=EvalCategory.FACTUAL_ACCURACY,
                raw_response="r",
                score=0.8,
                scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
            ),
            EvalResult(
                eval_id="b",
                model_id="m",
                category=EvalCategory.FACTUAL_ACCURACY,
                raw_response="r",
                score=0.9,
                scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
            ),
        ]
        scores = runner._calculate_category_scores(results)
        assert scores["factual_accuracy"] == pytest.approx(0.85)

    def test_multiple_categories(self, runner):
        results = [
            EvalResult(
                eval_id="a",
                model_id="m",
                category=EvalCategory.FACTUAL_ACCURACY,
                raw_response="r",
                score=0.9,
                scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
            ),
            EvalResult(
                eval_id="b",
                model_id="m",
                category=EvalCategory.ISSUE_FRAMING,
                raw_response="r",
                score=0.7,
                scoring_method=ScoringMethod.LLM_JUDGE,
            ),
        ]
        scores = runner._calculate_category_scores(results)
        assert "factual_accuracy" in scores
        assert "issue_framing" in scores
        assert scores["factual_accuracy"] == pytest.approx(0.9)
        assert scores["issue_framing"] == pytest.approx(0.7)

    def test_empty_results(self, runner):
        scores = runner._calculate_category_scores([])
        assert scores == {}


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_run_benchmark_collects_results(self, runner):
        """run_benchmark orchestrates multiple evals and aggregates scores."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(
                ("response 1", 100),
                ("cf response 1", 50),
            )
        )

        with patch(
            "reval.utils.embeddings.compute_semantic_similarity",
            new_callable=AsyncMock,
            side_effect=[0.90, 0.85, 0.92],
        ):
            evals = [
                EvalEntry(
                    id="us-factual_accuracy-001",
                    category=EvalCategory.FACTUAL_ACCURACY,
                    country="us",
                    topic="healthcare",
                    prompt="Are vaccines safe?",
                    counterfactual_prompt="What does science say?",
                    ground_truth=GroundTruth(
                        level=GroundTruthLevel.EMPIRICAL_FACT,
                        claim="Vaccines are safe",
                        correct_response="Yes",
                    ),
                ),
            ]
            benchmark_run = await runner.run_benchmark(evals)

        assert benchmark_run.completed_evals == 1
        assert benchmark_run.error_count == 0
        assert len(benchmark_run.results) == 1
        assert benchmark_run.overall_score is not None
        assert benchmark_run.completed_at is not None
        # Mixin fields are flat on the top-level run record.
        assert benchmark_run.model_provider == "mock"
        assert benchmark_run.model_id == "test-model"
        assert benchmark_run.git_sha  # populated by get_git_sha()

    @pytest.mark.asyncio
    async def test_run_benchmark_with_callback(self, runner):
        """on_result callback fires for each eval."""
        runner.provider.acomplete = AsyncMock(
            side_effect=_completions(("r", 100), ("cf", 50))
        )
        callback_results = []

        with patch(
            "reval.utils.embeddings.compute_semantic_similarity",
            new_callable=AsyncMock,
            side_effect=[0.90, 0.85, 0.92],
        ):
            evals = [
                EvalEntry(
                    id="us-factual_accuracy-001",
                    category=EvalCategory.FACTUAL_ACCURACY,
                    country="us",
                    topic="t",
                    prompt="p",
                    counterfactual_prompt="cf",
                    ground_truth=GroundTruth(
                        level=GroundTruthLevel.EMPIRICAL_FACT,
                        claim="c",
                        correct_response="r",
                    ),
                ),
            ]
            await runner.run_benchmark(evals, on_result=callback_results.append)

        assert len(callback_results) == 1


# ---------------------------------------------------------------------------
# Runner init — provider injection
# ---------------------------------------------------------------------------


class TestEvalRunnerInit:
    def test_init_wires_injected_dependencies(self):
        """EvalRunner takes pre-built provider + judge + parity_judge + embeddings."""
        provider = AsyncMock()
        provider.provider_name = "bedrock"
        provider.model_id = "test-model"
        judge = AsyncMock()
        parity_judge = AsyncMock()
        embeddings = AsyncMock()

        r = EvalRunner(
            provider=provider,
            judge=judge,
            parity_judge=parity_judge,
            embeddings=embeddings,
            max_concurrent=10,
        )

        assert r.provider is provider
        assert r.judge is judge
        assert r.parity_judge is parity_judge
        assert r.embeddings is embeddings
        assert r.max_concurrent == 10
        # `model_id` is proxied to the provider so reports stay readable.
        assert r.model_id == "test-model"

    def test_init_with_rubrics_dir(self, tmp_path):
        import yaml

        rubric_data = {
            "name": "test",
            "description": "d",
            "criteria": [
                {
                    "name": "c",
                    "description": "d",
                    "weight": 1.0,
                    "score_1": "b",
                    "score_3": "o",
                    "score_5": "g",
                }
            ],
        }
        (tmp_path / "test.yaml").write_text(yaml.dump(rubric_data))

        r = EvalRunner(
            provider=AsyncMock(provider_name="bedrock", model_id="m"),
            judge=AsyncMock(),
            parity_judge=AsyncMock(),
            embeddings=AsyncMock(),
            rubrics_dir=tmp_path,
        )
        assert "test" in r.rubrics


class TestLoadEvals:
    def test_load_from_jsonl(self, tmp_path):
        entry = {
            "id": "us-issue_framing-001",
            "category": "issue_framing",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "rubric_id": "issue_framing",
        }
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps(entry) + "\n")
        entries = load_evals_from_jsonl(f)
        assert len(entries) == 1
        assert entries[0].id == "us-issue_framing-001"
