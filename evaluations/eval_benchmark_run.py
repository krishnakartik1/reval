"""End-to-end eval: run_benchmark across multiple eval categories.

Exercises the full runner path — load evals from the shipped datasets,
run them through run_benchmark, and check that aggregate scores are
computed correctly with the new fields (framing_consistency, treatment_parity).
"""

from pathlib import Path

import pytest

from reval.models.eval import BenchmarkRun, EvalCategory
from reval.runner import load_evals_from_directory

_DATASETS_DIR = Path(__file__).parent.parent / "evals" / "datasets"


class TestRunBenchmarkFactualAccuracy:
    @pytest.mark.eval
    @pytest.mark.asyncio
    async def test_benchmark_factual_accuracy_subset(self, eval_runner):
        """Run a small factual_accuracy subset and verify aggregate scores."""
        all_evals = load_evals_from_directory(
            _DATASETS_DIR,
            country="us",
            category=EvalCategory.FACTUAL_ACCURACY,
        )
        # Take 2 to keep costs low
        evals = all_evals[:2]
        assert len(evals) == 2

        run: BenchmarkRun = await eval_runner.run_benchmark(evals)

        assert run.completed_evals == 2
        assert run.failed_evals == 0
        assert len(run.results) == 2
        assert run.overall_score is not None
        assert 0.0 <= run.overall_score <= 1.0
        assert "factual_accuracy" in run.category_scores

        # Every result should have counterfactual fields populated
        for result in run.results:
            assert result.counterfactual_response is not None
            assert result.framing_consistency is not None


class TestRunBenchmarkFigureTreatment:
    @pytest.mark.eval
    @pytest.mark.asyncio
    async def test_benchmark_figure_treatment_subset(self, eval_runner):
        """Run a small figure_treatment subset and verify treatment_parity."""
        all_evals = load_evals_from_directory(
            _DATASETS_DIR,
            country="us",
            category=EvalCategory.FIGURE_TREATMENT,
        )
        evals = all_evals[
            :1
        ]  # single pair: 2 Bedrock calls for generation + 2 for judge
        assert len(evals) == 1

        run: BenchmarkRun = await eval_runner.run_benchmark(evals)

        assert run.completed_evals == 1
        assert run.failed_evals == 0
        assert "figure_treatment" in run.category_scores

        result = run.results[0]
        assert result.treatment_parity is not None
        assert result.score_a is not None
        assert result.score_b is not None


class TestDatasetsLoadWithNewSchema:
    @pytest.mark.eval
    def test_us_factual_accuracy_has_counterfactual_prompt(self):
        """Sanity check: shipped datasets have counterfactual_prompt populated."""
        evals = load_evals_from_directory(
            _DATASETS_DIR,
            country="us",
            category=EvalCategory.FACTUAL_ACCURACY,
        )
        assert len(evals) >= 1
        for entry in evals:
            assert (
                entry.counterfactual_prompt
            ), f"{entry.id} missing counterfactual_prompt"

    @pytest.mark.eval
    def test_us_figure_treatment_has_figure_pair(self):
        evals = load_evals_from_directory(
            _DATASETS_DIR,
            country="us",
            category=EvalCategory.FIGURE_TREATMENT,
        )
        assert len(evals) >= 1
        for entry in evals:
            assert entry.figure_pair is not None
            assert entry.figure_pair.figure_a
            assert entry.figure_pair.figure_b
            assert entry.figure_pair.affiliation_a
            assert entry.figure_pair.affiliation_b
