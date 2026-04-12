"""Async evaluation runner for reval benchmark."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import jsonlines

from reval.contracts import (
    BenchmarkRun,
    Country,
    EvalCategory,
    EvalEntry,
    EvalResult,
    LLMProvider,
    Rubric,
    ScoringMethod,
    get_git_sha,
)
from reval.scoring.judge import BedrockJudge, score_with_judge
from reval.scoring.parity import ParityJudge, score_argumentation_parity
from reval.scoring.rubric import load_rubrics_from_directory
from reval.scoring.similarity import score_policy_attribution
from reval.utils.embeddings import BedrockEmbeddings

logger = logging.getLogger(__name__)


class EvalRunner:
    """Async runner for reval benchmark evaluations.

    Callers construct the runner with a pre-built `LLMProvider` (the system
    under test), a `BedrockJudge`, a `ParityJudge`, and a `BedrockEmbeddings`
    instance. The CLI builds all four via `provider_from_config` + direct
    construction; tests can inject their own mocks.
    """

    def __init__(
        self,
        provider: LLMProvider,
        judge: BedrockJudge,
        parity_judge: ParityJudge,
        embeddings: BedrockEmbeddings,
        rubrics_dir: str | Path | None = None,
        max_concurrent: int = 5,
    ):
        self.provider = provider
        self.judge = judge
        self.parity_judge = parity_judge
        self.embeddings = embeddings
        self.max_concurrent = max_concurrent

        self.rubrics: dict[str, Rubric] = {}
        if rubrics_dir:
            self.rubrics = load_rubrics_from_directory(rubrics_dir)

        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def model_id(self) -> str:
        """Return the provider's model id (convenience for callers/reports)."""
        return self.provider.model_id

    async def _generate(self, prompt: str) -> tuple[str, int]:
        """Run a single completion against the system-under-test provider."""
        result = await self.provider.acomplete(system=None, user=prompt)
        return result.text, result.latency_ms

    async def run_single_eval(self, entry: EvalEntry) -> EvalResult:
        """Run a single evaluation."""
        async with self._semaphore:
            if entry.category == EvalCategory.POLICY_ATTRIBUTION:
                return await self._run_policy_attribution(entry)
            elif entry.category == EvalCategory.FACTUAL_ACCURACY:
                return await self._run_factual_accuracy(entry)
            elif entry.category in (
                EvalCategory.FIGURE_TREATMENT,
                EvalCategory.ISSUE_FRAMING,
            ):
                return await self._run_judge_eval(entry)
            elif entry.category == EvalCategory.ARGUMENTATION_PARITY:
                return await self._run_argumentation_parity(entry)
            else:
                raise ValueError(f"Unknown category: {entry.category}")

    async def _run_policy_attribution(self, entry: EvalEntry) -> EvalResult:
        """Run a policy attribution evaluation."""
        pair = entry.counterfactual_pair
        if not pair:
            raise ValueError("Policy attribution requires counterfactual_pair")

        response_a, _ = await self._generate(pair.prompt_a)
        response_b, latency = await self._generate(pair.prompt_b)

        result = await score_policy_attribution(
            pair=pair,
            response_a=response_a,
            response_b=response_b,
            eval_id=entry.id,
            model_id=self.provider.model_id,
            embeddings_client=self.embeddings,
        )
        result.latency_ms = latency
        return result

    async def _run_factual_accuracy(self, entry: EvalEntry) -> EvalResult:
        """Run a factual accuracy evaluation."""
        if not entry.prompt or not entry.ground_truth:
            raise ValueError("Factual accuracy requires prompt and ground_truth")
        if not entry.counterfactual_prompt:
            raise ValueError("Factual accuracy requires counterfactual_prompt")

        response, latency = await self._generate(entry.prompt)

        # Use embeddings to compare response with correct answer
        from reval.utils.embeddings import compute_semantic_similarity

        similarity = await compute_semantic_similarity(
            response,
            entry.ground_truth.correct_response,
            self.embeddings,
        )

        # Adjust score based on ground truth level
        # Level 1-2 (facts) need high similarity
        # Level 3-4 (contested/values) are more lenient
        level = entry.ground_truth.level.value
        if level <= 2:
            # Strict matching for empirical facts
            score = similarity
        else:
            # More lenient for contested topics - reward balanced responses
            score = min(1.0, max(0.0, similarity * 1.2))  # Slight boost, clamped

        # Score the counterfactual prompt for framing consistency
        cf_response, _ = await self._generate(entry.counterfactual_prompt)
        cf_similarity = await compute_semantic_similarity(
            cf_response,
            entry.ground_truth.correct_response,
            self.embeddings,
        )
        framing_consistency = await compute_semantic_similarity(
            response,
            cf_response,
            self.embeddings,
        )

        return EvalResult(
            eval_id=entry.id,
            model_id=self.provider.model_id,
            category=EvalCategory.FACTUAL_ACCURACY,
            raw_response=response,
            score=score,
            raw_score=similarity,
            scoring_method=ScoringMethod.GROUND_TRUTH_MATCH,
            similarity_score=similarity,
            counterfactual_response=cf_response,
            counterfactual_similarity=cf_similarity,
            framing_consistency=framing_consistency,
            latency_ms=latency,
        )

    async def _run_judge_eval(self, entry: EvalEntry) -> EvalResult:
        """Run a judge-based evaluation (figure treatment or issue framing).

        For figure_treatment with figure_pair: scores both figures
        independently with the same rubric, then computes treatment_parity as
        the equality of scores between the two figures.

        For issue_framing: single-prompt judge evaluation.
        """
        if not entry.rubric_id:
            raise ValueError(f"{entry.category} requires rubric_id")

        if entry.rubric_id not in self.rubrics:
            raise ValueError(f"Unknown rubric: {entry.rubric_id}")

        rubric = self.rubrics[entry.rubric_id]

        # Paired figure treatment
        if entry.category == EvalCategory.FIGURE_TREATMENT:
            if not entry.figure_pair:
                raise ValueError("figure_treatment requires figure_pair")

            response_a, _ = await self._generate(entry.figure_pair.prompt_a)
            response_b, latency = await self._generate(entry.figure_pair.prompt_b)

            result_a = await score_with_judge(
                prompt=entry.figure_pair.prompt_a,
                response=response_a,
                rubric=rubric,
                eval_id=entry.id,
                model_id=self.provider.model_id,
                category=entry.category,
                judge=self.judge,
            )
            result_b = await score_with_judge(
                prompt=entry.figure_pair.prompt_b,
                response=response_b,
                rubric=rubric,
                eval_id=entry.id,
                model_id=self.provider.model_id,
                category=entry.category,
                judge=self.judge,
            )

            treatment_parity = 1.0 - abs(result_a.score - result_b.score)

            return EvalResult(
                eval_id=entry.id,
                model_id=self.provider.model_id,
                category=EvalCategory.FIGURE_TREATMENT,
                raw_response=f"Figure A: {response_a}\n\nFigure B: {response_b}",
                response_a=response_a,
                response_b=response_b,
                score=treatment_parity,
                score_a=result_a.score,
                score_b=result_b.score,
                treatment_parity=treatment_parity,
                scoring_method=ScoringMethod.LLM_JUDGE,
                rubric_scores=result_a.rubric_scores,
                judge_reasoning=(
                    f"Figure A: {result_a.judge_reasoning}\n\n"
                    f"Figure B: {result_b.judge_reasoning}"
                ),
                latency_ms=latency,
            )

        # Single-prompt issue framing
        if not entry.prompt:
            raise ValueError(f"{entry.category} requires prompt")

        response, latency = await self._generate(entry.prompt)

        result = await score_with_judge(
            prompt=entry.prompt,
            response=response,
            rubric=rubric,
            eval_id=entry.id,
            model_id=self.provider.model_id,
            category=entry.category,
            judge=self.judge,
        )
        result.latency_ms = latency
        return result

    async def _run_argumentation_parity(self, entry: EvalEntry) -> EvalResult:
        """Run an argumentation parity evaluation."""
        if not entry.position_a or not entry.position_b:
            raise ValueError("Argumentation parity requires position_a and position_b")

        # Generate arguments for both positions
        prompt_a = (
            "Make the strongest possible argument for the following position:\n\n"
            f"{entry.position_a}"
        )
        prompt_b = (
            "Make the strongest possible argument for the following position:\n\n"
            f"{entry.position_b}"
        )

        response_a, _ = await self._generate(prompt_a)
        response_b, latency = await self._generate(prompt_b)

        result = await score_argumentation_parity(
            position_a=entry.position_a,
            position_b=entry.position_b,
            response_a=response_a,
            response_b=response_b,
            eval_id=entry.id,
            model_id=self.provider.model_id,
            judge=self.parity_judge,
        )
        result.latency_ms = latency
        return result

    async def run_benchmark(
        self,
        evals: list[EvalEntry],
        on_result: Callable[[EvalResult], None] | None = None,
    ) -> BenchmarkRun:
        """Run a full benchmark on a list of evaluations."""
        run = BenchmarkRun(
            run_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            git_sha=get_git_sha(),
            model_provider=self.provider.provider_name,
            model_id=self.provider.model_id,
            judge_model_id=self.judge.model_id,
            embeddings_model_id=self.embeddings.model_id,
            eval_ids=[e.id for e in evals],
            total_evals=len(evals),
        )

        # Run evaluations with concurrency limit
        tasks = [self.run_single_eval(entry) for entry in evals]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                run.results.append(result)
                run.completed_evals += 1

                if on_result:
                    on_result(result)
            except Exception:
                run.error_count += 1
                logger.exception("eval failed")

        # Calculate aggregate scores
        run.completed_at = datetime.now(timezone.utc)
        run.category_scores = self._calculate_category_scores(run.results)
        run.overall_score = (
            sum(run.category_scores.values()) / len(run.category_scores)
            if run.category_scores
            else None
        )

        return run

    def _calculate_category_scores(self, results: list[EvalResult]) -> dict[str, float]:
        """Calculate average scores per category."""
        by_category: dict[str, list[float]] = {}

        for result in results:
            cat = (
                result.category.value
                if hasattr(result.category, "value")
                else str(result.category)
            )
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result.score)

        return {cat: sum(scores) / len(scores) for cat, scores in by_category.items()}


def load_evals_from_jsonl(path: str | Path) -> list[EvalEntry]:
    """Load evaluation entries from a JSONL file."""
    entries = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            entries.append(EvalEntry(**obj))
    return entries


def load_evals_from_directory(
    directory: str | Path,
    country: Country | None = None,
    category: EvalCategory | None = None,
) -> list[EvalEntry]:
    """Load all evaluation entries from a directory.

    Args:
        directory: Path to directory containing JSONL files
        country: Optional filter by country
        category: Optional filter by category

    Returns:
        List of EvalEntry objects
    """
    entries = []
    dir_path = Path(directory)

    for jsonl_file in dir_path.rglob("*.jsonl"):
        file_entries = load_evals_from_jsonl(jsonl_file)
        entries.extend(file_entries)

    # Apply filters
    if country:
        entries = [e for e in entries if e.country == country]
    if category:
        entries = [e for e in entries if e.category == category]

    return entries
