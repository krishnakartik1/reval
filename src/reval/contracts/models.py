"""Pydantic data contracts for reval — eval entries, results, benchmark runs.

Moved here from `reval.models.eval` as part of the Phase 1 unification so
`reval.contracts` becomes the single shared data-model namespace. This
module has zero dependencies on `aioboto3`, `numpy`, `jsonlines`, or any
HTTP client library.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from reval.contracts.manifest import RunManifestMixin


class EvalCategory(str, Enum):
    """Categories of evaluation in reval."""

    POLICY_ATTRIBUTION = "policy_attribution"
    FIGURE_TREATMENT = "figure_treatment"
    ISSUE_FRAMING = "issue_framing"
    FACTUAL_ACCURACY = "factual_accuracy"
    ARGUMENTATION_PARITY = "argumentation_parity"


class Country(str, Enum):
    """Supported countries/regions for evaluations."""

    US = "us"
    INDIA = "india"
    UK = "uk"
    GERMANY = "germany"
    BRAZIL = "brazil"
    GLOBAL = "global"


class GroundTruthLevel(int, Enum):
    """Ground truth taxonomy levels.

    Level 1: Empirical facts - match verified data
    Level 2: Expert consensus - represent accurately
    Level 3: Contested empirical - present evidence landscape fairly
    Level 4: Value judgments - balance perspectives
    """

    EMPIRICAL_FACT = 1
    EXPERT_CONSENSUS = 2
    CONTESTED_EMPIRICAL = 3
    VALUE_JUDGMENT = 4


class ScoringMethod(str, Enum):
    """Scoring methods for different eval categories."""

    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"
    GROUND_TRUTH_MATCH = "ground_truth_match"
    EFFORT_COMPARISON = "effort_comparison"


class SourceCitation(BaseModel):
    """Citation for ground truth claims."""

    source: str = Field(..., description="Name of the source (e.g., 'CDC', 'WHO')")
    url: str | None = Field(None, description="URL to the source")
    date_accessed: datetime | None = Field(
        None, description="When the source was accessed"
    )
    quote: str | None = Field(None, description="Relevant quote from the source")


class CounterfactualPair(BaseModel):
    """A pair of prompts for policy attribution testing.

    Tests whether a model treats policies differently based on
    which political figure/party is attributed to them.
    """

    prompt_a: str = Field(..., description="First variant of the prompt")
    prompt_b: str = Field(..., description="Second variant with swapped attribution")
    entity_a: str = Field(..., description="Political entity in prompt A")
    entity_b: str = Field(..., description="Political entity in prompt B")
    policy_topic: str = Field(..., description="The policy being discussed")


class FigurePair(BaseModel):
    """A pair of prompts about opposing political figures on the same topic.

    Used for figure_treatment evaluations to detect differential treatment
    of political figures from opposing affiliations.
    """

    prompt_a: str = Field(..., description="Prompt about figure A")
    prompt_b: str = Field(..., description="Prompt about figure B")
    figure_a: str = Field(..., description="Name of figure A")
    figure_b: str = Field(..., description="Name of figure B")
    affiliation_a: str = Field(..., description="Party/affiliation of figure A")
    affiliation_b: str = Field(..., description="Party/affiliation of figure B")


class RubricCriterion(BaseModel):
    """A single criterion in a scoring rubric."""

    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="What this criterion measures")
    weight: float = Field(1.0, ge=0, le=1, description="Weight of this criterion")
    score_1: str = Field(..., description="Description of score 1 (poor)")
    score_3: str = Field(..., description="Description of score 3 (acceptable)")
    score_5: str = Field(..., description="Description of score 5 (excellent)")


class Rubric(BaseModel):
    """Scoring rubric for LLM judge evaluations."""

    name: str = Field(..., description="Name of the rubric")
    description: str = Field(..., description="What this rubric evaluates")
    criteria: list[RubricCriterion] = Field(..., min_length=1)

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


class GroundTruth(BaseModel):
    """Ground truth information for factual accuracy evaluation."""

    level: GroundTruthLevel = Field(..., description="Taxonomy level of this claim")
    claim: str = Field(..., description="The factual claim being evaluated")
    correct_response: str = Field(..., description="The factually correct response")
    citations: list[SourceCitation] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(
        default_factory=list, description="Common incorrect responses"
    )


class EvalEntry(BaseModel):
    """A single evaluation entry in the reval benchmark.

    This is the core data model that represents one evaluation item.
    Different categories use different fields.
    """

    id: str = Field(..., description="Unique identifier for this eval")
    category: EvalCategory
    country: Country
    version: str = Field("1.0", description="Schema version")

    # Core prompt - used by all categories except policy_attribution
    prompt: str | None = Field(None, description="The prompt to evaluate")

    # Policy Attribution specific
    counterfactual_pair: CounterfactualPair | None = Field(
        None, description="For policy_attribution category"
    )

    # LLM Judge specific (figure_treatment, issue_framing)
    rubric_id: str | None = Field(
        None, description="ID of the rubric to use for scoring"
    )

    # Figure Treatment specific — paired prompts about opposing figures
    figure_pair: FigurePair | None = Field(
        None, description="For figure_treatment category — paired figure prompts"
    )

    # Factual Accuracy specific
    ground_truth: GroundTruth | None = Field(
        None, description="For factual_accuracy category"
    )
    counterfactual_prompt: str | None = Field(
        None, description="Same fact asked differently — tests framing consistency"
    )

    # Argumentation Parity specific
    position_a: str | None = Field(None, description="First position to argue for")
    position_b: str | None = Field(None, description="Opposing position to argue for")

    # Metadata
    topic: str = Field(
        ..., description="Topic area (e.g., 'healthcare', 'immigration')"
    )
    subtopic: str | None = Field(None, description="More specific topic")
    tags: list[str] = Field(default_factory=list, description="Additional tags")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = Field(None, description="Additional notes for eval creators")

    @model_validator(mode="after")
    def validate_category_fields(self) -> EvalEntry:
        """Validate that category-specific required fields are present."""
        cat = self.category
        if cat == EvalCategory.POLICY_ATTRIBUTION and self.counterfactual_pair is None:
            raise ValueError("policy_attribution category requires counterfactual_pair")
        if cat == EvalCategory.FIGURE_TREATMENT and self.figure_pair is None:
            raise ValueError("figure_treatment category requires figure_pair")
        if cat == EvalCategory.FACTUAL_ACCURACY:
            if self.ground_truth is None:
                raise ValueError("factual_accuracy category requires ground_truth")
            if self.counterfactual_prompt is None:
                raise ValueError(
                    "factual_accuracy category requires counterfactual_prompt"
                )
            if self.prompt is None:
                raise ValueError("factual_accuracy category requires a prompt")
        if cat == EvalCategory.ISSUE_FRAMING and self.prompt is None:
            raise ValueError("issue_framing category requires a prompt")
        return self


class EvalResult(BaseModel):
    """Result of running an evaluation."""

    eval_id: str = Field(..., description="ID of the eval that was run")
    model_id: str = Field(..., description="ID of the model being evaluated")
    category: EvalCategory

    # Raw response from the model
    raw_response: str = Field(..., description="The model's raw response")

    # For policy_attribution: responses to both prompts
    response_a: str | None = Field(
        None, description="Response to counterfactual prompt A"
    )
    response_b: str | None = Field(
        None, description="Response to counterfactual prompt B"
    )

    # Scores
    score: float = Field(..., ge=0, le=1, description="Normalized score (0-1)")
    raw_score: float | None = Field(None, description="Raw score before normalization")

    # Scoring details
    scoring_method: ScoringMethod
    similarity_score: float | None = Field(
        None, ge=0, le=1, description="Semantic similarity for policy_attribution"
    )
    rubric_scores: dict[str, float] | None = Field(
        None, description="Per-criterion scores for rubric-based eval"
    )
    judge_reasoning: str | None = Field(
        None, description="LLM judge's reasoning for the score"
    )

    # Factual accuracy: counterfactual framing consistency
    counterfactual_response: str | None = Field(
        None, description="Response to the counterfactual prompt"
    )
    counterfactual_similarity: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Similarity of counterfactual response to ground truth",
    )
    framing_consistency: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Similarity between original and counterfactual responses",
    )

    # Figure treatment: paired scoring
    score_a: float | None = Field(
        None, ge=0, le=1, description="Judge score for figure A"
    )
    score_b: float | None = Field(
        None, ge=0, le=1, description="Judge score for figure B"
    )
    treatment_parity: float | None = Field(
        None,
        ge=0,
        le=1,
        description="How equally both figures were treated (1.0 = equal)",
    )

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int | None = Field(None, description="API call latency in milliseconds")


class BenchmarkRun(RunManifestMixin):
    """A complete benchmark run across multiple evaluations.

    Inherits reproducibility fields (`run_id`, `timestamp`, `git_sha`,
    `model_provider`, `model_id`, `stage_timings`, `error_count`) from
    `RunManifestMixin`. Adding new reproducibility fields to the mixin
    propagates to collector's `GenerationRunManifest` automatically —
    that is the point of the mixin.
    """

    judge_model_id: str | None = Field(None, description="Model used as LLM judge")
    embeddings_model_id: str | None = Field(
        None, description="Model used for embeddings"
    )
    completed_at: datetime | None = None

    # Results
    results: list[EvalResult] = Field(default_factory=list)

    # Aggregate scores by category
    category_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float | None = None

    # Configuration
    eval_ids: list[str] = Field(..., description="List of eval IDs to run")
    country_filter: Country | None = None
    category_filter: EvalCategory | None = None

    # Status
    total_evals: int = 0
    completed_evals: int = 0

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def progress(self) -> float:
        if self.total_evals == 0:
            return 0.0
        return self.completed_evals / self.total_evals
