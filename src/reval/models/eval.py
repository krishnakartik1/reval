"""Pydantic models for REVAL evaluation entries."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EvalCategory(str, Enum):
    """Categories of evaluation in REVAL."""

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
    date_accessed: datetime | None = Field(None, description="When the source was accessed")
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
    """A single evaluation entry in the REVAL benchmark.

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
    rubric_id: str | None = Field(None, description="ID of the rubric to use for scoring")

    # Factual Accuracy specific
    ground_truth: GroundTruth | None = Field(
        None, description="For factual_accuracy category"
    )

    # Argumentation Parity specific
    position_a: str | None = Field(None, description="First position to argue for")
    position_b: str | None = Field(None, description="Opposing position to argue for")

    # Metadata
    topic: str = Field(..., description="Topic area (e.g., 'healthcare', 'immigration')")
    subtopic: str | None = Field(None, description="More specific topic")
    tags: list[str] = Field(default_factory=list, description="Additional tags")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = Field(None, description="Additional notes for eval creators")

    @field_validator("counterfactual_pair")
    @classmethod
    def validate_counterfactual_for_policy_attribution(
        cls, v: CounterfactualPair | None, info
    ) -> CounterfactualPair | None:
        if info.data.get("category") == EvalCategory.POLICY_ATTRIBUTION and v is None:
            raise ValueError("policy_attribution category requires counterfactual_pair")
        return v

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth_for_factual_accuracy(
        cls, v: GroundTruth | None, info
    ) -> GroundTruth | None:
        if info.data.get("category") == EvalCategory.FACTUAL_ACCURACY and v is None:
            raise ValueError("factual_accuracy category requires ground_truth")
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt_for_non_policy_attribution(
        cls, v: str | None, info
    ) -> str | None:
        category = info.data.get("category")
        needs_prompt = category in [
            EvalCategory.FIGURE_TREATMENT,
            EvalCategory.ISSUE_FRAMING,
            EvalCategory.FACTUAL_ACCURACY,
        ]
        if needs_prompt and v is None:
            raise ValueError(f"{category} category requires a prompt")
        return v


class EvalResult(BaseModel):
    """Result of running an evaluation."""

    eval_id: str = Field(..., description="ID of the eval that was run")
    model_id: str = Field(..., description="ID of the model being evaluated")
    category: EvalCategory

    # Raw response from the model
    raw_response: str = Field(..., description="The model's raw response")

    # For policy_attribution: responses to both prompts
    response_a: str | None = Field(None, description="Response to counterfactual prompt A")
    response_b: str | None = Field(None, description="Response to counterfactual prompt B")

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

    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int | None = Field(None, description="API call latency in milliseconds")


class BenchmarkRun(BaseModel):
    """A complete benchmark run across multiple evaluations."""

    run_id: str = Field(..., description="Unique identifier for this run")
    model_id: str = Field(..., description="Model being benchmarked")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    failed_evals: int = 0

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    @property
    def progress(self) -> float:
        if self.total_evals == 0:
            return 0.0
        return self.completed_evals / self.total_evals
