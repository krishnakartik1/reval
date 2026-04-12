"""reval.contracts — shared data contracts with zero runtime dependencies.

This namespace is the source of truth for types both reval and
reval-factual-collector rely on. It intentionally imports only `pydantic` +
stdlib; see `tests/test_contracts_imports.py` for the zero-dep guard.
"""

from reval.contracts.manifest import RunManifestMixin, get_git_sha
from reval.contracts.models import (
    BenchmarkRun,
    CounterfactualPair,
    Country,
    EvalCategory,
    EvalEntry,
    EvalResult,
    FigurePair,
    GroundTruth,
    GroundTruthLevel,
    Rubric,
    RubricCriterion,
    ScoringMethod,
    SourceCitation,
)
from reval.contracts.provider import (
    CompletionResult,
    LLMProvider,
    RateLimitError,
)

__all__ = [
    "BenchmarkRun",
    "CompletionResult",
    "CounterfactualPair",
    "Country",
    "EvalCategory",
    "EvalEntry",
    "EvalResult",
    "FigurePair",
    "GroundTruth",
    "GroundTruthLevel",
    "LLMProvider",
    "RateLimitError",
    "Rubric",
    "RubricCriterion",
    "RunManifestMixin",
    "ScoringMethod",
    "SourceCitation",
    "get_git_sha",
]
