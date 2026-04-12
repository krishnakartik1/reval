"""REVAL - Robust Evaluation of Values and Alignment in LLMs.

A benchmark for measuring political bias in LLMs using fact-aligned scoring.
"""

__version__ = "0.1.0"

from reval.models.eval import (
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

__all__ = [
    "BenchmarkRun",
    "Country",
    "CounterfactualPair",
    "EvalCategory",
    "EvalEntry",
    "EvalResult",
    "FigurePair",
    "GroundTruth",
    "GroundTruthLevel",
    "Rubric",
    "RubricCriterion",
    "ScoringMethod",
    "SourceCitation",
]
