"""Scoring functions for REVAL evaluations."""

from reval.scoring.judge import BedrockJudge, score_with_judge
from reval.scoring.parity import ParityJudge, score_argumentation_parity
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

__all__ = [
    "BedrockJudge",
    "ParityJudge",
    "SIMILARITY_THRESHOLD",
    "compute_weighted_score",
    "format_rubric_for_judge",
    "is_consistent",
    "load_rubric",
    "load_rubrics_from_directory",
    "score_argumentation_parity",
    "score_policy_attribution",
    "score_with_judge",
]
