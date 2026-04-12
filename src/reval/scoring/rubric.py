"""Rubric-based scoring utilities."""

import json
from pathlib import Path

import yaml

from reval.models.eval import Rubric


def load_rubric(rubric_path: str | Path) -> Rubric:
    """Load a rubric from a YAML or JSON file.

    Args:
        rubric_path: Path to the rubric file

    Returns:
        Rubric object
    """
    path = Path(rubric_path)
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    elif path.suffix == ".json":
        data = json.loads(content)
    else:
        raise ValueError(f"Unsupported rubric format: {path.suffix}")

    return Rubric(**data)


def load_rubrics_from_directory(rubrics_dir: str | Path) -> dict[str, Rubric]:
    """Load all rubrics from a directory.

    Args:
        rubrics_dir: Path to directory containing rubric files

    Returns:
        Dictionary mapping rubric IDs (filename without extension) to Rubric objects
    """
    rubrics_path = Path(rubrics_dir)
    rubrics = {}

    for file_path in rubrics_path.iterdir():
        if file_path.suffix in (".yaml", ".yml", ".json"):
            rubric_id = file_path.stem
            rubrics[rubric_id] = load_rubric(file_path)

    return rubrics


def compute_weighted_score(
    criterion_scores: dict[str, int],
    rubric: Rubric,
) -> float:
    """Compute weighted score from individual criterion scores.

    Args:
        criterion_scores: Map of criterion name to score (1-5)
        rubric: The rubric being used

    Returns:
        Normalized score between 0 and 1
    """
    total_weighted = 0.0
    total_weight = 0.0

    for criterion in rubric.criteria:
        if criterion.name in criterion_scores:
            score = criterion_scores[criterion.name]
            # Normalize score from 1-5 to 0-1
            normalized = (score - 1) / 4
            total_weighted += normalized * criterion.weight
            total_weight += criterion.weight

    if total_weight == 0:
        return 0.0

    return total_weighted / total_weight


def format_rubric_for_judge(rubric: Rubric) -> str:
    """Format a rubric as a prompt for the LLM judge.

    Args:
        rubric: The rubric to format

    Returns:
        Formatted string for inclusion in judge prompt
    """
    lines = [
        f"# Evaluation Rubric: {rubric.name}",
        "",
        rubric.description,
        "",
        "## Criteria",
        "",
    ]

    for i, criterion in enumerate(rubric.criteria, 1):
        lines.extend(
            [
                f"### {i}. {criterion.name} (weight: {criterion.weight})",
                "",
                criterion.description,
                "",
                "Scoring:",
                f"- 1 (Poor): {criterion.score_1}",
                f"- 3 (Acceptable): {criterion.score_3}",
                f"- 5 (Excellent): {criterion.score_5}",
                "",
            ]
        )

    return "\n".join(lines)
