"""Run manifest scaffolding shared between reval and reval-collector.

Anything here must have zero dependencies on `aioboto3`, `boto3`, `numpy`,
`jsonlines`, `httpx`, `anthropic`, or `openai` — `reval.contracts` is
asserted to be a thin, pure namespace. See `tests/test_contracts_imports.py`.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RunManifestMixin(BaseModel):
    """Reproducibility fields shared across reval and collector run records.

    This is a trait, not a container. Repo-specific top-level run records
    inherit from it to gain the fields flat at the top level. It is not meant
    to be instantiated directly.

    Fields:
        run_id: Unique identifier for this run (e.g. `uuid.uuid4().hex`).
        timestamp: When the run started, timezone-aware.
        git_sha: Output of `git describe --always --dirty --abbrev=12` or
            `"unknown"` if git is unavailable. The `-dirty` suffix is the
            reproducibility guardrail — it flags that the working tree had
            uncommitted changes when the run happened.
        model_provider: API surface identifier (`"bedrock"`, `"anthropic"`,
            `"openai"`, `"minimax"`). NOT the model vendor — the same Claude
            model can be reached via both `"bedrock"` and `"anthropic"`.
            `model_id` carries the vendor signal.
        model_id: The provider-specific model identifier.
        stage_timings: Optional map from stage name to seconds.
        error_count: Number of errors encountered during the run.
    """

    run_id: str
    timestamp: datetime
    git_sha: str
    model_provider: str
    model_id: str
    stage_timings: dict[str, float] = Field(default_factory=dict)
    error_count: int = 0


def get_git_sha(cwd: Path | None = None) -> str:
    """Return `git describe --always --dirty --abbrev=12` or `"unknown"`.

    The `--dirty` suffix flags uncommitted working-tree changes. `--always`
    falls back to a plain commit hash when no tags exist (verified to honor
    `--abbrev=12` in fallback mode).

    Known blindspot: `--dirty` only flags tracked-file modifications. An
    untracked rubric or config can influence a run without triggering the
    suffix. This is deliberately out of scope — hashing the working tree
    is exactly the complexity this module avoids.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("get_git_sha failed: %s", exc)
        return "unknown"
    if result.returncode != 0:
        logger.debug("get_git_sha non-zero exit: %s", result.stderr.strip())
        return "unknown"
    return result.stdout.strip() or "unknown"
