"""Factory for constructing `BenchmarkRun` in tests with sentinel mixin fields.

The `RunManifestMixin` adds required fields (`git_sha`, `model_provider`,
`timestamp`) that aren't meaningful inside most unit tests. Using this helper
keeps individual tests focused on whatever they actually care about while
still satisfying Pydantic validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from reval.contracts import BenchmarkRun


def make_benchmark_run(**overrides: Any) -> BenchmarkRun:
    """Construct a `BenchmarkRun` filled with sentinel reproducibility fields.

    Any keyword in `overrides` replaces the sentinel value. `eval_ids`
    defaults to `["test-eval"]` so tests that don't care about it still
    satisfy the required-field constraint.
    """
    defaults: dict[str, Any] = {
        "run_id": "test-run",
        "timestamp": datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc),
        "git_sha": "test-sha",
        "model_provider": "test",
        "model_id": "test-model",
        "eval_ids": ["test-eval"],
    }
    defaults.update(overrides)
    return BenchmarkRun(**defaults)
