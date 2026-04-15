"""Per-run HTML report (reval.report.generate_html_report) load-time validation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from playwright.sync_api import Page

from reval.contracts.models import (
    BenchmarkRun,
    EvalCategory,
    EvalResult,
    ScoringMethod,
)
from reval.report import generate_html_report

from .conftest import CATEGORIES

pytestmark = pytest.mark.ui


def test_generated_report_loads_clean(
    page: Page, tmp_path: Path, js_errors: dict[str, list[str]]
) -> None:
    results = [
        EvalResult(
            eval_id=f"fixture-{cat.value}",
            model_id="amazon.nova-pro-v1:0",
            category=cat,
            raw_response="fixture response",
            score=0.80,
            scoring_method=ScoringMethod.LLM_JUDGE,
        )
        for cat in EvalCategory
    ]
    run = BenchmarkRun(
        run_id="per-run-fixture",
        timestamp=datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 4, 14, 12, 5, 0, tzinfo=timezone.utc),
        git_sha="testfixture",
        model_provider="bedrock",
        model_id="amazon.nova-pro-v1:0",
        judge_model_id="amazon.nova-lite-v1:0",
        eval_ids=[r.eval_id for r in results],
        results=results,
        overall_score=0.80,
        category_scores={cat: 0.80 for cat in CATEGORIES},
        total_evals=len(results),
        completed_evals=len(results),
    )

    output = tmp_path / "report.html"
    generate_html_report(run, output)

    page.goto(f"file://{output}")
    page.wait_for_load_state("networkidle")

    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"pageerrors: {js_errors['page']}"

    html = page.content()
    for cat in EvalCategory:
        assert (
            cat.value in html
        ), f"category {cat.value!r} missing from generated report"

    assert 'data-lucide="circle"' not in html, (
        "report rendered the fallback 'circle' icon — a category is missing "
        "from reval.report._CATEGORY_ICONS"
    )
