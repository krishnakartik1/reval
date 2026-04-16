"""Playwright tests for the leaderboard chart overhaul.

Exercises the four ethics-focused charts on the index page and the
per-criterion bars (with label truncation fix + tooltips) on the
model detail page.

Index charts:
  A. Overall Score Ranking (horizontal bar)
  B. Category Strength Profile (grouped horizontal bar)
  C. Rubric Criterion Heatmap (CSS-grid, declarative Alpine)
  D. Top-5 Category Radar Overlay

The `site_url` session fixture in `conftest.py` builds the leaderboard
against `multi_judge_showcase`, whose three runs carry populated
`results[*]` with `latency_ms` and raw 1-5 Likert `rubric_scores`.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


# ── Helpers ──────────────────────────────────────────────────────


def _wait_for_leaderboard_hydration(page: Page) -> None:
    """Wait for Alpine + the leaderboardApp component to mount."""
    page.wait_for_selector("[data-testid='model-row']")


def _wait_for_chart(page: Page, canvas_id: str) -> None:
    """Wait for a Chart.js instance to be populated on the canvas."""
    page.wait_for_function(
        f"() => window.Chart && Chart.getChart(document.getElementById('{canvas_id}'))"
    )


# ── Chart A — Overall Score Ranking ─────────────────────────────


def test_overall_bar_renders_without_console_errors(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    canvas = page.locator("#overallBarChart")
    expect(canvas).to_be_visible()
    box = canvas.bounding_box()
    assert box is not None and box["width"] > 0 and box["height"] > 0
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"page errors: {js_errors['page']}"


def test_overall_bar_has_correct_data_count(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    _wait_for_chart(page, "overallBarChart")
    count = page.evaluate("""() => {
        const c = Chart.getChart(document.getElementById('overallBarChart'));
        return c.data.datasets[0].data.length;
    }""")
    table_rows = page.locator("[data-testid='model-row']").count()
    assert count == table_rows


# ── Chart B — Category Strength Profile ─────────────────────────


def test_category_grouped_bar_renders(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    canvas = page.locator("#categoryGroupedBar")
    expect(canvas).to_be_visible()
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"


def test_category_grouped_bar_has_datasets_per_category(
    page: Page, site_url: str
) -> None:
    """One dataset per category in the data (fixture has 5 categories)."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    _wait_for_chart(page, "categoryGroupedBar")
    ds_count = page.evaluate("""() => {
        const c = Chart.getChart(document.getElementById('categoryGroupedBar'));
        return c.data.datasets.length;
    }""")
    assert ds_count >= 4


# ── Chart C — Criterion Heatmap ─────────────────────────────────


def test_criterion_heatmap_renders_with_row_per_model(
    page: Page, site_url: str
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    rows = page.locator(".criterion-heatmap-grid .heatmap-row")
    table_rows = page.locator("[data-testid='model-row']").count()
    expect(rows).to_have_count(table_rows)


def test_criterion_heatmap_has_8_criterion_columns(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    sub_headers = page.locator(".criterion-heatmap-sub-header > *")
    # 1 model label + 8 criterion headers = 9
    expect(sub_headers).to_have_count(9)


def test_criterion_heatmap_has_grouped_header(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    group_header = page.locator(".criterion-heatmap-group-header")
    expect(group_header).to_be_visible()
    expect(group_header).to_contain_text("Figure Treatment")
    expect(group_header).to_contain_text("Issue Framing")


def test_criterion_heatmap_cells_use_score_color_classes(
    page: Page, site_url: str
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    cells = page.locator(".criterion-heatmap-grid .heatmap-cell")
    assert cells.count() > 0
    first_class = cells.first.get_attribute("class") or ""
    assert any(
        c in first_class for c in ["score-high", "score-mid", "score-low", "score-none"]
    )


def test_criterion_heatmap_row_count_matches_after_filter(
    page: Page, site_url: str
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    search = page.locator("input[x-model='search']")
    search.fill("gpt-4o")
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=\"model-row\"]').length === 1"
    )
    heatmap_rows = page.locator(".criterion-heatmap-grid .heatmap-row")
    expect(heatmap_rows).to_have_count(1)


# ── Chart D — Top-5 Radar ───────────────────────────────────────


def test_radar_renders_without_console_errors(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    canvas = page.locator("#radarChart")
    expect(canvas).to_be_visible()
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"


def test_radar_dataset_count_matches_top5(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    _wait_for_chart(page, "radarChart")
    ds_count = page.evaluate("""() => {
        const c = Chart.getChart(document.getElementById('radarChart'));
        return c.data.datasets.length;
    }""")
    # Fixture has 3 runs, so radar shows all 3 (< 5 cap)
    assert ds_count == 3


# ── Per-criterion bars on model detail page ──────────────────────


def test_model_page_has_criterion_bars_for_figure_treatment(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    page.locator("[data-testid='model-row']").first.click()
    page.wait_for_url("**/models/*.html")

    figure_canvas = page.locator("#criterionBars-figure_treatment")
    expect(figure_canvas).to_be_visible()
    box = figure_canvas.bounding_box()
    assert box is not None and box["width"] > 0 and box["height"] > 0

    issue_canvas = page.locator("#criterionBars-issue_framing")
    expect(issue_canvas).to_be_visible()

    expect(page.locator("text=/Figure A only/i")).to_be_visible()
    expect(page.locator("text=/argumentation_parity/i")).to_be_visible()

    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"page errors: {js_errors['page']}"


def test_model_page_criterion_bars_have_data_category(
    page: Page, site_url: str
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    page.locator("[data-testid='model-row']").first.click()
    page.wait_for_url("**/models/*.html")

    figure_canvas = page.locator("#criterionBars-figure_treatment")
    expect(figure_canvas).to_have_attribute("data-category", "figure_treatment")
    issue_canvas = page.locator("#criterionBars-issue_framing")
    expect(issue_canvas).to_have_attribute("data-category", "issue_framing")
