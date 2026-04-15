"""Playwright tests for the leaderboard chart overhaul.

Exercises the three new chart surfaces on the static site:

  1. Pareto scatter (overall score × median latency, log x-axis)
  2. Model × category heatmap bound to sortedFilteredRows
  3. Per-criterion horizontal bars on the model detail page

The `site_url` session fixture in `conftest.py` builds the
leaderboard against `multi_judge_showcase`, whose three runs carry
populated `results[*]` with `latency_ms` and raw 1-5 Likert
`rubric_scores`. That gives the Pareto frontier and the
per-criterion bars real data to render.

The `no_latency_site_url` fixture serves a second build whose
runs deliberately have empty `results: []`, and is used only by
`test_scatter_falls_back_when_no_latency_data` to cover the
degraded-mode branch of `renderScatter()`.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


# ── Helpers ──────────────────────────────────────────────────────


def _wait_for_leaderboard_hydration(page: Page) -> None:
    """Wait for Alpine + the leaderboardApp component to mount.

    The `[data-testid='model-row']` selector is load-bearing — it
    only renders once Alpine's `x-for` has iterated the embedded
    JSON data. Without this wait, any locator that queries
    sortedFilteredRows-bound content races the JS init.
    """
    page.wait_for_selector("[data-testid='model-row']")


# ── Chart 1 — Pareto scatter (latency x-axis) ───────────────────


def test_scatter_renders_without_console_errors(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    # Canvas is created with a non-zero bounding box once Chart.js draws.
    canvas = page.locator("#scatterChart")
    expect(canvas).to_be_visible()
    box = canvas.bounding_box()
    assert box is not None and box["width"] > 0 and box["height"] > 0
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"page errors: {js_errors['page']}"


def test_scatter_xaxis_label_is_latency_not_evals(page: Page, site_url: str) -> None:
    """Read the Chart.js instance via page.evaluate and assert the
    x-axis title is the new "median latency" label — NOT the old
    "evals completed" label."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    # Give Chart.js a tick to populate after hydration.
    page.wait_for_function(
        "() => window.Chart && Chart.getChart(document.getElementById('scatterChart'))"
    )
    x_title = page.evaluate("""() => {
            const c = Chart.getChart(document.getElementById('scatterChart'));
            return c && c.options && c.options.scales && c.options.scales.x
                && c.options.scales.x.title && c.options.scales.x.title.text;
        }""")
    assert x_title is not None
    assert "latency" in x_title.lower()
    assert "evals" not in x_title.lower()


def test_scatter_xaxis_is_logarithmic(page: Page, site_url: str) -> None:
    """Log scale is load-bearing — without it the fast models (p50 ~200 ms)
    are invisible next to the slow ones (~4 000 ms)."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    page.wait_for_function(
        "() => window.Chart && Chart.getChart(document.getElementById('scatterChart'))"
    )
    x_type = page.evaluate("""() => {
            const c = Chart.getChart(document.getElementById('scatterChart'));
            return c && c.options && c.options.scales && c.options.scales.x
                && c.options.scales.x.type;
        }""")
    assert x_type == "logarithmic"


def test_pareto_frontier_computation(page: Page, site_url: str) -> None:
    """Unit-test the exposed `paretoFrontier()` helper via page.evaluate.

    The Alpine component exposes `paretoFrontier` as a plain method so
    the test can call it with fixture points rather than pixel-scraping
    the canvas. This is the only direct test of the net-new JS logic
    in this feature — the other scatter tests cover rendering.
    """
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    # Call paretoFrontier via Alpine.$data(el) — the component lives on
    # the outermost `<main x-data="leaderboardApp()">`.
    frontier = page.evaluate("""() => {
            const el = document.querySelector('[x-data*="leaderboardApp"]');
            const app = Alpine.$data(el);
            const pts = [
                { x: 100, y: 0.90, label: 'a' },
                { x: 200, y: 0.80, label: 'b' },
                { x: 500, y: 0.95, label: 'c' },
                { x: 300, y: 0.85, label: 'd' },
            ];
            return app.paretoFrontier(pts).map(p => p.label);
        }""")
    # a (100, 0.90) starts the frontier.  b (200, 0.80) is dominated.
    # d (300, 0.85) is dominated by a.  c (500, 0.95) extends frontier.
    assert frontier == ["a", "c"]


def test_pareto_frontier_empty_input(page: Page, site_url: str) -> None:
    """Empty input → empty frontier. No crash."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    result = page.evaluate("""() => {
            const el = document.querySelector('[x-data*="leaderboardApp"]');
            return Alpine.$data(el).paretoFrontier([]);
        }""")
    assert result == []


def test_pareto_frontier_handles_tied_x(page: Page, site_url: str) -> None:
    """Defensive regression test for tied-latency inputs.

    When two runs land in the same p50 bucket the sort must put the
    higher-y point first so the sweep keeps the better one and drops
    the worse one as dominated. The reviewer flagged this as a
    potential silent correctness gap — this test documents that the
    algorithm resolves ties toward the higher-score point and that
    strictly-dominated points at equal x are correctly excluded.

    Scenario: a=(100, 0.9) and b=(100, 0.7) share latency; b has a
    lower score so a dominates b. c=(200, 0.8) is then dominated by
    a (lower latency + higher score). The frontier is {a} alone.
    """
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    frontier = page.evaluate("""() => {
            const el = document.querySelector('[x-data*="leaderboardApp"]');
            const app = Alpine.$data(el);
            const pts = [
                { x: 100, y: 0.70, label: 'b' },
                { x: 100, y: 0.90, label: 'a' },
                { x: 200, y: 0.80, label: 'c' },
            ];
            return app.paretoFrontier(pts).map(p => p.label);
        }""")
    assert frontier == ["a"]


def test_pareto_frontier_keeps_tied_x_with_tied_y(page: Page, site_url: str) -> None:
    """Two runs with identical (x, y) collapse to the first-seen,
    then a later point with a higher x and higher y DOES extend
    the frontier — the duplicate has the same x but no higher y
    than the first-seen, so it's dropped under strict-greater
    semantics; the later point then passes the strict-greater
    check and gets kept.

    This makes the tie-resolution rule explicit: the frontier is
    a stable staircase, not a duplicate-rung line. V8's sort is
    guaranteed stable since ES2019, so `a` (first in the input)
    comes out first in the sorted order between equal keys.
    """
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    labels = page.evaluate("""() => {
            const el = document.querySelector('[x-data*="leaderboardApp"]');
            const app = Alpine.$data(el);
            const pts = [
                { x: 100, y: 0.80, label: 'a' },
                { x: 100, y: 0.80, label: 'b' },  // exact duplicate
                { x: 200, y: 0.95, label: 'c' },  // extends frontier
            ];
            return app.paretoFrontier(pts).map(p => p.label);
        }""")
    assert labels == ["a", "c"]


def test_scatter_falls_back_when_no_latency_data(
    page: Page, no_latency_site_url: str, js_errors: dict[str, list[str]]
) -> None:
    """`no_latency_site_url` is a dedicated build whose runs have
    empty `results: []`, so every row's `latency_p50_ms` is `None`.
    The scatter must render its 1-D strip plot fallback with a
    visible "latency data not available" subtitle and no console
    errors — crash-safety + graceful degradation in one test."""
    page.goto(f"{no_latency_site_url}/")
    _wait_for_leaderboard_hydration(page)
    # Canvas still mounts.
    canvas = page.locator("#scatterChart")
    expect(canvas).to_be_visible()
    # Fallback marker on the canvas data attribute.
    expect(canvas).to_have_attribute("data-fallback", "true")
    # Fallback note is visible.
    note = page.locator("#scatterFallbackNote")
    expect(note).to_be_visible()
    expect(note).to_contain_text("Latency data not available")
    # No JS errors despite the degraded path.
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"page errors: {js_errors['page']}"


# ── Chart 2 — Heatmap ───────────────────────────────────────────


def test_heatmap_renders_with_row_per_model(page: Page, site_url: str) -> None:
    """One heatmap row per model in the fixture (3 runs = 3 rows).

    Binds to sortedFilteredRows so the count matches the table rows.
    """
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    rows = page.locator(".heatmap-row")
    expect(rows).to_have_count(3)


def test_heatmap_row_count_matches_table_row_count(page: Page, site_url: str) -> None:
    """The heatmap must stay in lockstep with the sorted+filtered
    leaderboard table — typing in the search filter should prune both."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    table_rows_before = page.locator("[data-testid='model-row']").count()
    heatmap_rows_before = page.locator(".heatmap-row").count()
    assert table_rows_before == heatmap_rows_before

    # Filter to just one model via the search box bound to Alpine
    # `search`. Selecting on the x-model attribute is more stable
    # than a placeholder string that may be reworded. The
    # multi_judge_showcase fixture has a gpt-4o row that uniquely
    # matches this query.
    search = page.locator("input[x-model='search']")
    search.fill("gpt-4o")
    # Alpine is synchronous on x-model updates — still, give it a tick.
    page.wait_for_function(
        "() => document.querySelectorAll('[data-testid=\"model-row\"]').length === 1"
    )
    assert page.locator(".heatmap-row").count() == 1


def test_heatmap_cells_use_score_color_classes(page: Page, site_url: str) -> None:
    """Cells carry the existing `.score-high`/`.score-mid`/`.score-low`
    classes, not an invented `score-hi`/`score-lo` variant. This guards
    against the naming drift the plan-reviewer flagged."""
    page.goto(f"{site_url}/")
    _wait_for_leaderboard_hydration(page)
    cells = page.locator(".heatmap-cell").first
    class_name = cells.get_attribute("class") or ""
    # Either high/mid/low/none — but NEVER score-hi or score-lo.
    assert "score-hi" not in class_name.split() or "score-high" in class_name
    assert "score-lo" not in class_name.split() or "score-low" in class_name


# ── Chart 3 — Per-criterion bars on model detail page ──────────


def test_model_page_has_criterion_bars_for_figure_treatment(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    """A per-model page for a run with figure_treatment rubric data must
    render the `<canvas id="criterionBars-figure_treatment">` canvas,
    and Chart.js must produce a non-zero bounding box."""
    # Navigate via the leaderboard table, not a hard-coded URL, so
    # any future slug change doesn't break the test.
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

    # The Figure-A-only caveat must be visible to readers.
    expect(page.locator("text=/Figure A only/i")).to_be_visible()

    # And the parity pointer note is present.
    expect(page.locator("text=/argumentation_parity/i")).to_be_visible()

    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"page errors: {js_errors['page']}"
