"""Playwright tests against the Alpine.js-driven leaderboard index page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


def _wait_for_hydration(page: Page) -> None:
    page.wait_for_selector('[data-testid="judge-pill"]')
    page.wait_for_selector('[data-testid="model-row"]')


def test_no_js_errors_on_load(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/")
    _wait_for_hydration(page)
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"pageerrors: {js_errors['page']}"


def test_judge_pills_render_real_model_id(page: Page, site_url: str) -> None:
    """Regression test for the Bedrock judge=0 bug.

    Before the fix, shortJudge('amazon.nova-lite-v1:0') returned '0'.
    All Bedrock judges rendered as the literal text '0' in the filter pills.
    """
    page.goto(f"{site_url}/")
    _wait_for_hydration(page)

    pills = page.get_by_test_id("judge-pill").locator("span.mono")
    expect(pills).to_have_count(3)

    texts = pills.all_text_contents()
    assert "0" not in texts, f"bug still present — got literal '0' in {texts}"

    assert "amazon.nova-lite-v1:0" in texts
    assert "claude-3-opus" in texts
    assert "claude-3-5-sonnet-20241022" in texts


def test_filter_by_judge_round_trip(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_hydration(page)

    rows = page.get_by_test_id("model-row")
    expect(rows).to_have_count(3)

    bedrock_pill = page.get_by_test_id("judge-pill").filter(
        has=page.locator("span.mono", has_text="amazon.nova-lite-v1:0")
    )
    bedrock_pill.click()

    visible_rows = page.get_by_test_id("model-row")
    expect(visible_rows).to_have_count(1)
    assert visible_rows.first.get_attribute("data-model-id") == "amazon.nova-pro-v1:0"

    bedrock_pill.click()
    expect(page.get_by_test_id("model-row")).to_have_count(3)


def test_sort_by_overall_score(page: Page, site_url: str) -> None:
    page.goto(f"{site_url}/")
    _wait_for_hydration(page)

    def model_id_sequence() -> list[str]:
        return [
            row.get_attribute("data-model-id") or ""
            for row in page.get_by_test_id("model-row").all()
        ]

    descending = [
        "claude-3-5-sonnet-20241022",
        "amazon.nova-pro-v1:0",
        "gpt-4o-2024-11-20",
    ]
    assert model_id_sequence() == descending

    page.get_by_test_id("sort-overall").click()
    page.wait_for_function("""() => {
            const rows = document.querySelectorAll('[data-testid="model-row"]');
            return rows[0]?.getAttribute('data-model-id') === 'gpt-4o-2024-11-20';
        }""")
    assert model_id_sequence() == list(reversed(descending))


def test_per_row_radar_has_one_axis_per_category(
    page: Page, site_url: str, built_site_dir: Path
) -> None:
    leaderboard_data = json.loads(
        (built_site_dir / "data" / "leaderboard.json").read_text()
    )
    expected_axes = len(leaderboard_data["categories"])
    assert expected_axes == 5

    page.goto(f"{site_url}/")
    _wait_for_hydration(page)

    first_row = page.get_by_test_id("model-row").first
    axes = first_row.locator("svg.radar-svg line.radar-axis")
    expect(axes).to_have_count(expected_axes)
