"""Per-model detail page — only a smoke test. Chart.js canvas is out of scope."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.ui


def test_no_js_errors_on_load(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/models/bedrock_nova.html")
    page.wait_for_load_state("networkidle")
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"pageerrors: {js_errors['page']}"
