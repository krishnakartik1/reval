"""Playwright tests for the Docs tab on the static leaderboard site.

Complements the Python-layer tests in `tests/test_docs_build.py` and
`tests/test_leaderboard_build.py::TestNavTabBar` by asserting against
the **post-JS DOM** — Alpine.js hydration, Lucide icon rendering,
client-side sidebar filtering, copy-to-clipboard state transitions,
and cross-page navigation. String-presence tests on Jinja output
cannot catch the class of hydration and reactivity bugs this suite
targets.

The `built_site_dir` fixture in `conftest.py` builds against the
repo's real `reval/docs/` tree, so these tests exercise the
production content, not a synthetic fixture. Content drift is
surfaced directly.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.ui


# ── Helpers ──────────────────────────────────────────────────────


def _wait_for_docs_hydration(page: Page) -> None:
    """Wait until Alpine + Lucide have hydrated the docs DOM.

    The sidebar and Alpine state are the two things that need to be
    live before any interactive test can run. We wait on a
    hydrated-only selector (the sidebar `<input>` whose
    `x-model="q"` binding is Alpine-managed) plus the `aria-label`
    on the nav region.
    """
    page.wait_for_selector(".docs-sidebar input[type='search']")
    page.wait_for_selector("nav[aria-label='Docs navigation']")


# ── Landing page ─────────────────────────────────────────────────


def test_docs_landing_no_js_errors(
    page: Page, site_url: str, js_errors: dict[str, list[str]]
) -> None:
    page.goto(f"{site_url}/docs/index.html")
    _wait_for_docs_hydration(page)
    assert js_errors["console"] == [], f"console errors: {js_errors['console']}"
    assert js_errors["page"] == [], f"pageerrors: {js_errors['page']}"


def test_docs_landing_renders_card_grid_with_docs_tab_active(
    page: Page, site_url: str
) -> None:
    """Landing page shows the Pinecone-style card grid, and the
    primary nav tab bar highlights Docs (not Leaderboard)."""
    page.goto(f"{site_url}/docs/index.html")
    _wait_for_docs_hydration(page)

    # Card grid present with at least 6 cards
    cards = page.locator(".docs-card")
    expect(cards).to_have_count(6)

    # Nav tab bar: Docs is the active tab, Leaderboard is not
    docs_tab = page.locator("nav a.docs-tab", has_text="Docs")
    leaderboard_tab = page.locator("nav a.docs-tab", has_text="Leaderboard")
    expect(docs_tab).to_have_class("docs-tab is-active")
    expect(leaderboard_tab).to_have_class("docs-tab")


def test_docs_landing_has_sidebar_section_headers(page: Page, site_url: str) -> None:
    """Sidebar shows the four section headings from `reval/docs/`."""
    page.goto(f"{site_url}/docs/index.html")
    _wait_for_docs_hydration(page)

    sidebar = page.locator(".docs-sidebar")
    expect(sidebar.get_by_text("Getting started", exact=True)).to_be_visible()
    expect(sidebar.get_by_text("Concepts", exact=True)).to_be_visible()
    expect(sidebar.get_by_text("Reference", exact=True)).to_be_visible()
    expect(sidebar.get_by_text("Roadmap", exact=True)).to_be_visible()


# ── Page rendering ───────────────────────────────────────────────


def test_docs_page_highlights_current_in_sidebar(page: Page, site_url: str) -> None:
    """`install.html` must highlight the Install link in the sidebar
    via the `.is-active` class, and the breadcrumb must read
    `Docs › Getting started › Install`."""
    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    active_link = page.locator("a.docs-sidebar-link.is-active")
    expect(active_link).to_have_count(1)
    expect(active_link).to_have_text("Install")

    breadcrumb = page.locator(".docs-breadcrumb")
    expect(breadcrumb).to_contain_text("Docs")
    expect(breadcrumb).to_contain_text("Getting started")
    expect(breadcrumb).to_contain_text("Install")


def test_docs_page_renders_pygments_and_toc(page: Page, site_url: str) -> None:
    """Fenced code blocks get Pygments highlight classes and the
    right-hand 'On this page' TOC is populated from h2/h3 anchors."""
    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    # At least one Pygments-highlighted code block. `.hl` is the
    # Pygments cssclass; `.c1` / `.nb` / `.k` are token classes that
    # appear in the install page's bash blocks.
    hl_blocks = page.locator("pre.hl")
    assert hl_blocks.count() >= 1, "no Pygments-highlighted code block"

    # Right-hand TOC renders with at least one anchor link
    toc = page.locator(".docs-toc")
    expect(toc).to_be_visible()
    toc_links = toc.locator("a[href^='#doc-']")
    assert toc_links.count() >= 1, "docs TOC has no h2/h3 anchor links"


# ── Alpine sidebar filter ────────────────────────────────────────


def test_sidebar_filter_hides_non_matching_links(page: Page, site_url: str) -> None:
    """Typing in the sidebar search input hides links whose title
    doesn't contain the query. This is the Alpine hydration test —
    the `matches()` handler and `x-show` binding must both be live.
    """
    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    # Before filter: both Install (Getting started) and Rubrics
    # (Concepts) are visible.
    install_link = page.locator("a.docs-sidebar-link", has_text="Install")
    rubrics_link = page.locator("a.docs-sidebar-link", has_text="Rubrics")
    expect(install_link.first).to_be_visible()
    expect(rubrics_link.first).to_be_visible()

    # Type "rubr" — matches Rubrics (which shares a section blob with
    # Concepts), hides Install.
    search = page.locator(".docs-sidebar input[type='search']")
    search.fill("rubr")

    expect(rubrics_link.first).to_be_visible()
    expect(install_link.first).to_be_hidden()

    # Clear the filter — both visible again.
    search.fill("")
    expect(install_link.first).to_be_visible()
    expect(rubrics_link.first).to_be_visible()


# ── Copy-to-clipboard button ─────────────────────────────────────


def test_copy_button_click_toggles_state(page: Page, site_url: str) -> None:
    """Clicking a copy button flips Alpine's `copied` state from
    false to true, swapping the check icon for the copy icon.

    We assert state via `x-show` (which sets inline `display: none`)
    on the two `<i>` children, NOT via the real clipboard content —
    headless Chromium's clipboard grant flow is context-scoped and
    not worth the fixture complication just to validate the toggle.
    """
    # Grant clipboard-write so navigator.clipboard.writeText() resolves
    # rather than rejecting (which would skip the `copied=true` line
    # in the inline handler).
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    # First code-wrap on the page. The install page has several bash
    # blocks, so [0] is deterministic.
    wrap = page.locator(".code-wrap").first
    expect(wrap).to_be_visible()

    button = wrap.locator("button.code-copy")
    # Before click: copy icon visible, check icon hidden
    copy_icon = wrap.locator("[data-lucide='copy']")
    check_icon = wrap.locator("[data-lucide='check']")
    expect(copy_icon).to_be_visible()
    expect(check_icon).to_be_hidden()

    button.click()

    # After click: check icon visible, copy icon hidden
    expect(check_icon).to_be_visible()
    expect(copy_icon).to_be_hidden()


# ── Cross-page nav transitions ──────────────────────────────────


def test_docs_tab_click_from_leaderboard(page: Page, site_url: str) -> None:
    """Clicking the Docs tab from the leaderboard index lands on
    `docs/index.html`. Regression guard for the depth-0 href."""
    page.goto(f"{site_url}/")
    # Wait for the leaderboard's Alpine hydration so the nav is live.
    page.wait_for_selector("[data-testid='model-row']")

    page.locator("nav a.docs-tab", has_text="Docs").click()
    page.wait_for_url("**/docs/index.html")
    _wait_for_docs_hydration(page)

    expect(page.locator(".docs-card-grid")).to_be_visible()


def test_docs_tab_click_from_model_page(page: Page, site_url: str) -> None:
    """Clicking the Docs tab from a per-model page (`public/models/<slug>.html`)
    navigates to `public/docs/index.html` — NOT `public/models/docs/...`.

    This is the depth-1 regression guard for the blocker-2 fix. If the
    `{% set docs_href = "../docs/index.html" %}` line in `model.html.j2`
    is ever dropped, this test fails with a 404 or a "path not found".
    """
    # The leaderboard index has one anchor per model row pointing at
    # `models/<slug>.html`. Click the first one.
    page.goto(f"{site_url}/")
    page.wait_for_selector("[data-testid='model-row']")
    page.locator("[data-testid='model-row']").first.click()
    page.wait_for_url("**/models/*.html")

    # Now on a model page. Click the Docs tab.
    docs_tab = page.locator("nav a.docs-tab", has_text="Docs")
    expect(docs_tab).to_be_visible()
    docs_tab.click()
    page.wait_for_url("**/docs/index.html")
    _wait_for_docs_hydration(page)

    # Docs card grid should be visible and the model-page `wrap` should not
    expect(page.locator(".docs-card-grid")).to_be_visible()


# ── In-page markdown link rewriting ─────────────────────────────


def test_article_has_no_md_hrefs(page: Page, site_url: str) -> None:
    """Regression for the Cloudflare preview bug where internal
    docs-to-docs links shipped as `.md` hrefs and 404'd on the host.

    Checks every anchor inside `.docs-article` on a page that we know
    contains hand-written markdown links (`install.md` has
    `[Run your first eval](first-eval.md)` and
    `[Methodology](../concepts/methodology.md)`). None of them may
    end in `.md` or contain `.md#` — the renderer must rewrite them
    to `.html` before they hit disk.
    """
    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    offenders = page.evaluate("""() => {
            const links = document.querySelectorAll('.docs-prose a[href]');
            const bad = [];
            for (const a of links) {
                const href = a.getAttribute('href') || '';
                // Skip absolute URLs, mailto:, bare anchors, and
                // root-anchored paths — those are intentionally not
                // rewritten.
                if (
                    href.startsWith('http://') ||
                    href.startsWith('https://') ||
                    href.startsWith('mailto:') ||
                    href.startsWith('#') ||
                    href.startsWith('/')
                ) continue;
                if (href.endsWith('.md') || href.includes('.md#')) {
                    bad.push(href);
                }
            }
            return bad;
        }""")
    assert offenders == [], f"Bare .md hrefs leaked into the article: {offenders}"


def test_internal_article_link_navigates_not_404(page: Page, site_url: str) -> None:
    """End-to-end click-through: follow an internal link inside the
    prose and verify the target loads cleanly (not 404, and with the
    docs layout hydrating on the destination).

    The install page's body has `[Run your first eval](first-eval.md)` —
    after rewriting it should resolve to `first-eval.html` in the same
    directory, which is a real built page.
    """
    page.goto(f"{site_url}/docs/getting-started/install.html")
    _wait_for_docs_hydration(page)

    # Find the first prose anchor whose href is relative (our
    # rewriter's target). Scope to `.docs-prose` — the markdown body
    # wrapper — so we skip the breadcrumb (which lives in
    # `.docs-article` but outside `.docs-prose` and points to the
    # docs landing page, which has no `current_page` and therefore
    # no breadcrumb itself — a click there would fail this test for
    # the wrong reason).
    target = page.locator(
        ".docs-prose a[href$='.html'], .docs-prose a[href*='.html#']"
    ).first
    expect(target).to_be_visible()
    href = target.get_attribute("href")
    assert href is not None, "no eligible in-article link found"
    assert not href.endswith(".md"), f"link href is still .md: {href}"

    # Capture the response status for the navigation triggered by
    # the click so we can assert 200 rather than silently accepting
    # a 404 HTML page.
    with page.expect_response(lambda r: r.request.resource_type == "document") as info:
        target.click()
    response = info.value
    assert (
        response.status == 200
    ), f"internal article link {href} returned HTTP {response.status}"

    # And the destination must be a real docs page — breadcrumb
    # confirms the layout hydrated against a `current_page` that the
    # builder knew about.
    _wait_for_docs_hydration(page)
    expect(page.locator(".docs-breadcrumb")).to_be_visible()


def test_leaderboard_tab_click_from_docs(page: Page, site_url: str) -> None:
    """Round-trip: Docs landing → Leaderboard tab → leaderboard index."""
    page.goto(f"{site_url}/docs/index.html")
    _wait_for_docs_hydration(page)

    page.locator("nav a.docs-tab", has_text="Leaderboard").click()
    page.wait_for_url(lambda u: u.rstrip("/").endswith("index.html") or u.endswith("/"))

    # The leaderboard-specific testid is the source of truth that we
    # landed on the right page.
    page.wait_for_selector("[data-testid='model-row']")
