"""Unit tests for the Docs tab renderer (`reval.leaderboard.docs`).

No LLM calls and no network. All tests use a temporary `docs/` tree
with hand-crafted markdown fixtures and assert against the emitted
HTML, TOC, and asset output.

Coverage target: ≥ 85% on `src/reval/leaderboard/docs.py` per the
workspace coverage floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "mdit_py_plugins",
    reason="reval[docs] extra not installed — skipping Docs tab renderer tests",
)
pytest.importorskip("markdown_it", reason="reval[docs] extra not installed")
pytest.importorskip("pygments", reason="reval[docs] extra not installed")

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from reval.leaderboard.build import _templates_dir  # noqa: E402
from reval.leaderboard.docs import (  # noqa: E402
    DocPage,
    DocSection,
    TocEntry,
    _extract_toc,
    _highlight_code,
    _parse_front_matter,
    _render_markdown,
    _slugify,
    _wrap_code_blocks_for_copy,
    _write_pygments_css,
    load_docs,
    render_docs,
)

# ── Helpers ──────────────────────────────────────────────────────


def _write(path: Path, body: str) -> Path:
    """Write a UTF-8 text file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _fixture_docs(root: Path) -> Path:
    """Build a minimal two-section docs tree and return its root."""
    _write(
        root / "getting-started" / "_section.yaml",
        "title: Getting started\norder: 1\n",
    )
    _write(
        root / "getting-started" / "install.md",
        "---\ntitle: Install\norder: 1\n---\n"
        "## Prerequisites\n\nYou need Python.\n\n"
        "```bash\npip install reval\n```\n"
        "### Credentials\n\nOptional.\n",
    )
    _write(
        root / "getting-started" / "first-eval.md",
        "---\ntitle: First eval\norder: 2\ndescription: Run a single eval.\n---\n"
        "## Pick a model\n\nLike `claude-haiku-3-5`.\n",
    )
    _write(
        root / "concepts" / "_section.yaml",
        "title: Concepts\norder: 2\n",
    )
    _write(
        root / "concepts" / "methodology.md",
        "---\ntitle: Methodology\norder: 1\n---\n\nShort page.\n",
    )
    return root


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_templates_dir()),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ── `_slugify` + `_extract_toc` + `_render_markdown` ─────────────


class TestSlugify:
    def test_basic_words(self) -> None:
        assert _slugify("Get started locally") == "doc-get-started-locally"

    def test_punctuation_collapses_to_hyphen(self) -> None:
        assert _slugify("Rubrics & metrics") == "doc-rubrics-metrics"

    def test_leading_trailing_non_word_stripped(self) -> None:
        assert _slugify("  --- Hello ---  ") == "doc-hello"

    def test_empty_fallback(self) -> None:
        assert _slugify("") == "doc-section"
        assert _slugify("!!!") == "doc-section"

    def test_mixed_case_lowercased(self) -> None:
        assert _slugify("HTTP Response Codes") == "doc-http-response-codes"


class TestFrontMatter:
    def test_parses_valid_front_matter(self, tmp_path: Path) -> None:
        source = tmp_path / "page.md"
        raw = "---\ntitle: Hello\norder: 1\n---\nBody text.\n"
        meta, body = _parse_front_matter(raw, source)
        assert meta == {"title": "Hello", "order": 1}
        assert body.strip() == "Body text."

    def test_missing_front_matter_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "page.md"
        with pytest.raises(ValueError, match="missing YAML front-matter"):
            _parse_front_matter("No front-matter here.\n", source)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "page.md"
        with pytest.raises(ValueError, match="invalid front-matter YAML"):
            _parse_front_matter(
                "---\ntitle: [unclosed\n---\nBody.\n",
                source,
            )

    def test_non_mapping_front_matter_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "page.md"
        with pytest.raises(ValueError, match="front-matter must be a mapping"):
            _parse_front_matter(
                "---\n- just\n- a\n- list\n---\nBody.\n",
                source,
            )


class TestRenderMarkdown:
    def test_renders_headings_with_doc_anchors(self) -> None:
        html, toc = _render_markdown("## Alpha\n\n### Bravo\n")
        assert '<h2 id="doc-alpha"' in html
        assert '<h3 id="doc-bravo"' in html
        assert len(toc) == 2
        assert toc[0] == TocEntry(level=2, text="Alpha", slug="doc-alpha")
        assert toc[1] == TocEntry(level=3, text="Bravo", slug="doc-bravo")

    def test_skips_h1_and_h4_from_toc(self) -> None:
        _, toc = _render_markdown("# H1\n\n## H2\n\n#### H4\n")
        # Only h2 should be in the TOC
        levels = [e.level for e in toc]
        assert levels == [2]

    def test_fenced_code_uses_pygments(self) -> None:
        html, _ = _render_markdown("```python\nx = 1\n```\n")
        # After `_wrap_code_blocks_for_copy`, `<pre class="hl">` becomes
        # `<pre x-ref="code" class="hl">` — check for the class attr,
        # not a specific prefix.
        assert 'class="hl"' in html
        assert 'class="language-python"' in html
        assert 'class="n"' in html  # pygments name-token span

    def test_unknown_lexer_falls_back_to_text(self) -> None:
        html, _ = _render_markdown("```nonsense-lang-xyz\nwhatever\n```\n")
        assert 'class="hl"' in html
        assert "whatever" in html

    def test_code_blocks_wrapped_for_copy(self) -> None:
        html, _ = _render_markdown("```bash\necho hi\n```\n")
        assert '<div class="code-wrap"' in html
        assert 'x-data="{copied:false}"' in html
        assert 'x-ref="code"' in html
        assert "navigator.clipboard.writeText" in html

    def test_no_code_means_no_copy_wrap(self) -> None:
        html, _ = _render_markdown("Just prose, no code blocks.\n")
        assert "code-wrap" not in html


class TestExtractTocDirect:
    def test_empty_token_stream(self) -> None:
        assert _extract_toc([]) == []


class TestHighlightCode:
    def test_empty_lang_uses_text_lexer(self) -> None:
        html = _highlight_code("plain text\n", "", "")
        assert html.startswith('<pre class="hl">')
        assert "plain text" in html

    def test_known_lang_produces_tokens(self) -> None:
        html = _highlight_code("x = 1\n", "python", "")
        assert "<pre" in html
        assert "language-python" in html


class TestWrapCodeBlocks:
    def test_wraps_single_pre(self) -> None:
        html = _wrap_code_blocks_for_copy('<pre class="hl"><code>x</code></pre>')
        assert '<div class="code-wrap"' in html
        assert 'x-ref="code"' in html
        assert html.count("<pre") == html.count("</pre>") == 1

    def test_wraps_multiple_pre(self) -> None:
        html = _wrap_code_blocks_for_copy(
            '<pre class="hl">a</pre>\n<p>mid</p>\n<pre class="hl">b</pre>'
        )
        assert html.count('<div class="code-wrap"') == 2
        assert html.count("<p>mid</p>") == 1

    def test_preserves_content_outside_pre(self) -> None:
        html = _wrap_code_blocks_for_copy("<p>before</p><pre>code</pre><p>after</p>")
        assert "<p>before</p>" in html
        assert "<p>after</p>" in html


# ── `load_docs` ────────────────────────────────────────────────


class TestLoadDocs:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_docs(tmp_path / "nonexistent") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert load_docs(empty) == []

    def test_loads_sections_in_order(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs")
        sections = load_docs(docs_root)
        assert len(sections) == 2
        assert [s.slug for s in sections] == ["getting-started", "concepts"]
        assert [s.title for s in sections] == ["Getting started", "Concepts"]
        assert [s.order for s in sections] == [1, 2]

    def test_pages_sorted_by_order(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs")
        sections = load_docs(docs_root)
        getting_started = sections[0]
        assert [p.slug for p in getting_started.pages] == ["install", "first-eval"]
        assert [p.order for p in getting_started.pages] == [1, 2]

    def test_page_rel_url_is_depth_aware(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs")
        sections = load_docs(docs_root)
        install = sections[0].pages[0]
        assert install.rel_url == "docs/getting-started/install.html"

    def test_missing_section_yaml_raises(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        (docs_root / "broken").mkdir(parents=True)
        _write(
            docs_root / "broken" / "page.md",
            "---\ntitle: X\norder: 1\n---\nbody\n",
        )
        with pytest.raises(ValueError, match="missing _section.yaml"):
            load_docs(docs_root)

    def test_invalid_section_yaml_raises(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        _write(docs_root / "bad" / "_section.yaml", "title: [unclosed\n")
        with pytest.raises(ValueError, match="invalid YAML"):
            load_docs(docs_root)

    def test_section_missing_title_raises(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        _write(docs_root / "bad" / "_section.yaml", "order: 1\n")
        with pytest.raises(ValueError, match="string `title`"):
            load_docs(docs_root)

    def test_page_missing_title_raises(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        _write(docs_root / "s" / "_section.yaml", "title: S\norder: 1\n")
        _write(
            docs_root / "s" / "p.md",
            "---\norder: 1\n---\nbody\n",
        )
        with pytest.raises(ValueError, match="string `title`"):
            load_docs(docs_root)

    def test_page_non_int_order_raises(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        _write(docs_root / "s" / "_section.yaml", "title: S\norder: 1\n")
        _write(
            docs_root / "s" / "p.md",
            '---\ntitle: X\norder: "two"\n---\nbody\n',
        )
        with pytest.raises(ValueError, match="`order` must be an int"):
            load_docs(docs_root)

    def test_section_mapping_body_required(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        # Top-level scalar is rejected
        _write(docs_root / "bad" / "_section.yaml", "just-a-string\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_docs(docs_root)


# ── `render_docs` ──────────────────────────────────────────────


class TestRenderDocs:
    def test_empty_sections_short_circuits(self, tmp_path: Path) -> None:
        env = _jinja_env()
        render_docs(env, [], tmp_path)
        assert not (tmp_path / "docs").exists()
        assert not (tmp_path / "assets" / "pygments.css").exists()

    def test_writes_landing_and_pages(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        sections = load_docs(docs_root)
        env = _jinja_env()
        render_docs(env, sections, tmp_path / "public")

        public = tmp_path / "public"
        assert (public / "docs" / "index.html").exists()
        assert (public / "docs" / "getting-started" / "install.html").exists()
        assert (public / "docs" / "getting-started" / "first-eval.html").exists()
        assert (public / "docs" / "concepts" / "methodology.html").exists()
        assert (public / "assets" / "pygments.css").exists()

    def test_landing_has_card_grid_no_toc(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        landing = (tmp_path / "public" / "docs" / "index.html").read_text()
        assert 'class="docs-card-grid"' in landing
        assert '<aside class="docs-toc"' not in landing

    def test_page_has_breadcrumb_and_toc(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        install = (
            tmp_path / "public" / "docs" / "getting-started" / "install.html"
        ).read_text()
        assert 'class="docs-breadcrumb' in install
        assert "Getting started" in install
        assert '<aside class="docs-toc"' in install
        # Prerequisites + Credentials headings should yield 2 TOC entries
        assert "doc-prerequisites" in install
        assert "doc-credentials" in install

    def test_sidebar_links_resolve_on_disk(self, tmp_path: Path) -> None:
        """Every sidebar link on every page must point at a real file."""
        import re

        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        public = tmp_path / "public"
        render_docs(env, load_docs(docs_root), public)

        for html_path in (public / "docs").rglob("*.html"):
            html = html_path.read_text()
            links = re.findall(
                r'<a\s+href="([^"]+)"\s+class="docs-sidebar-link',
                html,
            )
            assert links, f"no sidebar links in {html_path}"
            for href in links:
                target = (html_path.parent / href).resolve()
                assert target.exists(), f"broken sidebar link {href!r} from {html_path}"

    def test_code_blocks_rendered_with_copy_button(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        install = (
            tmp_path / "public" / "docs" / "getting-started" / "install.html"
        ).read_text()
        assert 'class="code-wrap"' in install
        assert 'x-data="{copied:false}"' in install
        assert 'x-ref="code"' in install

    def test_pygments_css_contains_both_themes(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        pyg = (tmp_path / "public" / "assets" / "pygments.css").read_text()
        assert ".hl" in pyg
        assert '[data-theme="dark"]' in pyg
        # friendly light theme uses `.hl .k` for keywords
        assert ".hl .k" in pyg

    def test_depth_aware_asset_root_at_depth_2(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        install = (
            tmp_path / "public" / "docs" / "getting-started" / "install.html"
        ).read_text()
        # Depth 2: asset_root resolves through ../../assets
        assert 'href="../../assets/tokens.css"' in install
        assert 'href="../../assets/docs.css"' in install
        assert 'href="../../assets/pygments.css"' in install

    def test_depth_aware_asset_root_at_depth_1(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        landing = (tmp_path / "public" / "docs" / "index.html").read_text()
        # Depth 1: asset_root resolves through ../assets
        assert 'href="../assets/tokens.css"' in landing
        assert 'href="../assets/docs.css"' in landing

    def test_sidebar_filter_script_embedded(self, tmp_path: Path) -> None:
        docs_root = _fixture_docs(tmp_path / "docs_src")
        env = _jinja_env()
        render_docs(env, load_docs(docs_root), tmp_path / "public")
        install = (
            tmp_path / "public" / "docs" / "getting-started" / "install.html"
        ).read_text()
        assert "function docsNav()" in install
        assert 'x-model="q"' in install


# ── `_write_pygments_css` standalone ──────────────────────────


class TestWritePygmentsCss:
    def test_writes_non_empty_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "pygments.css"
        _write_pygments_css(dest)
        content = dest.read_text()
        assert len(content) > 100
        assert ".hl" in content
        assert '[data-theme="dark"]' in content


# ── Dataclass smoke tests ──────────────────────────────────────


class TestDataclasses:
    def test_doc_page_defaults(self) -> None:
        page = DocPage(
            slug="install",
            section_slug="getting-started",
            title="Install",
            order=1,
            rel_url="docs/getting-started/install.html",
            html="<p>hi</p>",
        )
        assert page.toc == []
        assert page.description is None
        assert page.source_path is None

    def test_doc_section_defaults(self) -> None:
        section = DocSection(slug="x", title="X", order=1)
        assert section.pages == []

    def test_toc_entry_is_comparable(self) -> None:
        a = TocEntry(level=2, text="Prereqs", slug="doc-prereqs")
        b = TocEntry(level=2, text="Prereqs", slug="doc-prereqs")
        assert a == b
