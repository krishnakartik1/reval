"""Docs tab renderer for the static leaderboard site.

Walks a `docs/` directory of markdown sources with YAML front-matter,
renders each page to HTML via `markdown-it-py`, and emits
`public/docs/<section>/<slug>.html` plus a `public/docs/index.html`
landing page. Also writes `public/assets/pygments.css` at build time
so code blocks have syntax-highlight styles without requiring the
source `assets/` directory to be writable (wheel installs bundle that
path read-only).

The module is imported lazily by `reval.leaderboard.build.build()` so
that the `[docs]` optional extra (`markdown-it-py`, `mdit-py-plugins`,
`pygments`) is only required when a docs tree is actually present.

Architecture:

    reval/docs/                    load_docs()            DocSection[]
    ├── getting-started/           ──────────▶            ├── pages: DocPage[]
    │   ├── _section.yaml                                 │   ├── html (str)
    │   ├── install.md                                    │   └── toc (TocEntry[])
    │   └── first-eval.md
    └── concepts/
        ├── _section.yaml                   render_docs()        public/docs/
        └── rubrics.md                      ──────────▶           ├── index.html
                                                                  ├── getting-started/
                                                                  │   ├── install.html
                                                                  │   └── first-eval.html
                                                                  └── concepts/
                                                                      └── rubrics.html

Rendering pipeline: front-matter via a tiny regex, body via
`markdown-it-py` + `mdit-py-plugins` anchors (h2/h3, `doc-` prefixed),
Pygments syntax highlighting with `nowrap=True` (so the highlighted
`<pre class="hl">` slots directly into markdown-it's raw-HTML
passthrough instead of being re-wrapped), TOC extraction by walking
the token stream, and an Alpine-powered copy-button scaffold wrapped
around every `<pre>` during post-processing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from jinja2 import TemplateNotFound

if TYPE_CHECKING:
    from jinja2 import Environment

logger = logging.getLogger(__name__)


_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*\n(?P<rest>.*)\Z",
    re.DOTALL,
)


@dataclass
class TocEntry:
    """One entry in a page's right-hand "on this page" TOC."""

    level: int  # 2 or 3
    text: str
    slug: str


@dataclass
class DocPage:
    """One rendered markdown page under `docs/<section>/<slug>.md`."""

    slug: str
    section_slug: str
    title: str
    order: int
    rel_url: str  # e.g. "docs/concepts/rubrics.html"
    html: str  # rendered HTML body
    toc: list[TocEntry] = field(default_factory=list)
    description: str | None = None
    source_path: Path | None = None


@dataclass
class DocSection:
    """One section of the docs — a directory under `docs/`."""

    slug: str
    title: str
    order: int
    pages: list[DocPage] = field(default_factory=list)


def _parse_front_matter(text: str, source: Path) -> tuple[dict, str]:
    """Split a markdown file into `(front_matter_dict, body)`.

    Raises `ValueError` with a file path if the opening `---` block is
    missing or the YAML body fails to parse.
    """
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise ValueError(
            f"{source}: missing YAML front-matter block. Every docs page "
            "must start with `---\\ntitle: ...\\norder: ...\\n---`."
        )
    try:
        meta = yaml.safe_load(match.group("body")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: invalid front-matter YAML — {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{source}: front-matter must be a mapping, got {type(meta)}")
    return meta, match.group("rest")


def _slugify(text: str) -> str:
    """Return a kebab-case, `doc-` prefixed slug for a heading.

    The `doc-` prefix avoids colliding with anchors on non-docs pages
    (e.g. leaderboard category filters) if the same document is ever
    inlined elsewhere. Matches `[a-z0-9-]+` only; runs of non-word
    characters collapse to a single hyphen.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return f"doc-{slug}" if slug else "doc-section"


def _highlight_code(code: str, lang: str, _attrs: str) -> str:
    """Pygments callback for markdown-it fenced code blocks.

    Returns an HTML string starting with `<pre`, which markdown-it
    uses **as-is** (without its default `<pre><code class="language-…">`
    wrapper — see markdown-it's `highlight` option contract). If we
    returned a div-wrapped Pygments payload instead, markdown-it would
    double-wrap the `<pre>` tag and the output would break layout.

    We therefore call Pygments with `nowrap=True` (no surrounding `<pre>`
    or `<div>` from Pygments) and assemble the wrapper ourselves as
    `<pre class="hl"><code class="language-…">…spans…</code></pre>`.
    The `.hl` class scopes `pygments.css` rules (`.hl .k`, `.hl .c1`,
    …) to fenced code blocks only, so Pygments styles never leak into
    leaderboard tables. Unknown / missing languages fall back to the
    `text` lexer.
    """
    from html import escape

    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import TextLexer, get_lexer_by_name
    from pygments.util import ClassNotFound

    try:
        lexer = get_lexer_by_name(lang, stripall=True) if lang else TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    formatter = HtmlFormatter(cssclass="hl", nowrap=True)
    inner = highlight(code, lexer, formatter).rstrip("\n")
    lang_class = f' class="language-{escape(lang)}"' if lang else ""
    return f'<pre class="hl"><code{lang_class}>{inner}</code></pre>'


def _build_markdown_renderer():
    """Construct a `markdown-it-py` instance configured for the docs tab.

    Configured with:
      - `commonmark` preset (stable baseline)
      - `linkify` + `tables` enabled (user-friendly defaults)
      - `anchors_plugin` from `mdit-py-plugins` attaching `id="doc-…"`
        slugs to h2 and h3 headings only (h1 is the page title, rendered
        by the Jinja template, not inside the markdown body)
      - A Pygments `highlight` callback for fenced code blocks
    """
    from markdown_it import MarkdownIt
    from mdit_py_plugins.anchors import anchors_plugin

    md = (
        MarkdownIt("commonmark", {"linkify": True, "html": False})
        .enable("table")
        .enable("strikethrough")
    )
    md.options["highlight"] = _highlight_code
    md.use(
        anchors_plugin,
        min_level=2,
        max_level=3,
        slug_func=_slugify,
        permalink=False,
    )
    return md


def _extract_toc(tokens: list) -> list[TocEntry]:
    """Walk a markdown-it token stream, return the h2/h3 TOC.

    Each heading emits a triple of tokens in order:
      `heading_open` (tag=h2|h3, attrs including `id` from the
      anchors plugin) → `inline` (with the heading text) → `heading_close`.

    We match `heading_open` at tags h2/h3, read the `id` attr from the
    anchor plugin, then pull the plain-text rendering of the next
    `inline` token as the heading text.
    """
    entries: list[TocEntry] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open" and tok.tag in ("h2", "h3"):
            level = int(tok.tag[1])
            anchor_id = ""
            if tok.attrs:
                # Token attrs is a dict in markdown-it-py >= 2.x.
                if isinstance(tok.attrs, dict):
                    anchor_id = str(tok.attrs.get("id", ""))
                else:  # legacy list of [key, value] pairs
                    for k, v in tok.attrs:
                        if k == "id":
                            anchor_id = str(v)
                            break
            text = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                text = tokens[i + 1].content
            if anchor_id and text:
                entries.append(TocEntry(level=level, text=text, slug=anchor_id))
        i += 1
    return entries


_PRE_WRAP_RE = re.compile(
    r"(<pre[^>]*>.*?</pre>)",
    re.DOTALL,
)

# Relative markdown-to-markdown link. Matches `foo.md`,
# `../concepts/rubrics.md`, `foo.md#anchor`, but NOT absolute URLs
# (`http://…`, `https://…`, `mailto:…`), root-anchored paths (`/foo.md`),
# or bare anchors (`#section`). The anchors plugin emits heading ids
# but the body still carries hand-written `.md` links from the source
# markdown — we rewrite both ends: the href suffix and any `#anchor`
# tail is preserved verbatim.
_INTERNAL_MD_LINK_RE = re.compile(r"^(?!https?://|mailto:|/|#)(.+?)\.md(#.*)?$")


def _rewrite_md_links(tokens: list) -> None:
    """In-place rewrite relative `.md` hrefs to `.html` on `link_open` tokens.

    Walks block and inline token trees recursively — markdown-it emits
    `link_open` tokens inside an inline token's `children` list, not at
    the top level, so a non-recursive loop would miss every link in the
    document body.

    Source markdown like `[install](install.md)` and
    `[config](../reference/config.md#judge)` is written against the
    source tree layout, but the build step emits `.html` files. Without
    this rewrite, every internal docs link 404s on Cloudflare (the host
    serves `.md` as `text/plain` or returns 404 depending on
    configuration — see revalbench.com preview deploy 2c1ec109).
    """
    for tok in tokens:
        if tok.type == "link_open" and tok.attrs:
            _rewrite_link_attrs(tok)
        children = getattr(tok, "children", None)
        if children:
            _rewrite_md_links(children)


def _rewrite_link_attrs(token) -> None:
    """Rewrite a single `link_open` token's `href` attr from `.md` to `.html`.

    Handles both the modern dict-shaped `attrs` (markdown-it-py ≥ 2.x)
    and the legacy list-of-pairs shape, so the rewrite survives minor
    version bumps without bisecting.
    """
    if isinstance(token.attrs, dict):
        href = token.attrs.get("href")
        if isinstance(href, str):
            new_href = _INTERNAL_MD_LINK_RE.sub(r"\1.html\2", href)
            if new_href != href:
                token.attrs["href"] = new_href
        return
    # Legacy list-of-[key, value]-pairs format.
    for i, pair in enumerate(token.attrs):
        if not pair or pair[0] != "href":
            continue
        href = pair[1]
        if not isinstance(href, str):
            continue
        new_href = _INTERNAL_MD_LINK_RE.sub(r"\1.html\2", href)
        if new_href != href:
            token.attrs[i] = [pair[0], new_href]


def _wrap_code_blocks_for_copy(html: str) -> str:
    """Wrap every top-level `<pre>` in a copy-button Alpine scaffold.

    Uses `str.replace`-style regex substitution rather than an HTML
    parser — the input is our own trusted markdown-it output, and
    `<pre>` never nests. The wrapper's Alpine state (`x-data="{copied:
    false}"`) is scoped per-block so each code block has its own
    copied indicator.
    """

    def _wrap(match: re.Match[str]) -> str:
        pre = match.group(1)
        return (
            '<div class="code-wrap" x-data="{copied:false}">'
            '<button type="button" class="code-copy" aria-label="Copy code"'
            ' @click="navigator.clipboard.writeText($refs.code.innerText);'
            ' copied=true; setTimeout(()=>copied=false,1200)">'
            '<i data-lucide="copy" x-show="!copied" class="w-4 h-4"></i>'
            '<i data-lucide="check" x-show="copied" class="w-4 h-4"></i>'
            "</button>" + pre.replace("<pre", '<pre x-ref="code"', 1) + "</div>"
        )

    return _PRE_WRAP_RE.sub(_wrap, html)


def _render_markdown(body: str) -> tuple[str, list[TocEntry]]:
    """Render a markdown body to HTML and extract its h2/h3 TOC.

    Steps:
      1. Tokenize via `markdown-it-py` (configured in
         `_build_markdown_renderer`).
      2. Walk tokens to extract the h2/h3 TOC.
      3. Render tokens → HTML (Pygments is applied via the
         `highlight` callback during render).
      4. Post-process HTML to wrap each `<pre>` in a copy-button
         Alpine scaffold for the docs tab UI.
    """
    md = _build_markdown_renderer()
    env: dict = {}
    tokens = md.parse(body, env)
    _rewrite_md_links(tokens)
    toc = _extract_toc(tokens)
    html = md.renderer.render(tokens, md.options, env)
    html = _wrap_code_blocks_for_copy(html)
    return html, toc


def load_docs(docs_dir: Path) -> list[DocSection]:
    """Walk `docs_dir` and return an ordered list of `DocSection`s.

    Every subdirectory of `docs_dir` is a section and must contain a
    `_section.yaml` with `title` + `order`. Every `*.md` file under a
    section must begin with YAML front-matter providing `title` and
    `order`. Sections are sorted by `order` (tiebreak: slug); pages
    inside a section are sorted by `order` (tiebreak: slug).

    Raises `ValueError` on missing front-matter or missing section
    metadata, with a file path in the message.
    """
    if not docs_dir.exists() or not docs_dir.is_dir():
        return []

    sections: list[DocSection] = []
    for section_dir in sorted(docs_dir.iterdir()):
        if not section_dir.is_dir():
            continue
        section_meta_path = section_dir / "_section.yaml"
        if not section_meta_path.exists():
            raise ValueError(
                f"{section_dir}: missing _section.yaml. Every docs section "
                "directory must declare its title and order."
            )
        try:
            section_meta = yaml.safe_load(section_meta_path.read_text("utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{section_meta_path}: invalid YAML — {exc}") from exc
        if not isinstance(section_meta, dict):
            raise ValueError(
                f"{section_meta_path}: must be a mapping with `title` and `order`."
            )

        section_title = section_meta.get("title")
        section_order = section_meta.get("order")
        if not isinstance(section_title, str) or not isinstance(section_order, int):
            raise ValueError(
                f"{section_meta_path}: requires string `title` and int `order`."
            )

        section = DocSection(
            slug=section_dir.name,
            title=section_title,
            order=section_order,
        )

        for md_path in sorted(section_dir.glob("*.md")):
            raw = md_path.read_text("utf-8")
            meta, body = _parse_front_matter(raw, md_path)
            page_title = meta.get("title")
            page_order = meta.get("order", 0)
            if not isinstance(page_title, str):
                raise ValueError(
                    f"{md_path}: front-matter must include a string `title`."
                )
            if not isinstance(page_order, int):
                raise ValueError(f"{md_path}: front-matter `order` must be an int.")

            page_slug = md_path.stem
            html, toc = _render_markdown(body)
            section.pages.append(
                DocPage(
                    slug=page_slug,
                    section_slug=section.slug,
                    title=page_title,
                    order=page_order,
                    rel_url=f"docs/{section.slug}/{page_slug}.html",
                    html=html,
                    toc=toc,
                    description=meta.get("description"),
                    source_path=md_path,
                )
            )

        section.pages.sort(key=lambda p: (p.order, p.slug))
        sections.append(section)

    sections.sort(key=lambda s: (s.order, s.slug))
    return sections


def render_docs(
    env: Environment,
    sections: list[DocSection],
    output_dir: Path,
) -> None:
    """Render `docs/index.html` and `docs/<section>/<slug>.html`.

    Also writes `<output_dir>/assets/pygments.css` at build time
    (NOT into the source `_assets_dir()` — that path is read-only on
    wheel installs under `site-packages`). `docs.css` is a hand-
    authored checked-in stylesheet under `src/reval/leaderboard/assets/`
    and is copied into `<output_dir>/assets/` by `build()`'s existing
    asset-copy loop, not by this function.

    Ordering constraint: `render_docs` must run AFTER `build()`'s
    asset-copy loop so that `pygments.css` — written here into the
    output directory — is not overwritten by a subsequent copy pass.
    `build()` is responsible for call order.
    """
    if not sections:
        return

    # Defensive guard for intermediate mid-step states where `docs.py`
    # (and thus `load_docs`) has landed but the docs templates
    # (`docs_index.html.j2`, `docs_page.html.j2`) have not yet been
    # added to `src/reval/leaderboard/templates/`. Without this guard,
    # running `reval leaderboard build` against a populated
    # `reval/docs/` tree would crash with `TemplateNotFound` instead
    # of gracefully skipping the docs render. Once the templates
    # exist, this branch never triggers on a correctly-installed
    # build and costs nothing at runtime.
    try:
        index_tpl = env.get_template("docs_index.html.j2")
        page_tpl = env.get_template("docs_page.html.j2")
    except TemplateNotFound as exc:
        logger.warning(
            "Docs templates not found (%s) — skipping docs render. "
            "This is expected during incremental development of the "
            "Docs tab feature; install the templates under "
            "src/reval/leaderboard/templates/ to enable docs output.",
            exc.name,
        )
        return

    docs_root = output_dir / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Write pygments.css at build time. Step 1 stub: an empty file.
    # Step 3 replaces this with `HtmlFormatter().get_style_defs('.hl')`.
    _write_pygments_css(output_dir / "assets" / "pygments.css")

    # Landing page — depth 1 (public/docs/index.html).
    (docs_root / "index.html").write_text(
        index_tpl.render(
            nav_sections=sections,
            current_page=None,
            asset_root="../assets",
            index_href="../index.html",
            docs_href="index.html",
        ),
        encoding="utf-8",
    )

    # Per-page pages — depth 2 (public/docs/<section>/<slug>.html).
    for section in sections:
        section_out = docs_root / section.slug
        section_out.mkdir(parents=True, exist_ok=True)
        for page in section.pages:
            (section_out / f"{page.slug}.html").write_text(
                page_tpl.render(
                    nav_sections=sections,
                    current_page=page,
                    asset_root="../../assets",
                    index_href="../../index.html",
                    docs_href="../index.html",
                ),
                encoding="utf-8",
            )


def _write_pygments_css(dest: Path) -> None:
    """Write the Pygments syntax-highlight stylesheet to ``dest``.

    Generates two stylesheets scoped to the ``.hl`` class:
      * A light-theme palette (default / `data-theme="light"`)
      * A dark-theme override under `:root[data-theme="dark"] .hl`

    Both are concatenated into one file so a single `<link>` tag in
    ``docs_base.html.j2`` loads both themes and the active one is
    selected by the existing theme toggle in ``base.html.j2``. We use
    ``friendly`` for light and ``monokai`` for dark — both ship with
    Pygments, no extra deps.
    """
    from pygments.formatters import HtmlFormatter

    light = HtmlFormatter(style="friendly", cssclass="hl").get_style_defs(".hl")
    dark_defs = HtmlFormatter(style="monokai", cssclass="hl").get_style_defs(".hl")
    # Scope the dark overrides under the explicit theme selector so the
    # light palette applies by default and dark takes over only when the
    # user has toggled the theme.
    dark = "\n".join(
        ':root[data-theme="dark"] ' + line if line.strip().startswith(".hl") else line
        for line in dark_defs.splitlines()
    )
    content = (
        "/* Auto-generated by reval.leaderboard.docs._write_pygments_css.\n"
        " * Light theme: Pygments `friendly`. Dark theme: Pygments `monokai`.\n"
        " * Scoped to `.hl` so these rules only touch fenced code blocks\n"
        " * inside docs prose; they do NOT affect the leaderboard tables.\n"
        " */\n\n" + light + "\n\n" + dark + "\n"
    )
    dest.write_text(content, encoding="utf-8")


__all__ = [
    "DocPage",
    "DocSection",
    "TocEntry",
    "load_docs",
    "render_docs",
]
