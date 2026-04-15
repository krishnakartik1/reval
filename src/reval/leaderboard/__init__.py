"""Static leaderboard generator for REVAL.

Reads `showcase/*/results.json` files produced by `reval run`, expands
Jinja2 templates, and writes a `public/` directory of static HTML + JSON
+ CSS + JS ready for SFTP/rsync to any web host. No runtime server
required — the leaderboard is plain files on disk.

Primary entry point is the `reval leaderboard build` CLI subcommand
(see `reval.cli`). The `build.build()` function is the library-level
entry point for callers that want to invoke the generator
programmatically (used by the deferred `reval-webui` plan).
"""

from reval.leaderboard.build import LeaderboardRow, build, get_style_css, load_rows

# Docs-tab symbols (`DocPage`, `DocSection`, `TocEntry`, `load_docs`,
# `render_docs`) are intentionally NOT re-exported from this package's
# top-level namespace. They live in `reval.leaderboard.docs` and must
# be imported from there explicitly.
#
# Rationale: `docs.py` is lazily imported by `build.build()` and by the
# CLI layer so that the `[docs]` optional extra (`markdown-it-py`,
# `mdit-py-plugins`, `pygments`) is only required when a docs build
# actually runs. Eagerly re-exporting from this __init__ would defeat
# that — any `from reval.leaderboard import build` would pull in
# `docs.py`, and once step 3 adds top-level `markdown_it` / `pygments`
# imports to that module, users without `[docs]` would hit an
# `ImportError` just importing the leaderboard.

__all__ = ["LeaderboardRow", "build", "get_style_css", "load_rows"]
