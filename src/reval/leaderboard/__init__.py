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

from reval.leaderboard.build import LeaderboardRow, build, load_rows

__all__ = ["LeaderboardRow", "build", "load_rows"]
