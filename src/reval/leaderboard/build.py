"""Build a static leaderboard site from `showcase/*/results.json`.

Flow:

    showcase/                   build()                 public/
    ├── <run_A>/               ────────────▶           ├── index.html    ← leaderboard table + JS sort
    │   ├── results.json                               ├── models/
    │   └── report.html                                │   ├── <run_A>.html
    ├── <run_B>/                                       │   └── <run_B>.html
    │   └── ...                                        ├── data/
    └── ...                                            │   └── leaderboard.json
                                                       ├── reports/    ← copied from showcase/<run>/report.html
                                                       │   └── <run_A>.html
                                                       └── assets/
                                                           ├── style.css
                                                           └── sort.js

The output is pure static files — no Python runtime on the host. The
generator is the only code that runs; deployment is `rsync public/
user@host:public_html/`.

`load_rows()` is library-importable: the deferred `reval-webui` plan
reuses the same Pydantic model + loader to render an interactive
leaderboard tab against the same `showcase/` data source.
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

#: Maximum number of recent runs shown per model in the leaderboard.
#: Currently unused (every row is a distinct run), but reserved for a
#: future "show N most recent runs per model" filter.
_MAX_ROWS_PER_MODEL = 10


class LeaderboardRow(BaseModel):
    """One row in the leaderboard table.

    Each row corresponds to one `results.json` file under `showcase/`.
    A single model can have multiple rows (e.g. one per judge or one
    per git_sha) — the leaderboard page groups + filters client-side
    via the data-* attributes emitted into the HTML.
    """

    #: Directory name of the showcase entry — used as the slug for
    #: `public/models/<slug>.html` per-model pages. Guaranteed unique
    #: across runs because `save_run_outputs` timestamps each run dir.
    slug: str
    model_id: str
    model_provider: str
    judge_model_id: str | None = None
    embeddings_model_id: str | None = None
    overall_score: float | None = None
    category_scores: dict[str, float] = Field(default_factory=dict)
    total_evals: int = 0
    completed_evals: int = 0
    error_count: int = 0
    timestamp: str | None = None
    git_sha: str | None = None
    #: Path to the `report.html` file (relative to the output dir), or
    #: None if the showcase entry had no report file. The index + per-
    #: model pages link to this if set.
    report_href: str | None = None


def load_rows(showcase_dir: Path) -> list[LeaderboardRow]:
    """Scan a showcase directory and return one row per `results.json`.

    Directories without a `results.json` are skipped silently. Rows are
    returned in no particular order — the index page sorts them
    client-side (default: overall score descending). For missing
    optional fields (no `overall_score`, no `category_scores`), we emit
    the row with `None` / empty dict so the UI can show "—" without
    crashing.
    """
    if not showcase_dir.exists():
        return []

    rows: list[LeaderboardRow] = []
    for entry in sorted(showcase_dir.iterdir()):
        if not entry.is_dir():
            continue
        results_path = entry / "results.json"
        if not results_path.exists():
            continue

        try:
            with results_path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Corrupt file or unreadable — skip rather than crash the
            # whole build. The user will notice a missing row.
            continue

        report_path = entry / "report.html"
        row = LeaderboardRow(
            slug=entry.name,
            model_id=data.get("model_id", "unknown"),
            model_provider=data.get("model_provider", "unknown"),
            judge_model_id=data.get("judge_model_id"),
            embeddings_model_id=data.get("embeddings_model_id"),
            overall_score=data.get("overall_score"),
            category_scores=data.get("category_scores") or {},
            total_evals=data.get("total_evals", 0),
            completed_evals=data.get("completed_evals", 0),
            error_count=data.get("error_count", 0),
            timestamp=data.get("timestamp"),
            git_sha=data.get("git_sha"),
            report_href=f"reports/{entry.name}.html" if report_path.exists() else None,
        )
        rows.append(row)

    return rows


def _collect_categories(rows: list[LeaderboardRow]) -> list[str]:
    """Return the union of category names across all rows, sorted.

    The column set on the index page is stable even if different rows
    were run with different `--category` filters — missing cells render
    as "—".
    """
    categories: set[str] = set()
    for row in rows:
        categories.update(row.category_scores.keys())
    return sorted(categories)


def _templates_dir() -> Path:
    """Return the filesystem path to the bundled Jinja2 templates.

    Works for both editable installs (where the templates live next to
    this module in the source tree) and wheel installs (where hatch's
    `force-include` copies them alongside the installed `reval/` package).
    """
    try:
        # importlib.resources.files() returns a Traversable — for a
        # file-based package it behaves like a Path.
        anchor = resources.files("reval.leaderboard").joinpath("templates")
        return Path(str(anchor))
    except (ModuleNotFoundError, AttributeError):
        return Path(__file__).parent / "templates"


def _assets_dir() -> Path:
    """Return the filesystem path to the bundled CSS + JS assets."""
    try:
        anchor = resources.files("reval.leaderboard").joinpath("assets")
        return Path(str(anchor))
    except (ModuleNotFoundError, AttributeError):
        return Path(__file__).parent / "assets"


def get_style_css() -> str:
    """Return the leaderboard `style.css` contents as a string.

    Public helper so `reval.report.generate_html_report` can inline the
    same CSS palette into individual run reports without duplicating the
    file. Keeps the leaderboard + reports visually in sync — edit
    `leaderboard/assets/style.css` once, both surfaces update.
    """
    return (_assets_dir() / "style.css").read_text(encoding="utf-8")


def build(
    showcase_dir: Path,
    output_dir: Path,
    include_reports: bool = True,
) -> None:
    """Render the static leaderboard site.

    Args:
        showcase_dir: Directory containing per-run subdirectories with
            `results.json` (and optionally `report.html`) files.
        output_dir: Destination directory. Created if missing. Existing
            contents are **overwritten file-by-file**; not wiped, so
            sibling files (e.g. a user's `CNAME` or `robots.txt`) are
            preserved.
        include_reports: If True (default), copy each showcase entry's
            `report.html` into `public/reports/<slug>.html` so the
            per-model pages can link to the full dashboard.
    """
    rows = load_rows(showcase_dir)
    categories = _collect_categories(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)

    # Jinja2 env
    env = Environment(
        loader=FileSystemLoader(_templates_dir()),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["score_color"] = _score_color
    env.filters["fmt_score"] = _fmt_score

    # index.html — the main leaderboard table
    index_tpl = env.get_template("index.html.j2")
    (output_dir / "index.html").write_text(
        index_tpl.render(rows=rows, categories=categories),
        encoding="utf-8",
    )

    # data/leaderboard.json — raw rows for any JS-side consumer (or
    # the deferred reval-webui Streamlit leaderboard tab).
    leaderboard_json = {
        "rows": [row.model_dump(mode="json") for row in rows],
        "categories": categories,
    }
    (output_dir / "data" / "leaderboard.json").write_text(
        json.dumps(leaderboard_json, indent=2),
        encoding="utf-8",
    )

    # Per-model detail pages
    model_tpl = env.get_template("model.html.j2")
    for row in rows:
        (output_dir / "models" / f"{row.slug}.html").write_text(
            model_tpl.render(row=row, categories=categories),
            encoding="utf-8",
        )

    # Copy assets (CSS + JS)
    assets_src = _assets_dir()
    if assets_src.exists():
        for asset_file in assets_src.iterdir():
            if asset_file.is_file():
                shutil.copy2(asset_file, output_dir / "assets" / asset_file.name)

    # Copy report.html files from showcase/ into public/reports/ so the
    # per-model pages can link to the full interactive dashboard.
    if include_reports:
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        for row in rows:
            src = showcase_dir / row.slug / "report.html"
            if src.exists():
                shutil.copy2(src, reports_dir / f"{row.slug}.html")


# ── Template filters ────────────────────────────────────────────────────


def _score_color(score: float | None) -> str:
    """Return a Tailwind-ish color class based on the score band."""
    if score is None:
        return "score-none"
    if score >= 0.85:
        return "score-high"
    if score >= 0.70:
        return "score-mid"
    return "score-low"


def _fmt_score(score: float | None) -> str:
    """Format a score as `0.XXX` or `—` for None."""
    if score is None:
        return "—"
    return f"{score:.3f}"
