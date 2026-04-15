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
import logging
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from statistics import median

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Maximum number of recent runs shown per model in the leaderboard.
#: Currently unused (every row is a distinct run), but reserved for a
#: future "show N most recent runs per model" filter.
_MAX_ROWS_PER_MODEL = 10


@dataclass
class BuildReport:
    """Diagnostic counters emitted by `build()` for the CLI to print.

    Only populated when `dataset_dir` is provided to `build()`. With no
    dataset, every per-run report is a verbatim copy of the showcase
    `report.html` and drift can't be measured.
    """

    #: Slugs where some (but not all) eval IDs from the run still exist
    #: in the current dataset. Tuple: (slug, matched_count, total_count).
    #: Cards for the missing IDs render without a Test case section.
    partial_matches: list[tuple[str, int, int]] = field(default_factory=list)

    #: Slugs with zero dataset matches where we fell back to copying
    #: `showcase/<slug>/report.html`. The copy may be stale.
    unmatched_copied: list[str] = field(default_factory=list)

    #: Slugs with zero dataset matches AND no showcase `report.html` to
    #: fall back to — no per-run report is published at all.
    unmatched_missing: list[str] = field(default_factory=list)


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

    #: Median latency in ms across every `EvalResult.latency_ms` in the
    #: run. `None` when no result recorded a latency — older showcase
    #: entries written before the field existed, or runs where every
    #: provider path failed to record it. The scatter chart on the
    #: index page falls back to a 1-D strip plot when this is `None`.
    latency_p50_ms: float | None = None

    #: Per-category per-criterion mean scores, normalized to 0-1 from
    #: the judge's raw 1-5 Likert output. Shape:
    #: `{"figure_treatment": {"factual_accuracy": 0.86, ...},
    #:   "issue_framing": {"perspective_coverage": 0.74, ...}}`.
    #: Only `figure_treatment` and `issue_framing` populate this —
    #: `argumentation_parity` has `rubric_scores` too but its dotted
    #: key shape (`A.logic`, `B.logic`) does not fit a flat horizontal
    #: bar chart, so it is excluded by whitelist.
    aggregated_rubric_scores: dict[str, dict[str, float]] = Field(default_factory=dict)


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
        raw_results = data.get("results") or []
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
            latency_p50_ms=_median_latency(raw_results),
            aggregated_rubric_scores=_aggregate_rubric_scores(raw_results),
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


#: Categories whose `rubric_scores` flatten cleanly into a per-criterion
#: horizontal bar. `argumentation_parity` also populates `rubric_scores`
#: but with dotted `A.<metric>` / `B.<metric>` keys (see
#: `reval.scoring.parity.score_argumentation_parity`), so it is excluded
#: by whitelist — a parity-shaped panel is a v2 follow-up.
_RUBRIC_BAR_CATEGORIES = frozenset({"figure_treatment", "issue_framing"})


def _median_latency(results: list[dict]) -> float | None:
    """Median of `latency_ms` across raw result dicts that recorded it.

    Operates on raw result dicts rather than typed `EvalResult`, so it
    plugs into `load_rows()`'s existing lazy JSON parse (results.json
    is never validated through the full `BenchmarkRun` model).

    Tolerates three sparsity modes:
      - result has `latency_ms = None` (recent runs where the field was
        left unpopulated on a failure branch)
      - result omits the `latency_ms` key entirely (legacy runs written
        before the field was introduced)
      - empty `results` list

    In all three cases the function returns `None`, which the scatter
    chart renders as a "latency data not available" fallback.
    """
    latencies = [
        r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), int | float)
    ]
    if not latencies:
        return None
    return float(median(latencies))


def _aggregate_rubric_scores(
    results: list[dict],
) -> dict[str, dict[str, float]]:
    """Per-category per-criterion mean scores, normalized to 0-1.

    The LLM judge emits raw 1-5 Likert ints into `rubric_scores` (see
    `reval.scoring.judge.score_with_judge` line 169). This helper
    normalizes each score via `(raw - 1) / 4` so the leaderboard charts
    share a 0-1 palette with category scores.

    Only `figure_treatment` and `issue_framing` are aggregated
    (`_RUBRIC_BAR_CATEGORIES`). `argumentation_parity` is excluded by
    whitelist because its dotted-key shape does not fit a flat
    horizontal bar chart — a parity panel is a v2 task.

    Out-of-range raw scores (< 1 or > 5) are dropped with a warning log
    rather than clamped, so judge drift or parser coercion bugs surface
    instead of hiding as silent green bars. Non-numeric values are
    skipped the same way.

    Categories with zero valid scores are absent from the return value
    (not empty-dict), so the frontend can use
    `cat in aggregated_rubric_scores` as its "has data" predicate.

    Known limitation: the `figure_treatment` rubric_scores dict stored
    on a paired result reflects ONLY side A's response. See
    `reval.runner.EvalRunner._run_figure_treatment`, which currently
    constructs the merged result with
    `rubric_scores=result_a.rubric_scores` — grep for that assignment
    if you need to find the exact line in a future refactor. The
    aggregate therefore measures "how the model treats Figure A
    across paired prompts", not a balanced Figure-A/Figure-B mean.
    Fixing the runner to emit a merged rubric is a v2 follow-up.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for r in results:
        category = r.get("category")
        if category not in _RUBRIC_BAR_CATEGORIES:
            continue
        rubric_scores = r.get("rubric_scores") or {}
        if not rubric_scores:
            continue
        bucket = buckets.setdefault(category, {})
        for criterion, raw in rubric_scores.items():
            # bool is a subclass of int in Python — check it first so a
            # judge that accidentally returns True/False doesn't sneak
            # through the numeric arm as 1.0 / 0.0.
            if isinstance(raw, bool):
                continue
            if not isinstance(raw, int | float):
                continue
            raw_f = float(raw)
            if not (1.0 <= raw_f <= 5.0):
                logger.warning(
                    "rubric_scores out-of-range: category=%s "
                    "criterion=%s raw=%s — dropped from aggregate",
                    category,
                    criterion,
                    raw,
                )
                continue
            normalized = (raw_f - 1.0) / 4.0
            bucket.setdefault(criterion, []).append(normalized)

    return {
        category: {
            criterion: sum(values) / len(values)
            for criterion, values in criteria.items()
        }
        for category, criteria in buckets.items()
        if criteria
    }


def _average_category_scores(
    rows: list[LeaderboardRow], categories: list[str]
) -> dict[str, float]:
    """Per-category mean score across the rows that have a value.

    Used by the per-model page's radar chart (this model vs leaderboard
    average) and the "delta vs avg" column. Categories with no data
    points are omitted from the result.
    """
    result: dict[str, float] = {}
    for cat in categories:
        values = [
            row.category_scores[cat]
            for row in rows
            if cat in row.category_scores and row.category_scores[cat] is not None
        ]
        if values:
            result[cat] = sum(values) / len(values)
    return result


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
    """Return the concatenated brand stylesheet as a string.

    Combines `tokens.css` (the design token layer — CSS custom
    properties for colors, typography, spacing, etc.) with `style.css`
    (custom utilities and component styles that consume those tokens)
    so that `reval.report.generate_html_report` can inline a single
    `<style>` block and produce a fully self-contained report.

    The leaderboard itself loads the two files as separate
    `<link rel="stylesheet">` tags (see `base.html.j2`); this helper
    only matters for the per-run report, which is shipped as one HTML
    file.
    """
    assets = _assets_dir()
    tokens = (assets / "tokens.css").read_text(encoding="utf-8")
    style = (assets / "style.css").read_text(encoding="utf-8")
    return tokens + "\n\n" + style


def build(
    showcase_dir: Path,
    output_dir: Path,
    include_reports: bool = True,
    dataset_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> BuildReport:
    """Render the static leaderboard site.

    Args:
        showcase_dir: Directory containing per-run subdirectories with
            `results.json` (and optionally `report.html`) files.
        output_dir: Destination directory. Created if missing. Existing
            contents are **overwritten file-by-file**; not wiped, so
            sibling files (e.g. a user's `CNAME` or `robots.txt`) are
            preserved.
        include_reports: If True (default), put each showcase entry's
            report into `public/reports/<slug>.html` so the per-model
            pages can link to the full dashboard.
        dataset_dir: Optional path to the `evals/datasets/` directory.
            When provided AND `include_reports=True`, each report is
            **regenerated** from `showcase/<slug>/results.json` against
            the current dataset (so the Test case section of every
            expandable result card shows the actual prompts sent to the
            model). When None, `public/reports/<slug>.html` is a
            verbatim copy of `showcase/<slug>/report.html` (old
            behavior, kept as a fallback when no dataset is available).
        docs_dir: Optional path to the `docs/` directory containing
            markdown source for the Docs tab. When provided and the
            directory exists, `render_docs()` emits `public/docs/` as
            a sibling of `public/index.html`. When None or missing,
            the Docs nav tab in `base.html.j2` still renders (its
            target is the default `docs/index.html` path) but nothing
            is written under `public/docs/`. The docs build runs
            AFTER the asset-copy loop so `pygments.css`, which
            `render_docs` writes into `output_dir/assets/`, survives.
    """
    build_report = BuildReport()

    rows = load_rows(showcase_dir)
    categories = _collect_categories(rows)
    avg_scores = _average_category_scores(rows, categories)

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

    # Row data as plain dicts — embedded in the index page so the Alpine
    # component can render the reactive table without a fetch() call
    # (which would fail for file:// previews). Also written to
    # data/leaderboard.json so external consumers (the deferred
    # reval-webui tab, custom dashboards) can import it.
    leaderboard_data = {
        "rows": [row.model_dump(mode="json") for row in rows],
        "categories": categories,
    }

    # index.html — the main leaderboard table
    index_tpl = env.get_template("index.html.j2")
    (output_dir / "index.html").write_text(
        index_tpl.render(
            rows=rows,
            categories=categories,
            leaderboard_data=leaderboard_data,
        ),
        encoding="utf-8",
    )

    # data/leaderboard.json — raw rows for any JS-side consumer (or
    # the deferred reval-webui Streamlit leaderboard tab).
    (output_dir / "data" / "leaderboard.json").write_text(
        json.dumps(leaderboard_data, indent=2),
        encoding="utf-8",
    )

    # Per-model detail pages
    model_tpl = env.get_template("model.html.j2")
    for row in rows:
        (output_dir / "models" / f"{row.slug}.html").write_text(
            model_tpl.render(
                row=row,
                categories=categories,
                row_data=row.model_dump(mode="json"),
                avg_data=avg_scores,
            ),
            encoding="utf-8",
        )

    # Copy assets (CSS + JS)
    assets_src = _assets_dir()
    if assets_src.exists():
        for asset_file in assets_src.iterdir():
            if asset_file.is_file():
                shutil.copy2(asset_file, output_dir / "assets" / asset_file.name)

    # Populate public/reports/ with per-run HTML reports.
    #
    # When a dataset directory is provided we regenerate each report
    # from results.json + matching EvalEntry objects so that the Test
    # case section of every result card shows the actual prompts
    # sent to the model. This keeps the leaderboard build
    # idempotent and future-proof: stale showcase reports (written
    # before PR #18's 3-section expansion, or without evals=) get
    # refreshed on every build.
    #
    # When no dataset directory is provided, fall back to a verbatim
    # copy of the showcase's report.html — preserves the old behavior
    # so callers that don't care about prompts don't need to change.
    if include_reports:
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        matched_evals_by_slug: dict[str, list] = {}
        eval_ids_by_slug: dict[str, list[str]] = {}
        dataset_provided = dataset_dir is not None and dataset_dir.exists()
        if dataset_provided and dataset_dir is not None:
            # Lazy import — `reval.runner` pulls in provider / judge
            # SDKs, which `reval leaderboard build` would otherwise
            # have no reason to load. Importing inside `build()`
            # keeps the standalone leaderboard command lightweight.
            from reval.runner import load_evals_from_directory

            all_evals = load_evals_from_directory(dataset_dir, None, None)
            eval_by_id = {e.id: e for e in all_evals}
            for row in rows:
                results_path = showcase_dir / row.slug / "results.json"
                if not results_path.exists():
                    continue
                try:
                    with results_path.open() as f:
                        run_data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                eval_ids = run_data.get("eval_ids", [])
                eval_ids_by_slug[row.slug] = eval_ids
                matched = [eval_by_id[eid] for eid in eval_ids if eid in eval_by_id]
                if matched:
                    matched_evals_by_slug[row.slug] = matched

        # Lazy import to avoid circular dependency at module-load time
        # (reval.report imports reval.leaderboard.get_style_css).
        from reval.contracts import BenchmarkRun
        from reval.report import generate_html_report

        for row in rows:
            src_results = showcase_dir / row.slug / "results.json"
            src_report = showcase_dir / row.slug / "report.html"
            dest = reports_dir / f"{row.slug}.html"

            matched_for_row = matched_evals_by_slug.get(row.slug)
            if matched_for_row and src_results.exists():
                try:
                    with src_results.open() as f:
                        run_data = json.load(f)
                    run = BenchmarkRun.model_validate(run_data)
                    generate_html_report(run, dest, evals=matched_for_row)
                    total_ids = len(eval_ids_by_slug.get(row.slug, []))
                    if total_ids > len(matched_for_row):
                        build_report.partial_matches.append(
                            (row.slug, len(matched_for_row), total_ids)
                        )
                    continue
                except Exception:  # noqa: BLE001
                    # Fall through to copy on any rendering error so a
                    # single broken run doesn't kill the whole build.
                    pass

            # Track drift only when a dataset was provided AND the run
            # had eval_ids to check against. Legacy runs without
            # `eval_ids` predate the tracking and aren't "drift".
            if dataset_provided and eval_ids_by_slug.get(row.slug):
                if src_report.exists():
                    build_report.unmatched_copied.append(row.slug)
                else:
                    build_report.unmatched_missing.append(row.slug)

            if src_report.exists():
                shutil.copy2(src_report, dest)

    # Docs tab — runs LAST so `pygments.css`, written into
    # `output_dir/assets/` by `render_docs`, survives the asset-copy
    # loop above. `load_docs` is a no-op on a missing or empty
    # directory; `render_docs` short-circuits on an empty section list.
    if docs_dir is not None and docs_dir.exists():
        # Lazy import — keeps `markdown-it-py` / `mdit-py-plugins` /
        # `pygments` off the hot path for `reval leaderboard build`
        # users who don't install the `[docs]` optional extra. The
        # heavy deps themselves are imported lazily *inside* docs.py
        # (see `_render_markdown` / `_highlight_code`), so ImportError
        # surfaces at call time — wrap the calls, not the import.
        from reval.leaderboard.docs import load_docs, render_docs

        try:
            docs_sections = load_docs(docs_dir)
            render_docs(env, docs_sections, output_dir)
        except ImportError as exc:
            logger.warning(
                "Docs tab requested (--docs %s) but a required dependency "
                "is missing (%s). Skipping docs build. Install with: "
                "`pip install reval[docs]`.",
                docs_dir,
                exc,
            )

    return build_report


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
