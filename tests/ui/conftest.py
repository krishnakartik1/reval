"""Fixtures, server, and hooks for the Playwright-based UI tests."""

from __future__ import annotations

import http.server
import inspect
import json
import socketserver
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from reval.contracts.models import BenchmarkRun, EvalCategory
from reval.leaderboard.build import build

UI_TEST_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = UI_TEST_DIR.parent.parent / "artifacts" / "ui"

CATEGORIES = [c.value for c in EvalCategory]


def pytest_collection_modifyitems(config, items):
    """Forbid async tests under tests/ui/ — pytest-playwright's page fixture is sync-only.

    Scoped to tests/ui/ on purpose. `pytest_collection_modifyitems` in any conftest
    receives the full test-session item list (not just items under that conftest's
    directory), so an unscoped check would raise on the async provider tests under
    tests/ and break any default `pytest tests/` run.
    """
    for item in items:
        if not item.path.is_relative_to(UI_TEST_DIR):
            continue
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            raise pytest.UsageError(
                f"UI tests must be sync (pytest-playwright uses a sync page fixture): "
                f"{item.nodeid}"
            )


def _default_fixture_results(latency_base: int) -> list[dict[str, Any]]:
    """Synthetic `results[*]` list used by the default `multi_judge_showcase`.

    Every run in the UI fixture needs populated `results` so the
    charts suite has real latency and rubric data to render. Without
    this, `latency_p50_ms` is None and `aggregated_rubric_scores` is
    empty — the bar charts, criterion heatmap, and radar would have
    no data, violating the "extend `multi_judge_showcase`, don't
    shadow it" rule in `reval/AGENTS.md`.

    Shape each result dict to match `EvalResult.model_dump(mode="json")`
    exactly enough for `reval.leaderboard.build.load_rows()` and its
    helpers to aggregate. Only the fields they read need to be present.
    """
    return [
        {
            "eval_id": "fixture-figure-treatment",
            "model_id": "amazon.nova-pro-v1:0",
            "category": "figure_treatment",
            "raw_response": "fixture response",
            "score": 0.8,
            "scoring_method": "llm_judge",
            "latency_ms": latency_base + 100,
            "rubric_scores": {
                "factual_accuracy": 5,
                "tone_balance": 4,
                "context_fairness": 4,
                "source_attribution": 3,
            },
        },
        {
            "eval_id": "fixture-issue-framing",
            "model_id": "amazon.nova-pro-v1:0",
            "category": "issue_framing",
            "raw_response": "fixture response",
            "score": 0.8,
            "scoring_method": "llm_judge",
            "latency_ms": latency_base + 150,
            "rubric_scores": {
                "perspective_coverage": 4,
                "framing_neutrality": 5,
                "evidence_balance": 4,
                "stakeholder_representation": 5,
            },
        },
        {
            "eval_id": "fixture-policy-attribution",
            "model_id": "amazon.nova-pro-v1:0",
            "category": "policy_attribution",
            "raw_response": "fixture response",
            "score": 0.8,
            "scoring_method": "semantic_similarity",
            "latency_ms": latency_base,
        },
    ]


def _make_run(slug: str, latency_base: int = 500, **overrides: Any) -> dict:
    """Build a results.json-shaped dict by round-tripping through BenchmarkRun.

    Required mixin fields (run_id, timestamp, git_sha, model_provider, model_id)
    and required BenchmarkRun fields (eval_ids) are supplied with defaults;
    any override wins.

    `results` defaults to `_default_fixture_results(latency_base)` so
    the chart tests have latency and rubric data to aggregate. A caller
    that needs the legacy "empty results" shape (e.g. the no-latency
    fallback fixture) passes `results=[]` explicitly.
    """
    defaults: dict[str, Any] = {
        "run_id": slug,
        "timestamp": datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc),
        "git_sha": "testfixture",
        "model_provider": "bedrock",
        "model_id": "amazon.nova-pro-v1:0",
        "eval_ids": ["fixture-eval-1"],
        "overall_score": 0.80,
        "category_scores": {cat: 0.80 for cat in CATEGORIES},
        "total_evals": 1,
        "completed_evals": 1,
        "results": _default_fixture_results(latency_base),
    }
    defaults.update(overrides)
    run = BenchmarkRun(**defaults)
    return run.model_dump(mode="json")


@pytest.fixture(scope="session")
def multi_judge_showcase(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp showcase dir with three runs: Bedrock, OpenRouter-style, plain Anthropic.

    Scores are distinct so default descending sort has a determinable order:
    Run 3 (0.91) > Run 1 (0.82) > Run 2 (0.74).
    """
    showcase = tmp_path_factory.mktemp("showcase")

    # Distinct latency bases so latency_p50_ms has non-trivial spread
    # in leaderboard.json, exercising the sorting and bar charts with
    # real numeric data.
    runs = [
        _make_run(
            slug="bedrock_nova",
            latency_base=200,
            model_provider="bedrock",
            model_id="amazon.nova-pro-v1:0",
            judge_model_id="amazon.nova-lite-v1:0",
            overall_score=0.82,
        ),
        _make_run(
            slug="openrouter_opus",
            latency_base=2000,
            model_provider="openai",
            model_id="gpt-4o-2024-11-20",
            judge_model_id="openrouter/anthropic/claude-3-opus",
            overall_score=0.74,
        ),
        _make_run(
            slug="anthropic_sonnet",
            latency_base=1000,
            model_provider="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            judge_model_id="claude-3-5-sonnet-20241022",
            overall_score=0.91,
        ),
    ]

    for run in runs:
        run_dir = showcase / run["run_id"]
        run_dir.mkdir()
        (run_dir / "results.json").write_text(json.dumps(run, indent=2))

    return showcase


@pytest.fixture(scope="session")
def built_site_dir(
    multi_judge_showcase: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Run the real leaderboard build against the synthetic showcase.

    This fixture ALSO points `docs_dir` at the repo's real `reval/docs/`
    tree, so the Docs tab is built against production content. That
    gives the Playwright suite end-to-end coverage of:

      - the tab bar added to `base.html.j2`
      - the docs index and per-page templates at depth 1 / 2
      - the Alpine sidebar filter hydration
      - the copy-to-clipboard button on fenced code blocks
      - Lucide icon rendering in the docs tab card grid
      - relative path resolution across `index.html` ↔ `docs/...` ↔
        `models/<slug>.html`

    Passing the repo's actual docs tree rather than a tiny synthetic
    fixture keeps the UI suite honest — the content drifts, the tests
    see the drift. On a wheel install (CI rebuild scenarios) the path
    still resolves because `tests/ui/conftest.py` runs from the repo
    checkout, not from the installed package.
    """
    output = tmp_path_factory.mktemp("public")
    # Resolve the repo root by walking up from this conftest:
    # tests/ui/conftest.py → tests/ui → tests → <repo root>
    repo_root = UI_TEST_DIR.parent.parent
    docs_dir = repo_root / "docs"
    rubrics_dir = repo_root / "evals" / "rubrics"
    build(
        showcase_dir=multi_judge_showcase,
        output_dir=output,
        include_reports=False,
        docs_dir=docs_dir if docs_dir.exists() else None,
        rubrics_dir=rubrics_dir if rubrics_dir.exists() else None,
    )
    return output


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def _serve_dir(directory: Path) -> Iterator[str]:
    """Serve a directory on 127.0.0.1 with a random free port.

    Shared helper so the default `site_url` fixture and the
    charts-specific `charts_site_url` fixture don't duplicate the
    threaded-server boilerplate.
    """

    class _Handler(_QuietHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()


@pytest.fixture(scope="session")
def site_url(built_site_dir: Path) -> Iterator[str]:
    """Serve built_site_dir on 127.0.0.1 with a random free port."""
    yield from _serve_dir(built_site_dir)


@pytest.fixture
def page(page, request):  # type: ignore[override]
    """Extend pytest-playwright's page with console/pageerror capture."""
    console_errors: list[str] = []
    page_errors: list[str] = []

    def _on_console(msg: Any) -> None:
        if msg.type == "error":
            console_errors.append(msg.text)

    page.on("console", _on_console)
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    request.node._ui_console_errors = console_errors
    request.node._ui_page_errors = page_errors
    return page


@pytest.fixture
def js_errors(page, request: pytest.FixtureRequest) -> dict[str, list[str]]:
    """Shorthand for tests: assert js_errors['console'] == [] and js_errors['page'] == [].

    Depends on `page` explicitly so that pytest guarantees the overridden `page`
    fixture runs first and populates `request.node._ui_*` — otherwise a test that
    asks for `js_errors` without also asking for `page` would hit AttributeError.
    """
    del page  # only here to enforce fixture ordering
    return {
        "console": request.node._ui_console_errors,
        "page": request.node._ui_page_errors,
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    outcome = yield
    if call.when != "call":
        return
    page = item.funcargs.get("page")
    if page is None:
        return
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    name = item.name.replace("/", "_").replace("[", "_").replace("]", "_")
    try:
        page.screenshot(path=str(ARTIFACTS_DIR / f"{name}.png"), full_page=True)
    except Exception:
        pass
    report = outcome.get_result()
    if report.failed:
        try:
            (ARTIFACTS_DIR / f"{name}.html").write_text(page.content())
        except Exception:
            pass
