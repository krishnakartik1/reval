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

from reval.contracts.models import BenchmarkRun
from reval.leaderboard.build import build

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "ui"

CATEGORIES = [
    "argumentation_parity",
    "factual_accuracy",
    "figure_treatment",
    "issue_framing",
    "policy_attribution",
]


def pytest_collection_modifyitems(config, items):
    """Forbid async tests in tests/ui/ — pytest-playwright's page fixture is sync-only."""
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            raise pytest.UsageError(
                f"UI tests must be sync (pytest-playwright uses a sync page fixture): "
                f"{item.nodeid}"
            )


def _make_run(slug: str, **overrides: Any) -> dict:
    """Build a results.json-shaped dict by round-tripping through BenchmarkRun.

    Required mixin fields (run_id, timestamp, git_sha, model_provider, model_id)
    and required BenchmarkRun fields (eval_ids) are supplied with defaults;
    any override wins.
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

    runs = [
        _make_run(
            slug="bedrock_nova",
            model_provider="bedrock",
            model_id="amazon.nova-pro-v1:0",
            judge_model_id="amazon.nova-lite-v1:0",
            overall_score=0.82,
        ),
        _make_run(
            slug="openrouter_opus",
            model_provider="openai",
            model_id="gpt-4o-2024-11-20",
            judge_model_id="openrouter/anthropic/claude-3-opus",
            overall_score=0.74,
        ),
        _make_run(
            slug="anthropic_sonnet",
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
    """Run the real leaderboard build against the synthetic showcase."""
    output = tmp_path_factory.mktemp("public")
    build(
        showcase_dir=multi_judge_showcase,
        output_dir=output,
        include_reports=False,
    )
    return output


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


@pytest.fixture(scope="session")
def site_url(built_site_dir: Path) -> Iterator[str]:
    """Serve built_site_dir on 127.0.0.1 with a random free port."""

    class _Handler(_QuietHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(built_site_dir), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()


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
def js_errors(request: pytest.FixtureRequest) -> dict[str, list[str]]:
    """Shorthand for tests: assert js_errors['console'] == [] and js_errors['page'] == []."""
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
