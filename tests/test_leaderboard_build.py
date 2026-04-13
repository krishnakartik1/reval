"""Tests for the static leaderboard generator.

No live LLM calls — this is a pure file-in / file-out transformation
from `showcase/*/results.json` to `public/` HTML + JSON + assets. Tests
use a temporary `showcase/` directory with two hand-crafted
results.json fixtures and assert the resulting `public/` tree has the
right shape and content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reval.leaderboard import LeaderboardRow, build, load_rows
from reval.leaderboard.build import (
    _collect_categories,
    _fmt_score,
    _score_color,
)


def _mock_result(
    model_id: str,
    provider: str,
    overall: float,
    categories: dict[str, float],
    *,
    judge: str = "amazon.nova-lite-v1:0",
    embeddings: str = "amazon.titan-embed-text-v2:0",
    git_sha: str = "abc123",
    timestamp: str = "2026-04-12T20:00:00Z",
) -> dict[str, Any]:
    return {
        "run_id": f"run-{model_id}",
        "timestamp": timestamp,
        "git_sha": git_sha,
        "model_provider": provider,
        "model_id": model_id,
        "judge_model_id": judge,
        "embeddings_model_id": embeddings,
        "overall_score": overall,
        "category_scores": categories,
        "total_evals": sum(1 for _ in categories) * 5,
        "completed_evals": sum(1 for _ in categories) * 5,
        "error_count": 0,
        "results": [],  # not needed by the leaderboard
        "completed_at": timestamp,
    }


def _write_showcase_entry(
    showcase_dir: Path, slug: str, data: dict[str, Any], with_report: bool = False
) -> None:
    entry = showcase_dir / slug
    entry.mkdir(parents=True)
    (entry / "results.json").write_text(json.dumps(data))
    if with_report:
        (entry / "report.html").write_text("<html><body>mock report</body></html>")


# ── load_rows ──────────────────────────────────────────────────────────


class TestLoadRows:
    def test_loads_one_row_per_entry(self, tmp_path: Path) -> None:
        showcase = tmp_path / "showcase"
        _write_showcase_entry(
            showcase,
            "claude-haiku_20260412_120000",
            _mock_result(
                "us.anthropic.claude-3-5-haiku",
                "bedrock",
                0.87,
                {"policy_attribution": 0.92, "figure_treatment": 0.81},
            ),
        )
        _write_showcase_entry(
            showcase,
            "gpt-4o_20260412_130000",
            _mock_result("gpt-4o", "openai", 0.83, {"policy_attribution": 0.85}),
        )

        rows = load_rows(showcase)
        assert len(rows) == 2
        assert {row.model_id for row in rows} == {
            "us.anthropic.claude-3-5-haiku",
            "gpt-4o",
        }

    def test_skips_directories_without_results_json(self, tmp_path: Path) -> None:
        showcase = tmp_path / "showcase"
        (showcase / "orphaned").mkdir(parents=True)  # no results.json
        _write_showcase_entry(
            showcase,
            "real_run",
            _mock_result("gpt-4o", "openai", 0.80, {"issue_framing": 0.80}),
        )
        rows = load_rows(showcase)
        assert len(rows) == 1
        assert rows[0].slug == "real_run"

    def test_returns_empty_list_when_showcase_missing(self, tmp_path: Path) -> None:
        rows = load_rows(tmp_path / "does-not-exist")
        assert rows == []

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        showcase = tmp_path / "showcase"
        bad = showcase / "corrupt"
        bad.mkdir(parents=True)
        (bad / "results.json").write_text("{this is not valid json")
        _write_showcase_entry(
            showcase,
            "good",
            _mock_result("gpt-4o", "openai", 0.82, {}),
        )
        rows = load_rows(showcase)
        assert len(rows) == 1
        assert rows[0].slug == "good"

    def test_populates_report_href_when_report_exists(self, tmp_path: Path) -> None:
        showcase = tmp_path / "showcase"
        _write_showcase_entry(
            showcase,
            "has-report",
            _mock_result("gpt-4o", "openai", 0.82, {}),
            with_report=True,
        )
        _write_showcase_entry(
            showcase,
            "no-report",
            _mock_result("gpt-4o-mini", "openai", 0.75, {}),
            with_report=False,
        )
        rows = {row.slug: row for row in load_rows(showcase)}
        assert rows["has-report"].report_href == "reports/has-report.html"
        assert rows["no-report"].report_href is None

    def test_tolerates_missing_optional_fields(self, tmp_path: Path) -> None:
        """Runs pre-dating certain fields should still render."""
        showcase = tmp_path / "showcase"
        entry = showcase / "minimal"
        entry.mkdir(parents=True)
        (entry / "results.json").write_text(
            json.dumps(
                {
                    "model_id": "gpt-4o",
                    "model_provider": "openai",
                    # no overall_score, no category_scores, no timestamps
                }
            )
        )
        rows = load_rows(showcase)
        assert len(rows) == 1
        row = rows[0]
        assert row.overall_score is None
        assert row.category_scores == {}
        assert row.timestamp is None


# ── _collect_categories ────────────────────────────────────────────────


class TestCollectCategories:
    def test_union_across_rows(self) -> None:
        rows = [
            LeaderboardRow(
                slug="a",
                model_id="m",
                model_provider="p",
                category_scores={"policy_attribution": 0.9, "figure_treatment": 0.8},
            ),
            LeaderboardRow(
                slug="b",
                model_id="m",
                model_provider="p",
                category_scores={"figure_treatment": 0.7, "issue_framing": 0.8},
            ),
        ]
        assert _collect_categories(rows) == [
            "figure_treatment",
            "issue_framing",
            "policy_attribution",
        ]

    def test_empty_rows(self) -> None:
        assert _collect_categories([]) == []


# ── Template filters ───────────────────────────────────────────────────


class TestScoreColor:
    def test_high(self) -> None:
        assert _score_color(0.90) == "score-high"
        assert _score_color(0.85) == "score-high"

    def test_mid(self) -> None:
        assert _score_color(0.75) == "score-mid"
        assert _score_color(0.70) == "score-mid"

    def test_low(self) -> None:
        assert _score_color(0.50) == "score-low"

    def test_none(self) -> None:
        assert _score_color(None) == "score-none"


class TestFmtScore:
    def test_float(self) -> None:
        assert _fmt_score(0.8612) == "0.861"

    def test_none(self) -> None:
        assert _fmt_score(None) == "—"


# ── build() end-to-end ─────────────────────────────────────────────────


class TestBuild:
    @pytest.fixture
    def showcase(self, tmp_path: Path) -> Path:
        showcase_dir = tmp_path / "showcase"
        _write_showcase_entry(
            showcase_dir,
            "claude-haiku_20260412_120000",
            _mock_result(
                "us.anthropic.claude-3-5-haiku",
                "bedrock",
                0.87,
                {
                    "policy_attribution": 0.92,
                    "figure_treatment": 0.81,
                    "issue_framing": 0.88,
                },
            ),
            with_report=True,
        )
        _write_showcase_entry(
            showcase_dir,
            "gpt-4o_20260412_130000",
            _mock_result(
                "gpt-4o",
                "openai",
                0.83,
                {"policy_attribution": 0.85, "figure_treatment": 0.80},
            ),
        )
        return showcase_dir

    def test_writes_index_html(self, showcase: Path, tmp_path: Path) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        index = output / "index.html"
        assert index.exists()
        content = index.read_text()
        assert "gpt-4o" in content
        assert "us.anthropic.claude-3-5-haiku" in content
        assert "REVAL" in content

    def test_writes_leaderboard_json(self, showcase: Path, tmp_path: Path) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        data_path = output / "data" / "leaderboard.json"
        assert data_path.exists()
        payload = json.loads(data_path.read_text())
        assert "rows" in payload and "categories" in payload
        assert len(payload["rows"]) == 2
        assert "policy_attribution" in payload["categories"]
        # Rows carry full LeaderboardRow shape
        first = payload["rows"][0]
        assert "model_id" in first
        assert "overall_score" in first
        assert "category_scores" in first

    def test_writes_per_model_pages(self, showcase: Path, tmp_path: Path) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        model_files = list((output / "models").glob("*.html"))
        assert len(model_files) == 2
        slugs = {f.stem for f in model_files}
        assert slugs == {
            "claude-haiku_20260412_120000",
            "gpt-4o_20260412_130000",
        }

        # Per-model page contains the model's category scores
        claude_page = (
            output / "models" / "claude-haiku_20260412_120000.html"
        ).read_text()
        assert "us.anthropic.claude-3-5-haiku" in claude_page
        assert "0.920" in claude_page  # policy_attribution formatted score

    def test_copies_assets(self, showcase: Path, tmp_path: Path) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        assert (output / "assets" / "style.css").exists()
        assert (output / "assets" / "tokens.css").exists()
        assert (output / "assets" / "radar.js").exists()

    def test_copies_report_html_when_requested(
        self, showcase: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output, include_reports=True)

        reports_dir = output / "reports"
        assert reports_dir.exists()
        # Only the claude entry had a report.html in the fixture
        copied = list(reports_dir.glob("*.html"))
        assert len(copied) == 1
        assert copied[0].name == "claude-haiku_20260412_120000.html"

    def test_skips_report_copy_when_disabled(
        self, showcase: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output, include_reports=False)

        # reports/ dir exists but is empty
        reports_dir = output / "reports"
        if reports_dir.exists():
            assert not any(reports_dir.iterdir())

    def test_empty_showcase_still_writes_valid_index(self, tmp_path: Path) -> None:
        """No runs in showcase → still emit an empty index page."""
        showcase = tmp_path / "empty"
        showcase.mkdir()
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        index = output / "index.html"
        assert index.exists()
        content = index.read_text()
        assert "No runs" in content or "showcase" in content
        assert (output / "data" / "leaderboard.json").exists()

    def test_category_union_from_mixed_runs(
        self, showcase: Path, tmp_path: Path
    ) -> None:
        """Claude run has 3 categories, gpt-4o has 2. Union should be 3."""
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)

        data = json.loads((output / "data" / "leaderboard.json").read_text())
        assert set(data["categories"]) == {
            "policy_attribution",
            "figure_treatment",
            "issue_framing",
        }
        # gpt-4o row in leaderboard.json doesn't have issue_framing
        gpt_row = next(r for r in data["rows"] if r["model_id"] == "gpt-4o")
        assert "issue_framing" not in gpt_row["category_scores"]

    def test_output_is_idempotent(self, showcase: Path, tmp_path: Path) -> None:
        """Running build twice should produce the same output (regenerable)."""
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output)
        first_index = (output / "index.html").read_text()
        first_json = (output / "data" / "leaderboard.json").read_text()

        build(showcase_dir=showcase, output_dir=output)
        assert (output / "index.html").read_text() == first_index
        assert (output / "data" / "leaderboard.json").read_text() == first_json

    def test_regenerates_reports_against_dataset(self, tmp_path: Path) -> None:
        """With `dataset_dir=`, reports are re-rendered with Test case sections.

        Regression guard for the 'embedded report shows no prompts' bug:
        a run whose `showcase/<slug>/report.html` predates PR #18 (no
        3-section expansion) should still produce a fresh report on the
        public/ side when `reval leaderboard build` is given a dataset.
        """
        # Build a tiny showcase with a single issue_framing run
        showcase_dir = tmp_path / "showcase"
        entry_slug = "gpt-4o_20260412_120000"
        _write_showcase_entry(
            showcase_dir,
            entry_slug,
            {
                **_mock_result("gpt-4o", "openai", 0.88, {"issue_framing": 0.88}),
                "eval_ids": ["us-issue_framing-001"],
                "results": [
                    {
                        "eval_id": "us-issue_framing-001",
                        "model_id": "gpt-4o",
                        "category": "issue_framing",
                        "raw_response": "Healthcare is a complex policy area...",
                        "score": 0.88,
                        "scoring_method": "llm_judge",
                    }
                ],
            },
            with_report=False,
        )

        # Build a tiny dataset with ONE matching entry
        dataset_dir = tmp_path / "datasets" / "us"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "issue_framing.jsonl").write_text(
            json.dumps(
                {
                    "id": "us-issue_framing-001",
                    "category": "issue_framing",
                    "country": "us",
                    "topic": "healthcare",
                    "prompt": "Explain the debate around universal healthcare.",
                    "rubric_id": "issue_framing",
                }
            )
            + "\n"
        )

        output = tmp_path / "public"
        build(
            showcase_dir=showcase_dir,
            output_dir=output,
            include_reports=True,
            dataset_dir=tmp_path / "datasets",
        )

        report_path = output / "reports" / f"{entry_slug}.html"
        assert report_path.exists(), (
            "build() should regenerate public/reports/<slug>.html when "
            "dataset_dir is provided"
        )
        content = report_path.read_text()

        # The Test case section should be present, with the actual prompt
        assert (
            ">Test case<" in content
        ), "Regenerated report should include the Test case section header"
        assert (
            "Explain the debate around universal healthcare" in content
        ), "Regenerated report should embed the actual prompt from the dataset"

    def test_reports_fall_back_to_copy_without_dataset(
        self, showcase: Path, tmp_path: Path
    ) -> None:
        """Without `dataset_dir`, build() copies showcase/<slug>/report.html.

        Preserves the pre-fix behavior for callers that don't pass
        a dataset path. The Test case section won't be regenerated
        but the file will still exist.
        """
        output = tmp_path / "public"
        build(showcase_dir=showcase, output_dir=output, include_reports=True)
        # claude-haiku showcase entry had with_report=True in the fixture
        assert (output / "reports" / "claude-haiku_20260412_120000.html").exists()

    def test_build_report_flags_partial_dataset_match(self, tmp_path: Path) -> None:
        """A run with 3 eval_ids where only 2 exist in the dataset
        should be tracked as a partial match in the returned BuildReport.
        """
        showcase_dir = tmp_path / "showcase"
        entry_slug = "gpt-4o_20260412_120000"
        _write_showcase_entry(
            showcase_dir,
            entry_slug,
            {
                **_mock_result("gpt-4o", "openai", 0.88, {"issue_framing": 0.88}),
                "eval_ids": [
                    "us-issue_framing-001",
                    "us-issue_framing-002",
                    "us-issue_framing-999",  # not in dataset
                ],
                "results": [
                    {
                        "eval_id": "us-issue_framing-001",
                        "model_id": "gpt-4o",
                        "category": "issue_framing",
                        "raw_response": "Response A",
                        "score": 0.9,
                        "scoring_method": "llm_judge",
                    },
                ],
            },
        )

        dataset_dir = tmp_path / "datasets" / "us"
        dataset_dir.mkdir(parents=True)
        lines = [
            json.dumps(
                {
                    "id": "us-issue_framing-001",
                    "category": "issue_framing",
                    "country": "us",
                    "topic": "healthcare",
                    "prompt": "Prompt 1",
                    "rubric_id": "issue_framing",
                }
            ),
            json.dumps(
                {
                    "id": "us-issue_framing-002",
                    "category": "issue_framing",
                    "country": "us",
                    "topic": "education",
                    "prompt": "Prompt 2",
                    "rubric_id": "issue_framing",
                }
            ),
        ]
        (dataset_dir / "issue_framing.jsonl").write_text("\n".join(lines) + "\n")

        output = tmp_path / "public"
        report = build(
            showcase_dir=showcase_dir,
            output_dir=output,
            include_reports=True,
            dataset_dir=tmp_path / "datasets",
        )

        assert report.partial_matches == [(entry_slug, 2, 3)]
        assert report.unmatched_copied == []
        assert report.unmatched_missing == []

    def test_build_report_flags_unmatched_copied(self, tmp_path: Path) -> None:
        """A run with eval_ids that none exist in the dataset should be
        tracked as unmatched_copied when a showcase report.html exists.
        """
        showcase_dir = tmp_path / "showcase"
        entry_slug = "orphan_20260412_120000"
        _write_showcase_entry(
            showcase_dir,
            entry_slug,
            {
                **_mock_result("gpt-4o", "openai", 0.88, {"issue_framing": 0.88}),
                "eval_ids": ["us-issue_framing-999"],
                "results": [],
            },
            with_report=True,
        )

        dataset_dir = tmp_path / "datasets" / "us"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "issue_framing.jsonl").write_text(
            json.dumps(
                {
                    "id": "us-issue_framing-001",
                    "category": "issue_framing",
                    "country": "us",
                    "topic": "healthcare",
                    "prompt": "Prompt 1",
                    "rubric_id": "issue_framing",
                }
            )
            + "\n"
        )

        output = tmp_path / "public"
        report = build(
            showcase_dir=showcase_dir,
            output_dir=output,
            include_reports=True,
            dataset_dir=tmp_path / "datasets",
        )

        assert report.partial_matches == []
        assert report.unmatched_copied == [entry_slug]
        assert report.unmatched_missing == []
        # Old showcase report still copied as the fallback
        assert (output / "reports" / f"{entry_slug}.html").exists()
