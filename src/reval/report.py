"""Report generators and output helpers for REVAL benchmark results.

The HTML report uses the same CSS palette as the static leaderboard
(`reval.leaderboard.get_style_css`), so individual run reports and the
public leaderboard look consistent — clicking a row on the leaderboard
and landing on that run's detail page is visually seamless.

Each result card expands into three stacked sections:

1. **Test case**    — per-category rendering of the prompt(s),
                      entities / figures, ground truth, citations, etc.
                      Requires the original `EvalEntry` (pass `evals=`
                      to `generate_html_report` or `save_run_outputs`).
2. **Response**     — full, scrollable model output. Paired-prompt
                      categories (policy_attribution, figure_treatment,
                      argumentation_parity) show both responses side by
                      side. Truncation-free — callers can eyeball
                      exactly what the model produced.
3. **Score**        — top-line score, rubric bars, sub-scores, judge
                      reasoning, and a mini horizontal bar chart when
                      rubric scores are present.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from reval.contracts import BenchmarkRun, EvalCategory, EvalEntry, EvalResult
from reval.leaderboard import get_style_css

# ── Score band helpers ─────────────────────────────────────────────────


def _score_color(score: float | None) -> str:
    """Return a color class (score-high / mid / low / none)."""
    if score is None:
        return "score-none"
    if score >= 0.85:
        return "score-high"
    if score >= 0.70:
        return "score-mid"
    return "score-low"


def _score_bg(score: float) -> str:
    if score >= 0.85:
        return "#dcfce7"
    if score >= 0.70:
        return "#fef9c3"
    return "#fee2e2"


def _score_hex(score: float) -> str:
    if score >= 0.85:
        return "#22c55e"
    if score >= 0.70:
        return "#eab308"
    return "#ef4444"


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Low bias"
    if score >= 0.70:
        return "Moderate"
    return "Potential bias"


def _esc(text: object) -> str:
    return html.escape(str(text))


def _cat(value: object) -> str:
    """Unwrap an EvalCategory enum (or leave a string alone)."""
    return value.value if hasattr(value, "value") else str(value)


# ── Per-category test-case rendering ───────────────────────────────────


def _render_kv(label: str, value: object) -> str:
    """Render a single `label: value` row inside a test-case panel."""
    return (
        f'<div class="kv"><span class="kv-label">{_esc(label)}</span>'
        f'<span class="kv-value">{_esc(value)}</span></div>'
    )


def _render_prompt_block(label: str, prompt: str) -> str:
    return (
        f'<div class="prompt-block"><div class="prompt-label">{_esc(label)}</div>'
        f'<div class="prompt-body">{_esc(prompt)}</div></div>'
    )


def _render_test_case(entry: EvalEntry) -> str:
    """Render the 'Test case' section for a single EvalEntry.

    Per-category dispatch — each category pulls out the fields that
    matter for that category. Missing fields render nothing (not a
    dash) to keep the panel tight.
    """
    parts: list[str] = []

    meta_row = [
        ("ID", entry.id),
        ("Category", _cat(entry.category)),
        ("Country", _cat(entry.country)),
        ("Topic", entry.topic),
    ]
    if entry.subtopic:
        meta_row.append(("Subtopic", entry.subtopic))
    parts.append(
        '<div class="kv-grid">'
        + "".join(_render_kv(k, v) for k, v in meta_row)
        + "</div>"
    )

    cat = entry.category

    if cat == EvalCategory.POLICY_ATTRIBUTION and entry.counterfactual_pair:
        pair = entry.counterfactual_pair
        parts.append(
            '<div class="prompt-pair">'
            + _render_prompt_block(f"Prompt A · {pair.entity_a}", pair.prompt_a)
            + _render_prompt_block(f"Prompt B · {pair.entity_b}", pair.prompt_b)
            + "</div>"
            f'<div class="kv">{_render_kv("Policy topic", pair.policy_topic)}</div>'
        )

    elif cat == EvalCategory.FIGURE_TREATMENT and entry.figure_pair:
        pair = entry.figure_pair
        parts.append(
            '<div class="prompt-pair">'
            + _render_prompt_block(
                f"Prompt A · {pair.figure_a} ({pair.affiliation_a})", pair.prompt_a
            )
            + _render_prompt_block(
                f"Prompt B · {pair.figure_b} ({pair.affiliation_b})", pair.prompt_b
            )
            + "</div>"
        )
        if entry.rubric_id:
            parts.append(
                f'<div class="kv">{_render_kv("Rubric", entry.rubric_id)}</div>'
            )

    elif cat == EvalCategory.FACTUAL_ACCURACY:
        if entry.prompt:
            parts.append(_render_prompt_block("Prompt", entry.prompt))
        if entry.counterfactual_prompt:
            parts.append(
                _render_prompt_block(
                    "Counterfactual prompt", entry.counterfactual_prompt
                )
            )
        if entry.ground_truth:
            gt = entry.ground_truth
            level_label = (
                f"Level {gt.level.value} · {gt.level.name.replace('_', ' ').title()}"
            )
            parts.append(
                '<div class="ground-truth">'
                f'<div class="gt-header">Ground truth <span class="gt-level">{_esc(level_label)}</span></div>'
                f'<div class="gt-claim"><strong>Claim:</strong> {_esc(gt.claim)}</div>'
                f'<div class="gt-correct"><strong>Correct response:</strong> {_esc(gt.correct_response)}</div>'
            )
            if gt.citations:
                parts.append(
                    '<div class="gt-citations"><strong>Citations:</strong><ul>'
                )
                for cite in gt.citations:
                    url_html = (
                        f' · <a href="{_esc(cite.url)}" target="_blank" rel="noopener">source</a>'
                        if cite.url
                        else ""
                    )
                    parts.append(f"<li>{_esc(cite.source)}{url_html}</li>")
                parts.append("</ul></div>")
            if gt.common_misconceptions:
                parts.append(
                    '<div class="gt-misconceptions"><strong>Common misconceptions:</strong><ul>'
                    + "".join(f"<li>{_esc(m)}</li>" for m in gt.common_misconceptions)
                    + "</ul></div>"
                )
            parts.append("</div>")

    elif cat == EvalCategory.ISSUE_FRAMING:
        if entry.prompt:
            parts.append(_render_prompt_block("Prompt", entry.prompt))
        if entry.rubric_id:
            parts.append(
                f'<div class="kv">{_render_kv("Rubric", entry.rubric_id)}</div>'
            )

    elif cat == EvalCategory.ARGUMENTATION_PARITY:
        if entry.position_a:
            parts.append(_render_prompt_block("Position A", entry.position_a))
        if entry.position_b:
            parts.append(_render_prompt_block("Position B", entry.position_b))

    if entry.tags:
        parts.append(
            '<div class="tags">'
            + "".join(f'<span class="tag">{_esc(t)}</span>' for t in entry.tags)
            + "</div>"
        )

    return '<div class="section-body">' + "".join(parts) + "</div>"


# ── Response rendering ────────────────────────────────────────────────


def _render_response_block(label: str, body: str) -> str:
    """Scrollable response block — no truncation."""
    return (
        f'<div class="response-block">'
        f'<div class="response-label">{_esc(label)}</div>'
        f'<pre class="response-body">{_esc(body)}</pre>'
        f"</div>"
    )


def _render_responses(result: EvalResult) -> str:
    """Render the 'Response' section for a single EvalResult.

    Paired-prompt categories (policy_attribution, figure_treatment,
    argumentation_parity) show two blocks; factual_accuracy shows
    response + counterfactual_response; issue_framing shows just
    raw_response.
    """
    cat = result.category
    blocks: list[str] = []

    if cat in (
        EvalCategory.POLICY_ATTRIBUTION,
        EvalCategory.FIGURE_TREATMENT,
        EvalCategory.ARGUMENTATION_PARITY,
    ):
        if result.response_a:
            blocks.append(_render_response_block("Response A", result.response_a))
        if result.response_b:
            blocks.append(_render_response_block("Response B", result.response_b))
        if not blocks and result.raw_response:
            blocks.append(_render_response_block("Response", result.raw_response))
    elif cat == EvalCategory.FACTUAL_ACCURACY:
        if result.raw_response:
            blocks.append(_render_response_block("Response", result.raw_response))
        if result.counterfactual_response:
            blocks.append(
                _render_response_block(
                    "Counterfactual response", result.counterfactual_response
                )
            )
    else:
        if result.raw_response:
            blocks.append(_render_response_block("Response", result.raw_response))

    if not blocks:
        return '<p class="empty-section">No response captured.</p>'

    return '<div class="section-body">' + "".join(blocks) + "</div>"


# ── Score + graph rendering ───────────────────────────────────────────


def _render_rubric_bars(rubric_scores: dict[str, float]) -> str:
    """Horizontal bar chart for rubric scores (1–5 or 0–1 normalised)."""
    rows: list[str] = []
    for name, value in rubric_scores.items():
        if value > 1:
            pct = max(0.0, min(100.0, ((value - 1) / 4) * 100))
            display = f"{value:.0f}/5"
        else:
            pct = max(0.0, min(100.0, value * 100))
            display = f"{value:.2f}"
        rows.append(
            '<div class="rubric-row">'
            f'<span class="rubric-name">{_esc(name)}</span>'
            '<div class="rubric-bar-bg">'
            f'<div class="rubric-bar" style="width:{pct:.0f}%"></div>'
            "</div>"
            f'<span class="rubric-val">{_esc(display)}</span>'
            "</div>"
        )
    return '<div class="rubric-bars">' + "".join(rows) + "</div>"


def _render_sub_scores(result: EvalResult) -> str:
    """Render all optional numeric sub-scores as a compact list."""
    items: list[tuple[str, float]] = []
    if result.similarity_score is not None:
        items.append(("Similarity", result.similarity_score))
    if result.counterfactual_similarity is not None:
        items.append(("Counterfactual similarity", result.counterfactual_similarity))
    if result.framing_consistency is not None:
        items.append(("Framing consistency", result.framing_consistency))
    if result.score_a is not None:
        items.append(("Score A", result.score_a))
    if result.score_b is not None:
        items.append(("Score B", result.score_b))
    if result.treatment_parity is not None:
        items.append(("Treatment parity", result.treatment_parity))
    if result.raw_score is not None and result.raw_score != result.score:
        items.append(("Raw score", result.raw_score))

    if not items:
        return ""

    rows = "".join(
        f'<div class="sub-score"><span class="sub-score-label">{_esc(label)}</span>'
        f'<span class="sub-score-value score-cell {_score_color(value)}">{value:.3f}</span></div>'
        for label, value in items
    )
    return f'<div class="sub-scores">{rows}</div>'


def _render_score_section(result: EvalResult) -> str:
    """Render the 'Score & breakdown' section for a single EvalResult."""
    method = _cat(result.scoring_method) if result.scoring_method else "—"
    latency = f"{result.latency_ms} ms" if result.latency_ms is not None else "—"

    parts: list[str] = [
        f'<div class="score-hero score-cell {_score_color(result.score)}">'
        f'<div class="score-hero-label">Overall</div>'
        f'<div class="score-hero-value">{result.score:.3f}</div>'
        f'<div class="score-hero-sub">{_esc(_score_label(result.score))}</div>'
        f"</div>"
        f'<div class="kv-grid tight">'
        f'{_render_kv("Method", method)}'
        f'{_render_kv("Latency", latency)}'
        f"</div>"
    ]

    sub_scores = _render_sub_scores(result)
    if sub_scores:
        parts.append('<h4 class="section-h4">Sub-scores</h4>')
        parts.append(sub_scores)

    if result.rubric_scores:
        parts.append('<h4 class="section-h4">Rubric breakdown</h4>')
        parts.append(_render_rubric_bars(result.rubric_scores))

    if result.judge_reasoning:
        parts.append('<h4 class="section-h4">Judge reasoning</h4>')
        parts.append(
            f'<div class="judge-reasoning">{_esc(result.judge_reasoning)}</div>'
        )

    return '<div class="section-body">' + "".join(parts) + "</div>"


# ── Additional CSS on top of the leaderboard palette ──────────────────


_REPORT_EXTRA_CSS = """
/* Run report — inherits the palette from leaderboard/assets/style.css,
   adds the overall-card hero, category chart container, and the
   three-section expandable result cards. */

:root { --mono: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace; }

.run-hero {
  padding: 2rem;
  border-radius: 0.75rem;
  margin: 1rem 0 2rem;
  border: 1px solid var(--border);
}
.run-hero.score-high { background: var(--score-high-bg); color: var(--score-high-fg); }
.run-hero.score-mid  { background: var(--score-mid-bg);  color: var(--score-mid-fg); }
.run-hero.score-low  { background: var(--score-low-bg);  color: var(--score-low-fg); }
.run-hero-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8; }
.run-hero-value { font-size: 3rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; margin: 0.25rem 0; }
.run-hero-sub { font-size: 1rem; opacity: 0.85; }
.run-hero-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid rgba(0,0,0,0.1);
}
@media (prefers-color-scheme: dark) {
  .run-hero-meta { border-top-color: rgba(255,255,255,0.1); }
}
.run-hero-meta .kv-label { opacity: 0.75; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
.run-hero-meta .kv-value { font-size: 0.9rem; font-family: var(--mono); word-break: break-word; }

h2.section-title {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 2.5rem 0 1rem;
  color: var(--fg);
}

/* Category chart */
.chart-card {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1.5rem;
  margin-bottom: 2rem;
}
.chart-container { position: relative; height: 280px; }

/* Result cards (expandable) */
.result-list { display: flex; flex-direction: column; gap: 0.5rem; }
.result-card {
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  background: var(--bg);
  overflow: hidden;
  transition: border-color 0.15s;
}
.result-card:hover { border-color: var(--accent); }
.result-card.expanded { border-color: var(--accent); }
.result-header {
  display: grid;
  grid-template-columns: auto 1fr auto auto auto;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  cursor: pointer;
  user-select: none;
}
.result-header:hover { background: var(--bg-alt); }
.expand-icon {
  display: inline-block;
  width: 1rem;
  color: var(--fg-dim);
  transition: transform 0.15s;
  font-size: 0.7rem;
}
.result-card.expanded .expand-icon { transform: rotate(90deg); }
.result-eval-id { font-family: var(--mono); font-size: 0.85rem; color: var(--fg); }
.result-category { font-size: 0.8rem; color: var(--fg-dim); }
.result-method { font-size: 0.75rem; color: var(--fg-dim); font-family: var(--mono); }
.result-score {
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.result-score.score-high { background: var(--score-high-bg); color: var(--score-high-fg); }
.result-score.score-mid  { background: var(--score-mid-bg);  color: var(--score-mid-fg); }
.result-score.score-low  { background: var(--score-low-bg);  color: var(--score-low-fg); }

.result-body {
  display: none;
  padding: 0 1rem 1rem;
  border-top: 1px solid var(--border);
}
.result-card.expanded .result-body { display: block; }

.section {
  margin-top: 1rem;
  border-radius: 0.375rem;
  background: var(--bg-alt);
  overflow: hidden;
}
.section-header {
  padding: 0.6rem 1rem;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--fg-dim);
  font-weight: 600;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
}
.section-body { padding: 1rem; }

/* Test case section */
.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.5rem 1rem;
  margin-bottom: 1rem;
}
.kv-grid.tight { margin-bottom: 0; }
.kv { display: flex; flex-direction: column; font-size: 0.85rem; }
.kv-label { color: var(--fg-dim); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; }
.kv-value { font-family: var(--mono); word-break: break-word; }

.prompt-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
@media (max-width: 800px) { .prompt-pair { grid-template-columns: 1fr; } }
.prompt-block {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.75rem 1rem;
}
.prompt-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--fg-dim);
  margin-bottom: 0.5rem;
  font-weight: 600;
}
.prompt-body { font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; }

.ground-truth {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 1rem;
  margin-top: 1rem;
  font-size: 0.9rem;
}
.gt-header { display: flex; align-items: center; gap: 0.5rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--fg); }
.gt-level { font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 0.375rem; background: var(--bg-alt); color: var(--fg-dim); font-weight: 500; }
.gt-claim, .gt-correct { margin: 0.5rem 0; line-height: 1.5; }
.gt-citations, .gt-misconceptions { margin-top: 0.75rem; font-size: 0.85rem; }
.gt-citations ul, .gt-misconceptions ul { margin: 0.25rem 0 0 1.25rem; padding: 0; }
.gt-citations li, .gt-misconceptions li { margin: 0.15rem 0; }
.gt-citations a { color: var(--accent); }

.tags { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.tag {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 0.1rem 0.6rem;
  font-size: 0.7rem;
  color: var(--fg-dim);
  font-family: var(--mono);
}

/* Response section */
.response-block {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  overflow: hidden;
  margin-bottom: 1rem;
}
.response-block:last-child { margin-bottom: 0; }
.response-label {
  padding: 0.5rem 1rem;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--fg-dim);
  font-weight: 600;
  background: var(--bg-alt);
  border-bottom: 1px solid var(--border);
}
.response-body {
  max-height: 400px;
  overflow-y: auto;
  margin: 0;
  padding: 1rem;
  font-family: var(--mono);
  font-size: 0.85rem;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--fg);
}

/* Score section */
.score-hero {
  padding: 1.25rem 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
}
.score-hero-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.8;
}
.score-hero-value {
  font-size: 2.5rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  margin: 0.25rem 0;
}
.score-hero-sub { font-size: 0.9rem; opacity: 0.85; }

.section-h4 {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--fg-dim);
  font-weight: 600;
  margin: 1.25rem 0 0.5rem;
}

.rubric-bars { display: flex; flex-direction: column; gap: 0.4rem; }
.rubric-row { display: flex; align-items: center; gap: 0.75rem; }
.rubric-name { width: 200px; font-size: 0.85rem; flex-shrink: 0; color: var(--fg); }
.rubric-bar-bg {
  flex: 1;
  height: 10px;
  background: var(--border);
  border-radius: 5px;
  overflow: hidden;
}
.rubric-bar {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  border-radius: 5px;
  transition: width 0.3s;
}
.rubric-val {
  width: 50px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 0.85rem;
  font-family: var(--mono);
  color: var(--fg);
}

.sub-scores {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.5rem;
}
.sub-score {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
}
.sub-score-label { font-size: 0.85rem; color: var(--fg); }
.sub-score-value { padding: 0.1rem 0.5rem; border-radius: 0.25rem; font-variant-numeric: tabular-nums; font-size: 0.85rem; font-weight: 600; }

.judge-reasoning {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.85rem 1rem;
  font-size: 0.9rem;
  line-height: 1.55;
  white-space: pre-wrap;
  color: var(--fg);
}

.empty-section { color: var(--fg-dim); font-style: italic; padding: 1rem; }
"""


# ── HTML report builder ────────────────────────────────────────────────


def generate_html_report(
    run: BenchmarkRun,
    output_path: str | Path,
    evals: list[EvalEntry] | None = None,
) -> None:
    """Generate a self-contained HTML dashboard report.

    Args:
        run: Completed benchmark run with results.
        output_path: Path to write the HTML file.
        evals: Optional list of `EvalEntry` objects corresponding to
            the runs's `eval_ids`. When provided, each result expands
            into a three-section panel: test case (from the entry),
            response (from the result), and score breakdown. When None,
            the test-case section is omitted and the layout degrades
            gracefully.
    """
    overall = run.overall_score or 0.0
    started = (
        run.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if run.timestamp else "N/A"
    )

    # Index evals by id for per-result lookup
    eval_by_id: dict[str, EvalEntry] = {}
    if evals:
        for entry in evals:
            eval_by_id[entry.id] = entry

    # Sort results by category then eval_id
    sorted_results = sorted(
        run.results,
        key=lambda r: (_cat(r.category), r.eval_id),
    )

    # Category chart data
    cat_labels = json.dumps(list(run.category_scores.keys()))
    cat_values = json.dumps(list(run.category_scores.values()))
    cat_colors = json.dumps([_score_hex(v) for v in run.category_scores.values()])

    # Result cards
    card_chunks: list[str] = []
    for result in sorted_results:
        entry = eval_by_id.get(result.eval_id)
        category_label = _cat(result.category)
        method_label = _cat(result.scoring_method) if result.scoring_method else "—"

        sections: list[str] = []
        if entry is not None:
            sections.append(
                '<div class="section">'
                '<div class="section-header">Test case</div>'
                f"{_render_test_case(entry)}"
                "</div>"
            )
        sections.append(
            '<div class="section">'
            '<div class="section-header">Response</div>'
            f"{_render_responses(result)}"
            "</div>"
        )
        sections.append(
            '<div class="section">'
            '<div class="section-header">Score &amp; breakdown</div>'
            f"{_render_score_section(result)}"
            "</div>"
        )

        card_chunks.append(
            '<article class="result-card">'
            '<header class="result-header">'
            '<span class="expand-icon">&#9654;</span>'
            f'<span class="result-eval-id">{_esc(result.eval_id)}</span>'
            f'<span class="result-category">{_esc(category_label)}</span>'
            f'<span class="result-method">{_esc(method_label)}</span>'
            f'<span class="result-score {_score_color(result.score)}">{result.score:.3f}</span>'
            "</header>"
            f'<div class="result-body">{"".join(sections)}</div>'
            "</article>"
        )

    results_html = "\n".join(card_chunks)

    # Meta items for the hero
    meta_items = [
        ("Target model", run.model_id or "—"),
        ("Judge model", run.judge_model_id or "—"),
        ("Embeddings model", run.embeddings_model_id or "—"),
        (
            "Completed",
            f"{run.completed_evals}/{run.total_evals}"
            + (f" · {run.error_count} errors" if run.error_count else ""),
        ),
        ("Run date", started),
        ("Run ID", (run.run_id[:12] + "…") if run.run_id else "—"),
    ]
    meta_html = "".join(
        f'<div class="kv"><span class="kv-label">{_esc(k)}</span>'
        f'<span class="kv-value">{_esc(v)}</span></div>'
        for k, v in meta_items
    )

    style = get_style_css() + _REPORT_EXTRA_CSS

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>REVAL Report · {_esc(run.model_id)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
{style}
</style>
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <h1>REVAL <span class="subtitle">Benchmark report</span></h1>
    <p class="tagline">Per-eval breakdown · click any row to expand</p>
  </div>
</header>
<main class="wrap">

  <section class="run-hero {_score_color(overall)}">
    <div class="run-hero-label">Overall score</div>
    <div class="run-hero-value">{overall:.3f}</div>
    <div class="run-hero-sub">{_esc(_score_label(overall))}</div>
    <div class="run-hero-meta">{meta_html}</div>
  </section>

  <h2 class="section-title">Category scores</h2>
  <div class="chart-card">
    <div class="chart-container">
      <canvas id="categoryChart"></canvas>
    </div>
  </div>

  <h2 class="section-title">Individual results</h2>
  <div class="result-list">
    {results_html}
  </div>

</main>
<footer class="site-footer">
  <div class="wrap">
    <p>Generated by <code>reval run</code> · <a href="https://github.com/krishnakartik1/reval">reval on GitHub</a></p>
  </div>
</footer>

<script>
  new Chart(document.getElementById('categoryChart'), {{
    type: 'bar',
    data: {{
      labels: {cat_labels},
      datasets: [{{
        data: {cat_values},
        backgroundColor: {cat_colors},
        borderRadius: 6,
        barThickness: 30,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.x.toFixed(3) }} }}
      }},
      scales: {{
        x: {{ min: 0, max: 1, ticks: {{ stepSize: 0.2 }}, grid: {{ color: 'rgba(148,163,184,0.2)' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: 'rgba(148,163,184,0.9)' }} }}
      }}
    }}
  }});

  // Expand/collapse result cards. Click a header toggles the one card;
  // other cards stay as-is (no accordion — user can compare side by
  // side).
  document.querySelectorAll('.result-header').forEach(h => {{
    h.addEventListener('click', () => {{
      h.parentElement.classList.toggle('expanded');
    }});
  }});
</script>
</body>
</html>"""

    Path(output_path).write_text(page, encoding="utf-8")


def save_run_outputs(
    run: BenchmarkRun,
    output_dir: str | Path,
    run_name: str | None = None,
    evals: list[EvalEntry] | None = None,
) -> Path:
    """Save all run outputs (JSON, HTML, Markdown) into a subdirectory.

    Args:
        run: Completed benchmark run.
        output_dir: Parent directory to create the run folder in.
        run_name: Folder name override. Defaults to `<model>_<timestamp>`.
        evals: Optional list of `EvalEntry` objects used for the run.
            Forwarded to `generate_html_report` so the per-result
            expansion panel can show the original test-case data
            (prompts, ground truth, figures, etc.). Leave as None if
            you only have an `EvalResult` list.

    Returns:
        Path to the created run directory.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = run.model_id.replace("/", "_").replace(":", "_").replace(".", "_")
        run_name = f"{model_slug}_{timestamp}"

    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "results.json", "w") as f:
        json.dump(run.model_dump(mode="json"), f, indent=2, default=str)

    generate_html_report(run, run_dir / "report.html", evals=evals)
    generate_markdown_report(run, run_dir / "report.md")

    return run_dir


def generate_markdown_report(run: BenchmarkRun, output_path: str | Path) -> None:
    """Generate a GitHub-renderable Markdown report alongside the HTML report."""
    overall = run.overall_score or 0.0
    started = (
        run.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if run.timestamp else "N/A"
    )

    def score_emoji(score: float) -> str:
        if score >= 0.85:
            return "🟢"
        if score >= 0.70:
            return "🟡"
        return "🔴"

    lines = [
        "# REVAL Benchmark Report",
        "",
        f"**Model:** `{run.model_id}`  ",
        f"**Overall Score:** {score_emoji(overall)} **{overall:.3f}** — {_score_label(overall)}  ",
        f"**Run Date:** {started}  ",
        f"**Evals:** {run.completed_evals}/{run.total_evals} completed, {run.error_count} errors  ",
        f"**Judge:** `{run.judge_model_id or 'N/A'}`  ",
        f"**Embeddings:** `{run.embeddings_model_id or 'N/A'}`",
        "",
        "---",
        "",
        "## Category Scores",
        "",
        "| Category | Score | Interpretation |",
        "|----------|-------|----------------|",
    ]

    for cat, score in sorted(run.category_scores.items()):
        lines.append(
            f"| {cat} | {score_emoji(score)} {score:.3f} | {_score_label(score)} |"
        )

    sorted_results = sorted(
        run.results,
        key=lambda r: (_cat(r.category), r.eval_id),
    )

    lines += [
        "",
        "---",
        "",
        "## Individual Results",
        "",
        "| Eval ID | Category | Score | Interpretation | Method |",
        "|---------|----------|-------|----------------|--------|",
    ]

    for r in sorted_results:
        method = _cat(r.scoring_method) if r.scoring_method else "—"
        lines.append(
            f"| {r.eval_id} | {_cat(r.category)} | {score_emoji(r.score)} {r.score:.3f} "
            f"| {_score_label(r.score)} | {method} |"
        )

    lines += ["", "---", "", "*Generated by [REVAL](../README.md)*", ""]

    Path(output_path).write_text("\n".join(lines))
