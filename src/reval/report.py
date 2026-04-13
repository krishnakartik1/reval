"""Report generators and output helpers for REVAL benchmark results.

The per-run HTML report shares its design language with the static
leaderboard (`reval.leaderboard`) via `get_style_css()` — same
Archival Data Journalism aesthetic: warm parchment, rust accent,
Fraunces variable-serif display type, Inter body, JetBrains Mono for
IDs. Clicking a row on the public leaderboard and landing on that
run's embedded report is visually seamless.

Each result card expands into three stacked sections:

1. **Test case**    — per-category rendering of the prompt(s),
                      entities / figures, ground truth, citations, etc.
                      Requires the original `EvalEntry` (pass `evals=`
                      to `generate_html_report` or `save_run_outputs`).
2. **Response**     — full, scrollable model output. Paired-prompt
                      categories show both responses side by side.
                      Truncation-free.
3. **Score**        — hero score number, method, latency, sub-scores,
                      rubric bars, judge reasoning.
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


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Low bias"
    if score >= 0.70:
        return "Moderate"
    return "Potential bias"


def _score_label_upper(score: float) -> str:
    """Eyebrow-style variant of `_score_label`."""
    return _score_label(score).upper()


def _esc(text: object) -> str:
    return html.escape(str(text))


def _cat(value: object) -> str:
    """Unwrap an EvalCategory enum (or leave a string alone)."""
    return value.value if hasattr(value, "value") else str(value)


# ── Category icon mapping (Lucide names) ───────────────────────────────

_CATEGORY_ICONS: dict[EvalCategory, str] = {
    EvalCategory.POLICY_ATTRIBUTION: "scale",
    EvalCategory.FIGURE_TREATMENT: "users",
    EvalCategory.ISSUE_FRAMING: "message-square-quote",
    EvalCategory.FACTUAL_ACCURACY: "shield-check",
    EvalCategory.ARGUMENTATION_PARITY: "git-compare-arrows",
}


def _category_icon(category: object) -> str:
    """Return the Lucide icon name for a category, defaulting to 'circle'."""
    try:
        enum_val = (
            category if isinstance(category, EvalCategory) else EvalCategory(category)
        )
        return _CATEGORY_ICONS.get(enum_val, "circle")
    except (KeyError, ValueError):
        return "circle"


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
    matter for that category. Missing fields render nothing to keep the
    panel tight.
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
        cpair = entry.counterfactual_pair
        parts.append(
            '<div class="prompt-pair">'
            + _render_prompt_block(f"Prompt A · {cpair.entity_a}", cpair.prompt_a)
            + _render_prompt_block(f"Prompt B · {cpair.entity_b}", cpair.prompt_b)
            + "</div>"
            f'<div class="kv">{_render_kv("Policy topic", cpair.policy_topic)}</div>'
        )

    elif cat == EvalCategory.FIGURE_TREATMENT and entry.figure_pair:
        fpair = entry.figure_pair
        parts.append(
            '<div class="prompt-pair">'
            + _render_prompt_block(
                f"Prompt A · {fpair.figure_a} ({fpair.affiliation_a})",
                fpair.prompt_a,
            )
            + _render_prompt_block(
                f"Prompt B · {fpair.figure_b} ({fpair.affiliation_b})",
                fpair.prompt_b,
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
                f'<div class="gt-header">Ground truth'
                f' <span class="gt-level">{_esc(level_label)}</span></div>'
                f'<div class="gt-claim"><strong>Claim:</strong> {_esc(gt.claim)}</div>'
                f'<div class="gt-correct"><strong>Correct response:</strong>'
                f" {_esc(gt.correct_response)}</div>"
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
    """Render the 'Response' section for a single EvalResult."""
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
        f'<div class="score-hero {_score_color(result.score)}">'
        f'<div class="score-hero-label eyebrow">Overall</div>'
        f'<div class="score-hero-value tabular">{result.score:.3f}</div>'
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


# ── Report-specific CSS on top of tokens.css + style.css ──────────────
#
# Most of the layout now lives in the shared leaderboard stylesheet
# (tokens.css + style.css from `reval.leaderboard`). What's kept here
# is report-specific: the run hero card, expandable result cards,
# collapsible sections, and the test-case / prompt / ground-truth /
# response / rubric / sub-score panels unique to per-run reports.


_REPORT_EXTRA_CSS = """
/* ── Report page chrome ────────────────────────────────────────── */

.report-page { padding-top: 56px; padding-bottom: 96px; }

.report-hero {
  display: grid;
  grid-template-columns: 1fr;
  gap: 40px;
  align-items: end;
}
@media (min-width: 900px) {
  .report-hero { grid-template-columns: 1fr auto; gap: 64px; }
}

.report-hero-left .eyebrow-accent { margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.report-hero-left .eyebrow-accent::before {
  content: ""; display: inline-block; width: 24px; height: 1px; background: var(--accent);
}
.report-hero-title {
  font-family: var(--font-mono);
  font-feature-settings: "tnum", "zero";
  letter-spacing: -0.02em;
  font-size: clamp(24px, 4vw, 42px);
  line-height: 1.05;
  color: var(--fg);
  word-break: break-word;
  margin: 0;
}
.report-hero-sub {
  margin-top: 18px;
  color: var(--fg-dim);
  font-size: 13px;
  line-height: 1.5;
}

.report-hero-card {
  padding: 28px 32px;
  border-radius: 8px;
  min-width: 260px;
}
.report-hero-card.score-high { background: var(--score-high-bg); }
.report-hero-card.score-mid  { background: var(--score-mid-bg); }
.report-hero-card.score-low  { background: var(--score-low-bg); }
.report-hero-card.score-none { background: var(--score-none-bg); }
.report-hero-card .run-hero-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 500;
  opacity: 0.85;
}
.report-hero-card.score-high .run-hero-label,
.report-hero-card.score-high .run-hero-value,
.report-hero-card.score-high .run-hero-sub { color: var(--score-high-fg); }
.report-hero-card.score-mid .run-hero-label,
.report-hero-card.score-mid .run-hero-value,
.report-hero-card.score-mid .run-hero-sub { color: var(--score-mid-fg); }
.report-hero-card.score-low .run-hero-label,
.report-hero-card.score-low .run-hero-value,
.report-hero-card.score-low .run-hero-sub { color: var(--score-low-fg); }
.report-hero-card.score-none .run-hero-label,
.report-hero-card.score-none .run-hero-value,
.report-hero-card.score-none .run-hero-sub { color: var(--score-none-fg); }

.run-hero-value {
  font-family: var(--font-display);
  font-variation-settings: "opsz" 144, "wght" 500, "SOFT" 40;
  font-size: 72px;
  font-variant-numeric: tabular-nums;
  line-height: 0.95;
  margin: 8px 0 6px;
  letter-spacing: -0.02em;
}
.run-hero-sub {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.8;
}

/* Metadata strip — hairline grid, mirrors leaderboard model page */
.run-hero-meta {
  margin-top: 48px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
}
@media (min-width: 700px)  { .run-hero-meta { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1100px) { .run-hero-meta { grid-template-columns: repeat(6, 1fr); } }
.run-hero-meta .kv {
  padding: 20px;
  background: var(--bg);
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}
.run-hero-meta .kv-label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
}
.run-hero-meta .kv-value {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--fg);
  word-break: break-word;
}

/* ── Gradient hero rule (defined in style.css, but margins tuned) ── */
.report-rule { margin: 64px 0 48px; }

h2.section-title {
  font-family: var(--font-body);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
  margin: 0 0 16px;
}

/* ── Category chart card ──────────────────────────────────────── */
.chart-card {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 28px;
  margin-bottom: 56px;
}
.chart-container { position: relative; height: 280px; }

/* ── Result cards (expandable) ────────────────────────────────── */
.result-list { display: flex; flex-direction: column; gap: 8px; }

.result-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  overflow: hidden;
  transition: border-color var(--dur) var(--ease);
}
.result-card:hover { border-color: var(--accent); }
.result-card.expanded { border-color: var(--accent); }

.result-header {
  display: grid;
  grid-template-columns: auto auto 1fr auto auto auto;
  align-items: center;
  gap: 14px;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background-color var(--dur) var(--ease);
}
.result-header:hover { background: var(--bg-alt); }
.expand-icon {
  width: 12px;
  height: 12px;
  color: var(--fg-dim);
  transition: transform var(--dur) var(--ease);
  flex-shrink: 0;
}
.result-card.expanded .expand-icon { transform: rotate(90deg); color: var(--accent); }

.result-cat-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  color: var(--fg-dim);
}
.result-card:hover .result-cat-icon { color: var(--accent); border-color: var(--accent); }

.result-eval-id {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--fg);
  letter-spacing: -0.01em;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.result-category {
  font-size: 11px;
  color: var(--fg-dim);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500;
  white-space: nowrap;
}
.result-method {
  font-size: 10px;
  color: var(--fg-dim);
  font-family: var(--font-mono);
  white-space: nowrap;
  display: none;
}
@media (min-width: 900px) { .result-method { display: inline; } }

.result-body {
  display: none;
  padding: 0 20px 20px;
  border-top: 1px solid var(--border-soft);
}
.result-card.expanded .result-body { display: block; }

/* Stacked sections inside a result card */
.section {
  margin-top: 20px;
  border-radius: 6px;
  background: var(--bg-alt);
  overflow: hidden;
  border: 1px solid var(--border-soft);
}
.section-header {
  padding: 10px 16px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
  background: var(--bg);
  border-bottom: 1px solid var(--border-soft);
}
.section-body { padding: 20px; }

/* ── Test case panel ──────────────────────────────────────────── */
.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px 20px;
  margin-bottom: 20px;
}
.kv-grid.tight { margin-bottom: 0; }
.kv { display: flex; flex-direction: column; font-size: 12px; gap: 3px; }
.kv-label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-dim);
}
.kv-value {
  font-family: var(--font-mono);
  word-break: break-word;
  color: var(--fg);
}

.prompt-pair {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
  margin-bottom: 14px;
}
@media (min-width: 820px) {
  .prompt-pair { grid-template-columns: 1fr 1fr; }
}
.prompt-block {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 14px 18px;
}
.prompt-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--fg-dim);
  margin-bottom: 8px;
}
.prompt-body {
  font-size: 14px;
  line-height: 1.55;
  white-space: pre-wrap;
  color: var(--fg);
}

.ground-truth {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 18px 20px;
  margin-top: 14px;
  font-size: 13px;
  line-height: 1.55;
}
.gt-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg);
  margin-bottom: 14px;
}
.gt-level {
  font-size: 9px;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--accent-tint);
  color: var(--accent);
  font-weight: 600;
  letter-spacing: 0.1em;
}
.gt-claim, .gt-correct { margin: 10px 0; color: var(--fg); }
.gt-claim strong, .gt-correct strong,
.gt-citations strong, .gt-misconceptions strong {
  font-weight: 600;
  color: var(--fg);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  display: block;
  margin-bottom: 4px;
}
.gt-citations, .gt-misconceptions { margin-top: 14px; font-size: 12px; color: var(--fg-dim); }
.gt-citations ul, .gt-misconceptions ul { margin: 4px 0 0 20px; padding: 0; }
.gt-citations li, .gt-misconceptions li { margin: 3px 0; }
.gt-citations a { color: var(--accent); text-decoration: underline; }

.tags { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 10px;
  color: var(--fg-dim);
  font-family: var(--font-mono);
}

/* ── Response panel ───────────────────────────────────────────── */
.response-block {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 14px;
}
.response-block:last-child { margin-bottom: 0; }
.response-label {
  padding: 10px 16px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
  background: var(--bg-alt);
  border-bottom: 1px solid var(--border-soft);
}
.response-body {
  max-height: 420px;
  overflow-y: auto;
  margin: 0;
  padding: 18px 20px;
  font-family: var(--font-mono);
  font-size: 12.5px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--fg);
}

/* ── Score panel ──────────────────────────────────────────────── */
.score-hero {
  padding: 22px 26px;
  border-radius: 8px;
  margin-bottom: 20px;
}
.score-hero.score-high { background: var(--score-high-bg); color: var(--score-high-fg); }
.score-hero.score-mid  { background: var(--score-mid-bg);  color: var(--score-mid-fg); }
.score-hero.score-low  { background: var(--score-low-bg);  color: var(--score-low-fg); }
.score-hero.score-none { background: var(--score-none-bg); color: var(--score-none-fg); }
.score-hero-label {
  opacity: 0.85;
  color: inherit !important;
}
.score-hero-value {
  font-family: var(--font-display);
  font-variation-settings: "opsz" 96, "wght" 500, "SOFT" 30;
  font-size: 44px;
  font-variant-numeric: tabular-nums;
  line-height: 0.95;
  margin: 6px 0 4px;
  letter-spacing: -0.015em;
}
.score-hero-sub {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 500;
  opacity: 0.85;
}

.section-h4 {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--fg-dim);
  margin: 24px 0 10px;
}

.rubric-bars { display: flex; flex-direction: column; gap: 8px; }
.rubric-row { display: flex; align-items: center; gap: 14px; }
.rubric-name {
  width: 200px;
  font-size: 13px;
  flex-shrink: 0;
  color: var(--fg);
}
.rubric-row .rubric-bar-bg {
  flex: 1;
  height: 8px;
}
.rubric-val {
  width: 56px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--fg);
}

.sub-scores {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
}
.sub-score {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
}
.sub-score-label {
  font-size: 12px;
  color: var(--fg);
}

.judge-reasoning {
  background: var(--bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 16px 18px;
  font-size: 13px;
  line-height: 1.65;
  white-space: pre-wrap;
  color: var(--fg);
}

.empty-section {
  color: var(--fg-dim);
  font-style: italic;
  padding: 20px;
  font-size: 13px;
}
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
            the run's `eval_ids`. When provided, each result expands
            into a three-section panel: test case, response, score.
            When None, the test-case section is omitted.
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

    # Category chart data — colors pulled from CSS vars at render time
    # (see the <script> block below) so the chart respects light/dark
    # themes natively.
    cat_labels = json.dumps(list(run.category_scores.keys()))
    cat_values = json.dumps(list(run.category_scores.values()))

    # Result cards
    card_chunks: list[str] = []
    for result in sorted_results:
        matched_entry: EvalEntry | None = eval_by_id.get(result.eval_id)
        category_label = _cat(result.category)
        method_label = _cat(result.scoring_method) if result.scoring_method else "—"
        icon_name = _category_icon(result.category)

        sections: list[str] = []
        if matched_entry is not None:
            sections.append(
                '<div class="section">'
                '<div class="section-header">Test case</div>'
                f"{_render_test_case(matched_entry)}"
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
            '<svg class="expand-icon" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2">'
            '<path d="M4 2 L8 6 L4 10"/></svg>'
            f'<span class="result-cat-icon"><i data-lucide="{_esc(icon_name)}"'
            ' style="width:14px;height:14px"></i></span>'
            f'<span class="result-eval-id">{_esc(result.eval_id)}</span>'
            f'<span class="result-category">{_esc(category_label)}</span>'
            f'<span class="result-method">{_esc(method_label)}</span>'
            f'<span class="score-cell {_score_color(result.score)}">{result.score:.3f}</span>'
            "</header>"
            f'<div class="result-body">{"".join(sections)}</div>'
            "</article>"
        )

    results_html = "\n".join(card_chunks)

    # Hero meta items — same 6-cell grid as leaderboard model detail page
    meta_items = [
        ("Target model", run.model_id or "—"),
        ("Judge", run.judge_model_id or "—"),
        ("Embeddings", run.embeddings_model_id or "—"),
        (
            "Completed",
            f"{run.completed_evals}/{run.total_evals}"
            + (f" · {run.error_count} err" if run.error_count else ""),
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
<meta name="theme-color" content="#faf8f3">
<title>REVAL Report · {_esc(run.model_id)}</title>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..900,0..100&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/lucide@0.363.0/dist/umd/lucide.min.js"></script>

<style>
{style}
</style>
</head>
<body>

<div class="grain" aria-hidden="true"></div>

<header class="site-header">
  <div class="wrap">
    <h1>
      <a href="../index.html" class="flex items-baseline gap-3" style="text-decoration:none;">
        <span>REVAL</span>
        <span class="subtitle">Benchmark report</span>
      </a>
    </h1>
    <nav style="display:flex;align-items:center;gap:20px;">
      <a href="https://github.com/krishnakartik1/reval"
         style="color:var(--fg-dim);font-size:11px;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;">
        GitHub
      </a>
    </nav>
  </div>
</header>

<main class="wrap report-page">

  <section class="report-hero">
    <div class="report-hero-left">
      <div class="eyebrow-accent">
        <span>Reval benchmark</span>
        <span aria-hidden="true">·</span>
        <span>Run report</span>
      </div>
      <h2 class="report-hero-title">{_esc(run.model_id)}</h2>
      <p class="report-hero-sub">
        {run.completed_evals} of {run.total_evals} evaluations completed
        {f" · {run.error_count} errors" if run.error_count else ""}
      </p>
    </div>

    <div class="report-hero-card {_score_color(overall)}">
      <div class="eyebrow run-hero-label">Overall score</div>
      <div class="run-hero-value">{overall:.3f}</div>
      <div class="run-hero-sub">{_esc(_score_label_upper(overall))}</div>
    </div>
  </section>

  <div class="run-hero-meta">{meta_html}</div>

  <hr class="hero-rule report-rule">

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

<footer class="site-footer no-print">
  <div class="wrap">
    <p>Generated by <code class="mono">reval run</code></p>
    <p><a href="https://github.com/krishnakartik1/reval">github.com/krishnakartik1/reval</a></p>
  </div>
</footer>

<script>
  // Resolve palette colors from CSS vars at render time so the chart
  // respects light/dark theme automatically.
  (function () {{
    function cssVar(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    function scoreColor(v) {{
      if (v >= 0.85) return cssVar('--score-high-fg') || '#2d7a4f';
      if (v >= 0.70) return cssVar('--score-mid-fg')  || '#b77821';
      return cssVar('--score-low-fg') || '#a83032';
    }}

    var labels = {cat_labels};
    var values = {cat_values};
    var colors = values.map(scoreColor);

    var gridColor = cssVar('--border') || '#d9d3c3';
    var axisColor = cssVar('--fg-dim') || '#6b6660';

    var ctx = document.getElementById('categoryChart');
    if (ctx && window.Chart) {{
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: labels.map(l => l.replace(/_/g, ' ')),
          datasets: [{{
            data: values,
            backgroundColor: colors,
            borderRadius: 4,
            barThickness: 24,
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              backgroundColor: cssVar('--bg-alt') || '#f4f0e6',
              titleColor: cssVar('--fg') || '#1a1a1a',
              bodyColor: cssVar('--fg-dim') || '#6b6660',
              borderColor: cssVar('--accent') || '#a85c32',
              borderWidth: 1,
              padding: 12,
              titleFont: {{ family: 'Inter, sans-serif', size: 11, weight: '600' }},
              bodyFont: {{ family: 'JetBrains Mono, monospace', size: 11 }},
              callbacks: {{ label: function (ctx) {{ return ctx.parsed.x.toFixed(3); }} }}
            }}
          }},
          scales: {{
            x: {{
              min: 0, max: 1,
              ticks: {{ stepSize: 0.2, color: axisColor, font: {{ family: 'JetBrains Mono', size: 10 }} }},
              grid: {{ color: gridColor }}
            }},
            y: {{
              ticks: {{ color: axisColor, font: {{ family: 'Inter', size: 11, weight: '500' }} }},
              grid: {{ display: false }}
            }}
          }}
        }}
      }});
    }}
  }})();

  // Expand/collapse result cards — one-at-a-time toggling is ugly
  // here; we want independent cards so users can compare side by
  // side. So each click toggles its own card only.
  document.querySelectorAll('.result-header').forEach(function (h) {{
    h.addEventListener('click', function () {{
      h.parentElement.classList.toggle('expanded');
    }});
  }});

  // Render Lucide icons once ready
  (function () {{
    function draw() {{
      if (window.lucide) {{ window.lucide.createIcons(); return true; }}
      return false;
    }}
    if (!draw()) {{
      var tick = setInterval(function () {{ if (draw()) clearInterval(tick); }}, 50);
    }}
  }})();
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
            expansion panel can show the original test-case data.

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
