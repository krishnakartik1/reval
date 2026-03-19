"""Report generators and output helpers for REVAL benchmark results."""

import html
import json
from datetime import datetime
from pathlib import Path

from reval.models.eval import BenchmarkRun


def _score_color(score: float) -> str:
    if score >= 0.85:
        return "#22c55e"
    if score >= 0.70:
        return "#eab308"
    return "#ef4444"


def _score_bg(score: float) -> str:
    if score >= 0.85:
        return "#dcfce7"
    if score >= 0.70:
        return "#fef9c3"
    return "#fee2e2"


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Low bias"
    if score >= 0.70:
        return "Moderate"
    return "Potential bias"


def _esc(text: str) -> str:
    return html.escape(str(text))


def generate_html_report(run: BenchmarkRun, output_path: str | Path) -> None:
    """Generate a self-contained HTML dashboard report.

    Args:
        run: Completed benchmark run with results.
        output_path: Path to write the HTML file.
    """
    overall = run.overall_score or 0.0
    started = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if run.started_at else "N/A"

    # Sort results by category then eval_id
    sorted_results = sorted(
        run.results,
        key=lambda r: (
            r.category.value if hasattr(r.category, "value") else str(r.category),
            r.eval_id,
        ),
    )

    # Build category chart data
    cat_labels = json.dumps(list(run.category_scores.keys()))
    cat_values = json.dumps(list(run.category_scores.values()))
    cat_colors = json.dumps([_score_color(v) for v in run.category_scores.values()])

    # Build individual results rows
    result_rows = []
    for r in sorted_results:
        cat = r.category.value if hasattr(r.category, "value") else str(r.category)
        method = r.scoring_method.value if hasattr(r.scoring_method, "value") else str(r.scoring_method)
        color = _score_color(r.score)
        bg = _score_bg(r.score)

        # Detail panel content
        detail_parts = []

        if r.rubric_scores:
            rubric_html = '<div class="rubric-scores"><h4>Rubric Scores</h4>'
            for criterion, score_val in r.rubric_scores.items():
                # Normalize: rubric scores are 1-5 for judge, 0-1 for parity
                if score_val > 1:
                    pct = ((score_val - 1) / 4) * 100
                    display = f"{score_val:.0f}/5"
                else:
                    pct = score_val * 100
                    display = f"{score_val:.2f}"
                bar_color = _score_color(pct / 100)
                rubric_html += (
                    f'<div class="rubric-row">'
                    f'<span class="rubric-name">{_esc(criterion)}</span>'
                    f'<div class="rubric-bar-bg">'
                    f'<div class="rubric-bar" style="width:{pct:.0f}%;background:{bar_color}"></div>'
                    f'</div>'
                    f'<span class="rubric-val">{display}</span>'
                    f'</div>'
                )
            rubric_html += '</div>'
            detail_parts.append(rubric_html)

        if r.judge_reasoning:
            detail_parts.append(
                f'<div class="reasoning"><h4>Judge Reasoning</h4>'
                f'<p>{_esc(r.judge_reasoning)}</p></div>'
            )

        if r.similarity_score is not None:
            detail_parts.append(
                f'<div class="reasoning"><h4>Similarity Score</h4>'
                f'<p>{r.similarity_score:.4f}</p></div>'
            )

        if r.raw_response:
            truncated = r.raw_response[:500]
            if len(r.raw_response) > 500:
                truncated += "..."
            detail_parts.append(
                f'<div class="raw-response"><h4>Raw Response</h4>'
                f'<pre>{_esc(truncated)}</pre></div>'
            )

        detail_html = "".join(detail_parts) if detail_parts else '<p class="no-detail">No additional details</p>'

        result_rows.append(
            f'<tr class="result-row" onclick="toggleDetail(this)">'
            f'<td><span class="expand-icon">&#9654;</span> {_esc(r.eval_id)}</td>'
            f'<td>{_esc(cat)}</td>'
            f'<td><span class="score-badge" style="background:{bg};color:{color}">{r.score:.3f}</span></td>'
            f'<td>{_esc(_score_label(r.score))}</td>'
            f'<td>{_esc(method)}</td>'
            f'</tr>'
            f'<tr class="detail-row"><td colspan="5"><div class="detail-content">{detail_html}</div></td></tr>'
        )

    results_html = "\n".join(result_rows)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>REVAL Report - {_esc(run.model_id)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1.5rem; }}

  /* Header */
  .header {{ text-align: center; }}
  .header h1 {{ font-size: 1.5rem; color: #0f172a; margin-bottom: 0.25rem; }}
  .header .subtitle {{ color: #64748b; font-size: 0.875rem; }}
  .overall-score {{ font-size: 3rem; font-weight: 700; margin: 1rem 0 0.25rem; }}
  .overall-label {{ font-size: 1rem; font-weight: 500; margin-bottom: 1rem; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; text-align: left; margin-top: 1rem; }}
  .meta-item {{ background: #f8fafc; border-radius: 8px; padding: 0.75rem; }}
  .meta-item .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .meta-item .value {{ font-size: 0.875rem; font-weight: 500; word-break: break-all; }}

  /* Chart */
  .chart-container {{ position: relative; height: 250px; }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 0.75rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; border-bottom: 2px solid #e2e8f0; cursor: pointer; user-select: none; }}
  th:hover {{ color: #0f172a; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #f1f5f9; font-size: 0.875rem; }}
  .result-row {{ cursor: pointer; transition: background 0.15s; }}
  .result-row:hover {{ background: #f8fafc; }}
  .expand-icon {{ display: inline-block; transition: transform 0.2s; font-size: 0.7rem; color: #94a3b8; margin-right: 0.25rem; }}
  .result-row.expanded .expand-icon {{ transform: rotate(90deg); }}
  .score-badge {{ padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; }}
  .detail-row {{ display: none; }}
  .detail-row.visible {{ display: table-row; }}
  .detail-content {{ padding: 0.5rem 1rem 1rem 1.5rem; }}
  .detail-content h4 {{ font-size: 0.8rem; color: #475569; margin: 0.75rem 0 0.35rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .detail-content p {{ font-size: 0.85rem; color: #334155; line-height: 1.5; }}
  .detail-content pre {{ font-size: 0.8rem; background: #f8fafc; padding: 0.75rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #334155; }}
  .no-detail {{ color: #94a3b8; font-size: 0.85rem; }}

  /* Rubric bars */
  .rubric-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }}
  .rubric-name {{ font-size: 0.8rem; width: 180px; flex-shrink: 0; color: #475569; }}
  .rubric-bar-bg {{ flex: 1; height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; }}
  .rubric-bar {{ height: 100%; border-radius: 5px; transition: width 0.3s; }}
  .rubric-val {{ font-size: 0.8rem; width: 40px; text-align: right; font-weight: 500; }}

  .sort-arrow {{ font-size: 0.6rem; margin-left: 0.25rem; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header Card -->
  <div class="card header">
    <h1>REVAL Benchmark Report</h1>
    <p class="subtitle">Robust Evaluation of Values and Alignment in LLMs</p>
    <div class="overall-score" style="color:{_score_color(overall)}">{overall:.3f}</div>
    <div class="overall-label" style="color:{_score_color(overall)}">{_score_label(overall)}</div>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="label">Target Model</div>
        <div class="value">{_esc(run.model_id)}</div>
      </div>
      <div class="meta-item">
        <div class="label">Judge Model</div>
        <div class="value">{_esc(run.judge_model_id or 'N/A')}</div>
      </div>
      <div class="meta-item">
        <div class="label">Embeddings Model</div>
        <div class="value">{_esc(run.embeddings_model_id or 'N/A')}</div>
      </div>
      <div class="meta-item">
        <div class="label">Completed</div>
        <div class="value">{run.completed_evals} / {run.total_evals} (failed: {run.failed_evals})</div>
      </div>
      <div class="meta-item">
        <div class="label">Run Date</div>
        <div class="value">{started}</div>
      </div>
      <div class="meta-item">
        <div class="label">Run ID</div>
        <div class="value">{_esc(run.run_id[:12])}...</div>
      </div>
    </div>
  </div>

  <!-- Category Scores Chart -->
  <div class="card">
    <h2 style="font-size:1.1rem;margin-bottom:1rem;">Category Scores</h2>
    <div class="chart-container">
      <canvas id="categoryChart"></canvas>
    </div>
  </div>

  <!-- Individual Results -->
  <div class="card">
    <h2 style="font-size:1.1rem;margin-bottom:1rem;">Individual Results</h2>
    <table id="resultsTable">
      <thead>
        <tr>
          <th onclick="sortTable(0)">Eval ID <span class="sort-arrow"></span></th>
          <th onclick="sortTable(1)">Category <span class="sort-arrow"></span></th>
          <th onclick="sortTable(2)">Score <span class="sort-arrow"></span></th>
          <th onclick="sortTable(3)">Interpretation <span class="sort-arrow"></span></th>
          <th onclick="sortTable(4)">Method <span class="sort-arrow"></span></th>
        </tr>
      </thead>
      <tbody>
        {results_html}
      </tbody>
    </table>
  </div>

</div>

<script>
  // Category bar chart
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
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.x.toFixed(3)
          }}
        }}
      }},
      scales: {{
        x: {{ min: 0, max: 1, ticks: {{ stepSize: 0.2 }} }},
        y: {{ grid: {{ display: false }} }}
      }}
    }}
  }});

  // Expand/collapse detail rows
  function toggleDetail(row) {{
    const detail = row.nextElementSibling;
    const isVisible = detail.classList.contains('visible');
    // Collapse all first
    document.querySelectorAll('.detail-row.visible').forEach(r => r.classList.remove('visible'));
    document.querySelectorAll('.result-row.expanded').forEach(r => r.classList.remove('expanded'));
    if (!isVisible) {{
      detail.classList.add('visible');
      row.classList.add('expanded');
    }}
  }}

  // Sort table
  let sortDir = {{}};
  function sortTable(colIdx) {{
    const table = document.getElementById('resultsTable');
    const tbody = table.tBodies[0];
    // Collect result-row + detail-row pairs
    const pairs = [];
    const rows = Array.from(tbody.rows);
    for (let i = 0; i < rows.length; i += 2) {{
      pairs.push([rows[i], rows[i+1]]);
    }}
    sortDir[colIdx] = !(sortDir[colIdx] || false);
    const dir = sortDir[colIdx] ? 1 : -1;
    pairs.sort((a, b) => {{
      let va = a[0].cells[colIdx].textContent.trim();
      let vb = b[0].cells[colIdx].textContent.trim();
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return (na - nb) * dir;
      return va.localeCompare(vb) * dir;
    }});
    pairs.forEach(p => {{ tbody.appendChild(p[0]); tbody.appendChild(p[1]); }});
  }}
</script>
</body>
</html>"""

    Path(output_path).write_text(page)


def generate_markdown_report(run: BenchmarkRun, output_path: str | Path) -> None:
    """Generate a GitHub-renderable Markdown report.

    Args:
        run: Completed benchmark run with results.
        output_path: Path to write the Markdown file.
    """
    overall = run.overall_score or 0.0
    started = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if run.started_at else "N/A"

    def _emoji(score: float) -> str:
        if score >= 0.85:
            return "🟢"
        if score >= 0.70:
            return "🟡"
        return "🔴"

    lines = [
        "# REVAL Benchmark Report",
        "",
        f"**Model:** `{run.model_id}`  ",
        f"**Overall Score:** {_emoji(overall)} **{overall:.3f}** — {_score_label(overall)}  ",
        f"**Run Date:** {started}  ",
        f"**Evals:** {run.completed_evals}/{run.total_evals} completed, {run.failed_evals} failed  ",
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
        lines.append(f"| {cat} | {_emoji(score)} {score:.3f} | {_score_label(score)} |")

    sorted_results = sorted(
        run.results,
        key=lambda r: (
            r.category.value if hasattr(r.category, "value") else str(r.category),
            r.eval_id,
        ),
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
        cat = r.category.value if hasattr(r.category, "value") else str(r.category)
        method = r.scoring_method.value if hasattr(r.scoring_method, "value") else str(r.scoring_method)
        lines.append(
            f"| {r.eval_id} | {cat} | {_emoji(r.score)} {r.score:.3f} | {_score_label(r.score)} | {method} |"
        )

    lines += ["", "---", "", "*Generated by [REVAL](../README.md)*", ""]

    Path(output_path).write_text("\n".join(lines))


def save_run_outputs(run: BenchmarkRun, output_dir: str | Path, run_name: str | None = None) -> Path:
    """Save all run outputs (JSON, HTML, Markdown) into a subdirectory.

    Args:
        run: Completed benchmark run.
        output_dir: Parent directory to create the run folder in.
        run_name: Folder name override. Defaults to run_<timestamp>.

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

    generate_html_report(run, run_dir / "report.html")
    generate_markdown_report(run, run_dir / "report.md")

    return run_dir


def generate_markdown_report(run: BenchmarkRun, output_path: str | Path) -> None:
    """Generate a GitHub-renderable Markdown report alongside the HTML report."""
    overall = run.overall_score or 0.0
    started = run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if run.started_at else "N/A"

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
        f"**Evals:** {run.completed_evals}/{run.total_evals} completed, {run.failed_evals} failed  ",
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
        lines.append(f"| {cat} | {score_emoji(score)} {score:.3f} | {_score_label(score)} |")

    sorted_results = sorted(
        run.results,
        key=lambda r: (
            r.category.value if hasattr(r.category, "value") else str(r.category),
            r.eval_id,
        ),
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
        cat = r.category.value if hasattr(r.category, "value") else str(r.category)
        method = r.scoring_method.value if hasattr(r.scoring_method, "value") else str(r.scoring_method)
        lines.append(
            f"| {r.eval_id} | {cat} | {score_emoji(r.score)} {r.score:.3f} | {_score_label(r.score)} | {method} |"
        )

    lines += ["", "---", "", f"*Generated by [REVAL](../README.md)*", ""]

    Path(output_path).write_text("\n".join(lines))
