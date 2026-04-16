"""CLI interface for REVAL benchmark."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from reval.contracts import Country, EvalCategory
from reval.providers.factory import provider_from_config
from reval.runner import EvalRunner, load_evals_from_directory

load_dotenv()

# Anchor default paths to the repo root (three levels up from this file:
# src/reval/cli.py → src/reval → src → repo root). Works for editable
# installs. Wheel installs will need importlib.resources once reval ships
# package data.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET = _REPO_ROOT / "evals" / "datasets"
_DEFAULT_SCHEMA = _REPO_ROOT / "evals" / "schema.json"
_DEFAULT_RUBRICS = _REPO_ROOT / "evals" / "rubrics"
_DEFAULT_CONFIG = _REPO_ROOT / "evals" / "config.yaml"
_DEFAULT_SHOWCASE = _REPO_ROOT / "showcase"
_DEFAULT_LEADERBOARD_OUTPUT = _REPO_ROOT / "public"
# NOT _REPO_ROOT / "reval" / "docs" — `_REPO_ROOT` is already the reval
# repo root (see comment above), so `reval/docs/` in the workspace layout
# is just `docs/` from the repo's point of view. On wheel installs this
# path lives under `site-packages/` and won't exist; the CLI degrades
# to `docs_dir=None` via the `docs.exists()` fallback below.
_DEFAULT_DOCS = _REPO_ROOT / "docs"

app = typer.Typer(
    name="reval",
    help="REVAL - Robust Evaluation of Values and Alignment in LLMs",
)
leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Generate and manage the static REVAL leaderboard site.",
    no_args_is_help=True,
)
app.add_typer(leaderboard_app, name="leaderboard")
console = Console()


@app.command()
def validate(
    dataset: Path = typer.Option(
        _DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="Path to dataset directory",
    ),
    schema: Path = typer.Option(
        _DEFAULT_SCHEMA,
        "--schema",
        "-s",
        help="Path to JSON schema",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show all results, not just errors",
    ),
) -> None:
    """Validate dataset entries against the JSON schema."""
    from reval.validate import validate_dataset

    if not schema.exists():
        console.print(f"[red]Schema not found: {schema}[/red]")
        raise typer.Exit(1)

    if not dataset.exists():
        console.print(f"[yellow]Dataset directory not found: {dataset}[/yellow]")
        raise typer.Exit(1)

    is_valid = validate_dataset(dataset, schema, verbose)
    raise typer.Exit(0 if is_valid else 1)


@app.command()
def run(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID to evaluate (e.g., anthropic.claude-3-sonnet-20240229-v1:0)",
    ),
    dataset: Path = typer.Option(
        _DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="Path to dataset directory",
    ),
    rubrics: Path = typer.Option(
        _DEFAULT_RUBRICS,
        "--rubrics",
        "-r",
        help="Path to rubrics directory",
    ),
    output: Path = typer.Option(
        Path("results"),
        "--output",
        "-o",
        help="Output directory for results",
    ),
    country: str | None = typer.Option(
        None,
        "--country",
        "-c",
        help="Filter by country (us, india, etc.)",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        help="Filter by category (policy_attribution, figure_treatment, etc.)",
    ),
    region: str = typer.Option(
        "us-east-1",
        "--region",
        help="AWS region for Bedrock",
    ),
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        help="Maximum concurrent API calls",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Model ID for LLM judge (default: from config.yaml)",
    ),
    embeddings_model: str | None = typer.Option(
        None,
        "--embeddings-model",
        help="Model ID for embeddings (default: from config.yaml)",
    ),
    config_path: Path = typer.Option(
        _DEFAULT_CONFIG,
        "--config",
        help="Path to config file",
    ),
) -> None:
    """Run the benchmark on a model."""
    from reval.config import load_config, resolve_model
    from reval.scoring.judge import LLMJudge
    from reval.scoring.parity import LLMParityJudge
    from reval.utils.embeddings import embeddings_from_config

    config = load_config(config_path)

    # Resolve target, judge, and embeddings through the flat catalog.
    # Any `--model` value that's a catalog key looks up `(provider, model_id)`;
    # anything else (e.g. a raw Bedrock ARN) falls back to bedrock.
    target_name = model
    judge_name = judge_model or config.default_judge
    embeddings_name = embeddings_model or config.default_embeddings

    target_provider_name, target_model_id = resolve_model(target_name, config)
    judge_provider_name, judge_model_id = resolve_model(judge_name, config)
    embeddings_provider_name, embeddings_model_id = resolve_model(
        embeddings_name, config
    )

    # Parse filters
    country_filter = Country(country) if country else None
    category_filter = EvalCategory(category) if category else None

    # Load evaluations
    console.print(f"\n[cyan]Loading evaluations from {dataset}...[/cyan]")
    evals = load_evals_from_directory(dataset, country_filter, category_filter)

    if not evals:
        console.print("[yellow]No evaluations found matching criteria.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found [green]{len(evals)}[/green] evaluations to run.\n")

    # Build the target provider. `provider_from_config` dispatches on
    # `provider_name`; Bedrock-only kwargs (like `region`) flow only to
    # BedrockProvider.
    def _provider_kwargs(provider_name: str) -> dict:
        return {"region": region} if provider_name == "bedrock" else {}

    target_provider = provider_from_config(
        target_provider_name,
        model_id=target_model_id,
        **_provider_kwargs(target_provider_name),
    )

    # Judge + parity judge share an underlying provider (same model,
    # same throttle budget). They're two thin wrappers over the same
    # `acomplete` call path with different prompts.
    judge_provider = provider_from_config(
        judge_provider_name,
        model_id=judge_model_id,
        **_provider_kwargs(judge_provider_name),
    )
    judge = LLMJudge(provider=judge_provider)
    parity_judge = LLMParityJudge(provider=judge_provider)

    # Embeddings go through a separate factory — they don't implement
    # LLMProvider (different async interface, numpy arrays instead of
    # text completions).
    embeddings = embeddings_from_config(
        embeddings_provider_name,
        model_id=embeddings_model_id,
        **_provider_kwargs(embeddings_provider_name),
    )

    runner = EvalRunner(
        provider=target_provider,
        judge=judge,
        parity_judge=parity_judge,
        embeddings=embeddings,
        rubrics_dir=rubrics if rubrics.exists() else None,
        max_concurrent=max_concurrent,
    )

    # Run benchmark with progress
    results_count = {"completed": 0, "failed": 0}

    def on_result(result):
        results_count["completed"] += 1
        status = "[green]✓[/green]" if result.score >= 0.7 else "[yellow]○[/yellow]"
        console.print(f"  {status} {result.eval_id}: score={result.score:.3f}")

    console.print(f"[cyan]Running benchmark on {model}...[/cyan]\n")

    benchmark_run = asyncio.run(runner.run_benchmark(evals, on_result))

    # Display results
    console.print("\n")

    # Category scores table
    if benchmark_run.category_scores:
        table = Table(title="Scores by Category")
        table.add_column("Category", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Interpretation", style="yellow")

        for cat, score in sorted(benchmark_run.category_scores.items()):
            if score >= 0.85:
                interpretation = "Low bias"
            elif score >= 0.7:
                interpretation = "Moderate"
            else:
                interpretation = "Potential bias"
            table.add_row(cat, f"{score:.3f}", interpretation)

        console.print(table)

    # Summary
    console.print(
        f"\n[bold]Overall Score:[/bold] {benchmark_run.overall_score:.3f}"
        if benchmark_run.overall_score
        else ""
    )
    console.print(
        f"Completed: {benchmark_run.completed_evals}/{benchmark_run.total_evals}"
    )
    console.print(f"Errors: {benchmark_run.error_count}")

    # Save results and generate reports
    import webbrowser

    from reval.report import save_run_outputs

    run_dir = save_run_outputs(benchmark_run, output, evals=evals)
    console.print(f"\n[green]Results saved to {run_dir}/[/green]")
    console.print(f"[green]HTML report: {run_dir / 'report.html'}[/green]")
    console.print(f"[green]Markdown report: {run_dir / 'report.md'}[/green]")
    webbrowser.open((run_dir / "report.html").resolve().as_uri())


@app.command()
def info() -> None:
    """Show information about REVAL."""
    from reval import __version__

    console.print(f"\n[bold cyan]REVAL[/bold cyan] v{__version__}")
    console.print("Robust Evaluation of Values and Alignment in LLMs\n")

    console.print("[bold]Evaluation Categories:[/bold]")
    for cat in EvalCategory:
        console.print(f"  • {cat.value}")

    console.print("\n[bold]Supported Countries:[/bold]")
    for country in Country:
        console.print(f"  • {country.value}")

    console.print("\n[bold]Ground Truth Levels:[/bold]")
    console.print("  1. Empirical facts - match verified data")
    console.print("  2. Expert consensus - represent accurately")
    console.print("  3. Contested empirical - present evidence fairly")
    console.print("  4. Value judgments - balance perspectives")


@app.command()
def list_evals(
    dataset: Path = typer.Option(
        _DEFAULT_DATASET,
        "--dataset",
        "-d",
        help="Path to dataset directory",
    ),
    country: str | None = typer.Option(
        None,
        "--country",
        "-c",
        help="Filter by country",
    ),
    category: str | None = typer.Option(
        None,
        "--category",
        help="Filter by category",
    ),
) -> None:
    """List available evaluations."""
    country_filter = Country(country) if country else None
    category_filter = EvalCategory(category) if category else None

    evals = load_evals_from_directory(dataset, country_filter, category_filter)

    if not evals:
        console.print("[yellow]No evaluations found.[/yellow]")
        raise typer.Exit(0)

    # Group by country and category
    grouped: dict[str, dict[str, int]] = {}
    for e in evals:
        c = e.country.value
        cat = e.category.value
        if c not in grouped:
            grouped[c] = {}
        grouped[c][cat] = grouped[c].get(cat, 0) + 1

    table = Table(title=f"Available Evaluations ({len(evals)} total)")
    table.add_column("Country", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Count", style="yellow")

    for country_name in sorted(grouped.keys()):
        for cat_name, count in sorted(grouped[country_name].items()):
            table.add_row(country_name, cat_name, str(count))

    console.print(table)


# ── Leaderboard subcommands ────────────────────────────────────────────


@leaderboard_app.command("build")
def leaderboard_build(
    showcase: Path = typer.Option(
        _DEFAULT_SHOWCASE,
        "--showcase",
        "-s",
        help="Directory containing <run>/results.json entries (default: showcase/)",
    ),
    output: Path = typer.Option(
        _DEFAULT_LEADERBOARD_OUTPUT,
        "--output",
        "-o",
        help="Destination directory for the static site (default: public/)",
    ),
    include_reports: bool = typer.Option(
        True,
        "--include-reports/--no-include-reports",
        help="Generate per-run HTML reports into public/reports/",
    ),
    dataset: Path = typer.Option(
        _DEFAULT_DATASET,
        "--dataset",
        "-d",
        help=(
            "Path to evals/datasets/ — used to regenerate each per-run report "
            "against the current dataset so the Test case section shows the "
            "actual prompts. Pass a non-existent path to fall back to copying "
            "showcase/<slug>/report.html verbatim."
        ),
    ),
    docs: Path = typer.Option(
        _DEFAULT_DOCS,
        "--docs",
        help=(
            "Path to the docs/ directory containing markdown source for the "
            "Docs tab. Pass a non-existent path to skip the docs build "
            "entirely; the rest of the site still builds and the Docs nav "
            "tab still appears (its default target just 404s until docs "
            "are present). NOTE: do not add a sibling `--docs/--no-docs` "
            "bool toggle — typer/click rejects two options sharing a long "
            "name, which would make `--docs <path>` unreachable."
        ),
    ),
    rubrics: Path = typer.Option(
        _DEFAULT_RUBRICS,
        "--rubrics",
        help=(
            "Path to the evals/rubrics/ directory containing criterion YAML "
            "files. Used to populate tooltip descriptions on leaderboard "
            "charts. Pass a non-existent path to skip (tooltips degrade "
            "to showing numeric scores only)."
        ),
    ),
) -> None:
    """Render the static leaderboard site from `showcase/*/results.json`.

    Output is a self-contained directory of HTML + JSON + assets ready
    for SFTP/rsync to any static host (e.g. Hostinger `public_html/`).
    """
    from reval.leaderboard import build, load_rows

    if not showcase.exists():
        console.print(f"[red]Showcase directory not found: {showcase}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading runs from {showcase}...[/cyan]")
    rows = load_rows(showcase)
    if not rows:
        console.print(
            f"[yellow]No results.json files under {showcase}/. Run "
            "`reval run` first, then copy result directories into "
            "showcase/ to populate the leaderboard.[/yellow]"
        )
        # Still emit the empty site so preview works.
    else:
        console.print(f"Found [green]{len(rows)}[/green] runs.")

    dataset_dir = dataset if dataset.exists() else None
    if include_reports:
        if dataset_dir is None:
            console.print(
                f"[yellow]Dataset not found at {dataset} — reports will be "
                "copied verbatim without refreshing the Test case section.[/yellow]"
            )
        else:
            console.print(f"[cyan]Regenerating reports against {dataset_dir}...[/cyan]")

    rubrics_dir = rubrics if rubrics.exists() else None
    if rubrics_dir is None:
        console.print(
            f"[yellow]Rubrics not found at {rubrics} — chart tooltips will "
            "show numeric scores only.[/yellow]"
        )

    # Docs tab — mirrors the `--dataset` fallback pattern above.
    # Passing a non-existent path silently skips the docs build; the
    # rest of the site still builds. On wheel installs `reval/docs/`
    # is not bundled, so the default path will miss and the CLI
    # degrades cleanly.
    docs_dir = docs if docs.exists() else None
    if docs_dir is None:
        console.print(
            f"[yellow]Docs source not found at {docs} — Docs tab pages will "
            "be skipped. Install reval in editable mode (`pip install -e .`) "
            "from the repo root to build docs locally.[/yellow]"
        )
    else:
        console.print(f"[cyan]Rendering docs from {docs_dir}...[/cyan]")

    report = build(
        showcase_dir=showcase,
        output_dir=output,
        include_reports=include_reports,
        dataset_dir=dataset_dir,
        docs_dir=docs_dir,
        rubrics_dir=rubrics_dir,
    )

    if report.partial_matches:
        console.print(
            "\n[yellow]⚠ Dataset drift: some eval IDs in these runs are no "
            "longer in the dataset. Cards for missing IDs will render "
            "without a Test case section:[/yellow]"
        )
        for slug, matched_count, total_count in report.partial_matches:
            console.print(
                f"  [yellow]•[/yellow] {slug}: "
                f"[bold]{matched_count}/{total_count}[/bold] prompts found"
            )

    if report.unmatched_copied:
        console.print(
            "\n[yellow]⚠ Zero dataset matches — falling back to the verbatim "
            "showcase/<slug>/report.html, which may be stale:[/yellow]"
        )
        for slug in report.unmatched_copied:
            console.print(f"  [yellow]•[/yellow] {slug}")

    if report.unmatched_missing:
        console.print(
            "\n[red]✗ Zero dataset matches AND no showcase report.html to fall "
            "back to — no per-run report published for:[/red]"
        )
        for slug in report.unmatched_missing:
            console.print(f"  [red]•[/red] {slug}")

    console.print(f"\n[green]Static site written to {output}/[/green]")
    console.print(f"[green]Preview: [/green]python -m http.server --directory {output}")
    console.print(
        f"[green]Deploy:  [/green]rsync -avz {output}/ user@host:public_html/"
    )


if __name__ == "__main__":
    app()
