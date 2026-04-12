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

app = typer.Typer(
    name="reval",
    help="REVAL - Robust Evaluation of Values and Alignment in LLMs",
)
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
    from reval.scoring.judge import BedrockJudge
    from reval.scoring.parity import ParityJudge
    from reval.utils.embeddings import BedrockEmbeddings

    config = load_config(config_path)
    provider_name, model = resolve_model(model, config)
    judge_model_id = judge_model or config.judge_model_id
    embeddings_model_id = embeddings_model or config.embeddings_model_id

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

    # Build the provider + scoring clients. `provider_name` comes from the
    # YAML entry's `provider:` field (bedrock | anthropic | openai | minimax).
    # Bedrock-only kwargs (like `region`) flow to BedrockProvider; non-Bedrock
    # providers ignore them.
    provider_kwargs: dict = {}
    if provider_name == "bedrock":
        provider_kwargs["region"] = region
    provider = provider_from_config(provider_name, model_id=model, **provider_kwargs)
    judge = BedrockJudge(model_id=judge_model_id, region=region)
    parity_judge = ParityJudge(model_id=judge_model_id, region=region)
    embeddings = BedrockEmbeddings(model_id=embeddings_model_id, region=region)

    runner = EvalRunner(
        provider=provider,
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

    run_dir = save_run_outputs(benchmark_run, output)
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


if __name__ == "__main__":
    app()
