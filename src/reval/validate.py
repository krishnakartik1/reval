"""Validate REVAL dataset entries against the JSON schema."""

import json
from pathlib import Path

import jsonschema
import jsonlines
from rich.console import Console
from rich.table import Table

console = Console()


def load_schema(schema_path: Path) -> dict:
    """Load the JSON schema."""
    return json.loads(schema_path.read_text())


def validate_entry(entry: dict, schema: dict, entry_id: str) -> list[str]:
    """Validate a single entry against the schema.

    Returns list of error messages (empty if valid).
    """
    errors = []

    try:
        jsonschema.validate(entry, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    # Additional semantic validations
    category = entry.get("category")

    if category == "policy_attribution":
        pair = entry.get("counterfactual_pair", {})
        if pair.get("entity_a") == pair.get("entity_b"):
            errors.append("counterfactual_pair: entity_a and entity_b should differ")

    if category == "factual_accuracy":
        gt = entry.get("ground_truth", {})
        level = gt.get("level")
        citations = gt.get("citations", [])
        if level in (1, 2) and not citations:
            errors.append(
                f"ground_truth: Level {level} claims should have citations"
            )

    if category == "argumentation_parity":
        if entry.get("position_a") == entry.get("position_b"):
            errors.append("position_a and position_b should be different positions")

    # Check ID format
    eval_id = entry.get("id", "")
    if not eval_id:
        errors.append("Missing id field")
    else:
        parts = eval_id.split("-")
        if len(parts) < 3:
            errors.append(
                f"ID format should be: <country>-<category>-<number> (got: {eval_id})"
            )

    return errors


def validate_file(file_path: Path, schema: dict) -> tuple[int, int, list[tuple[str, list[str]]]]:
    """Validate all entries in a JSONL file.

    Returns: (valid_count, invalid_count, list of (entry_id, errors))
    """
    valid = 0
    invalid = 0
    all_errors = []

    try:
        with jsonlines.open(file_path) as reader:
            for i, entry in enumerate(reader):
                entry_id = entry.get("id", f"entry_{i}")
                errors = validate_entry(entry, schema, entry_id)

                if errors:
                    invalid += 1
                    all_errors.append((entry_id, errors))
                else:
                    valid += 1
    except Exception as e:
        all_errors.append((str(file_path), [f"File error: {e}"]))
        invalid += 1

    return valid, invalid, all_errors


def validate_dataset(
    dataset_dir: Path,
    schema_path: Path,
    verbose: bool = False,
) -> bool:
    """Validate entire dataset directory.

    Returns True if all entries are valid.
    """
    schema = load_schema(schema_path)

    total_valid = 0
    total_invalid = 0
    file_results = []

    # Find all JSONL files
    jsonl_files = list(dataset_dir.rglob("*.jsonl"))

    if not jsonl_files:
        console.print(f"[yellow]No JSONL files found in {dataset_dir}[/yellow]")
        return True

    console.print(f"\nValidating {len(jsonl_files)} file(s)...\n")

    for file_path in jsonl_files:
        valid, invalid, errors = validate_file(file_path, schema)
        total_valid += valid
        total_invalid += invalid
        file_results.append((file_path, valid, invalid, errors))

        if verbose or invalid > 0:
            rel_path = file_path.relative_to(dataset_dir)
            if invalid > 0:
                console.print(f"[red]✗[/red] {rel_path}: {valid} valid, {invalid} invalid")
                for entry_id, entry_errors in errors:
                    console.print(f"  [red]{entry_id}:[/red]")
                    for err in entry_errors:
                        console.print(f"    - {err}")
            elif verbose:
                console.print(f"[green]✓[/green] {rel_path}: {valid} valid")

    # Summary table
    console.print()
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")

    table.add_row("Total entries", str(total_valid + total_invalid))
    table.add_row("Valid", str(total_valid))
    table.add_row("Invalid", str(total_invalid) if total_invalid == 0 else f"[red]{total_invalid}[/red]")
    table.add_row("Files checked", str(len(jsonl_files)))

    console.print(table)

    return total_invalid == 0
