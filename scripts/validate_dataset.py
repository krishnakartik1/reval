#!/usr/bin/env python3
"""Standalone script to validate REVAL dataset entries.

This script can be run directly without installing the package.
For installed usage, use: reval validate
"""

import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from reval.validate import validate_dataset

console = Console()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate REVAL dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/datasets"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("evals/schema.json"),
        help="Path to JSON schema",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all results, not just errors",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root if needed
    project_root = Path(__file__).parent.parent
    dataset_path = (
        args.dataset if args.dataset.is_absolute() else project_root / args.dataset
    )
    schema_path = (
        args.schema if args.schema.is_absolute() else project_root / args.schema
    )

    if not schema_path.exists():
        console.print(f"[red]Schema not found: {schema_path}[/red]")
        sys.exit(1)

    if not dataset_path.exists():
        console.print(f"[yellow]Dataset directory not found: {dataset_path}[/yellow]")
        console.print("Creating empty dataset directory...")
        dataset_path.mkdir(parents=True, exist_ok=True)

    is_valid = validate_dataset(dataset_path, schema_path, args.verbose)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
