---
title: CLI reference
order: 3
description: Every reval subcommand with its flags and what it does.
---

The `reval` CLI is built on
[Typer](https://typer.tiangolo.com/). Every command supports
`--help` for inline documentation.

## Top-level commands

```bash
reval --help
```

- `reval run` — execute a benchmark run against a target model.
- `reval list-evals` — enumerate eval entries from the dataset.
- `reval validate` — validate `.jsonl` files against the JSON schema.
- `reval leaderboard build` — regenerate the static leaderboard site
  (including this docs tab).

## `reval run`

The primary command. Runs every eval in the filtered dataset against
the target model, scores the responses, and writes the results to
`results/<run>/`.

```bash
reval run --model claude-haiku-3-5 \
          --country us \
          --category issue_framing \
          --judge-model nova-pro \
          --embeddings-model titan-v2
```

Flags:

- **`--model`** (required) — Catalog handle of the target model.
  See [Providers & models](providers.md) for the list.
- **`--country`** — Filter by country (`us`, `india`). Omit to run
  both.
- **`--category`** — Filter by eval category
  (`policy_attribution`, `figure_treatment`, `issue_framing`,
  `factual_accuracy`, `argumentation_parity`). Omit to run all five.
- **`--judge-model`** — Override the scoring judge. Defaults to
  `nova-lite` from `evals/config.yaml`.
- **`--embeddings-model`** — Override the embeddings backend.
  Defaults to `titan-v2`.
- **`--limit N`** — Cap the number of evals run. Useful for smoke
  tests.
- **`--output-dir`** — Override the default `results/<run>/`
  destination.

Every run writes three files per entry:
`results.json`, `report.html`, `report.md`. See
[Viewing reports](../getting-started/viewing-reports.md) for what
each file contains.

## `reval list-evals`

Enumerates the shipped dataset without running anything. Doesn't hit
any LLM. Useful for sanity-checking filters.

```bash
reval list-evals
reval list-evals --country india
reval list-evals --category figure_treatment
reval list-evals --country india --category issue_framing
```

Output is a Rich-formatted table with `id`, `category`, `country`,
and `topic` columns.

## `reval validate`

Runs every `.jsonl` file under `--dataset` against `--schema`.
Exit code 0 on success, non-zero on any validation failure. Used
by CI to catch schema drift:

```bash
reval validate --dataset evals/datasets/ --schema evals/schema.json
reval validate --dataset evals/datasets/ --verbose
```

`--verbose` prints every successfully-validated entry ID in
addition to the failure summary.

## `reval leaderboard build`

Regenerates the static site under `public/`. Walks every
directory in `--showcase` looking for `results.json` files,
renders the leaderboard table, and (when `--docs` is supplied and
exists) the Docs tab.

```bash
reval leaderboard build
reval leaderboard build --showcase showcase --output public
reval leaderboard build --no-include-reports   # skip per-run reports
reval leaderboard build --docs /tmp/nonexistent # skip docs tab
```

Flags:

- **`--showcase/-s`** — Directory of per-run subdirectories. Default:
  `showcase/`.
- **`--output/-o`** — Destination directory. Default: `public/`.
- **`--include-reports/--no-include-reports`** — Generate per-run
  `report.html` files into `public/reports/`. Default: on.
- **`--dataset/-d`** — Dataset directory used to regenerate per-run
  reports against the current prompts. Default: `evals/datasets/`.
  Pass a non-existent path to fall back to copying
  `showcase/<slug>/report.html` verbatim (useful when the dataset
  has drifted and you want the historical prompts preserved).
- **`--docs`** — Path to the `docs/` directory containing markdown
  source for the Docs tab. Default: `docs/` in the reval repo root.
  Pass a non-existent path to skip the docs build entirely. On
  wheel installs (no `docs/` in the repo) the default path won't
  exist and the docs build is silently skipped.

**Note**: there is no `--no-docs` bool toggle. Typer rejects two
options that share the long name `--docs`, so the docs flag is a
path-only flag that you skip by pointing at a non-existent path.

## Exit codes

All commands follow standard UNIX exit code conventions:

- **0** — Success.
- **1** — Validation failure, missing file, or runtime error.
- **2** — Typer argument parse error (wrong flag, missing required arg).

Non-zero exits also print a Rich-formatted error message to stderr.
