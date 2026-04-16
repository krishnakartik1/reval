---
title: Run your first eval
order: 2
description: End-to-end walkthrough of `reval run` with real flags and outputs.
---

Once your [install](install.md) is working, this page walks through
the simplest possible full benchmark run. You'll pick a model, run a
slice of the dataset against it, and explore the HTML report.

## Pick a model

REVAL uses friendly CLI handles defined in `evals/config.yaml`. Every
handle maps to a `{provider, model_id}` pair. A non-exhaustive sample:

| Handle                | Provider  | Use case                     |
|-----------------------|-----------|------------------------------|
| `claude-haiku-3-5`    | bedrock   | Cheap default target         |
| `claude-sonnet-4`     | anthropic | Higher-capability target     |
| `gpt-4o-mini`         | openai    | OpenAI reference             |
| `nova-lite`           | bedrock   | Default judge                |
| `titan-v2`            | bedrock   | Default embeddings           |
| `gemma4-e2b-local`    | ollama    | Fully local, no cloud credentials |

Any entry in the catalog can also be used as the judge or embeddings
backend via `--judge-model` / `--embeddings-model`. Roles are
determined by the flag, not by where the entry sits in the YAML.

## Run a small slice

A full run sends up to ~200 provider calls (54 evals × up to 3 calls
each for target, judge, and embeddings). Start with a filtered slice to
keep costs manageable:

```bash
reval run --model claude-haiku-3-5 --country us --category issue_framing
```

What the flags do:

- `--model` names the *target* model — the system under test.
- `--country` filters the dataset by country. Omit to run both.
- `--category` filters by eval category. Omit to run all five.
- `--judge-model` (optional) overrides the scoring judge. Defaults
  to `nova-lite` from `evals/config.yaml`.
- `--embeddings-model` (optional) overrides the embedding backend
  used for similarity-based scoring. Defaults to `titan-v2`.

Every run creates a timestamped directory under `results/`:

```text
results/claude-haiku-3-5_2026-04-14T12-00-00Z/
├── results.json    # full structured run data
├── report.html     # interactive dashboard (sortable table, charts)
└── report.md       # GitHub-renderable summary
```

## Read the report

Open `report.html` in a browser. The dashboard has four panels:

1. **Overall score** — weighted mean across all completed evals.
2. **Per-category breakdown** — scores for each of the five eval
   categories, with bands color-coded via the thresholds in
   `evals/config.yaml` (see [Config reference](../reference/config.md)).
3. **Result cards** — one per eval, expandable to show the prompt,
   the model's response, the judge's reasoning (for rubric-scored
   categories), and the per-criterion rubric breakdown.
4. **Metadata footer** — git SHA, judge model, embeddings model,
   and timestamp so runs are reproducible.

`report.md` is a leaner Markdown version of the same data — useful
for pasting into PRs or issues.

`results.json` is the machine-readable source of truth: every field
in the HTML and Markdown reports is derived from it.

## View past runs in the leaderboard

REVAL ships a static leaderboard site that aggregates results across
every run in `showcase/`. To preview it locally:

```bash
# Copy a completed run into showcase/ to make it visible to the
# leaderboard build.
cp -r results/claude-haiku-3-5_2026-04-14T12-00-00Z showcase/

# Rebuild the static site into public/
reval leaderboard build

# Serve public/ on a local HTTP server
python -m http.server --directory public 8000
```

Open `http://localhost:8000` — you'll see the leaderboard table
with your new run, and `http://localhost:8000/docs` is this tab.
See [Viewing reports](viewing-reports.md) for a deeper walkthrough
of the HTML report anatomy.
