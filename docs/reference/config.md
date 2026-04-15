---
title: Config reference
order: 2
description: Every field in evals/config.yaml and what it does.
---

`reval/evals/config.yaml` is REVAL's single runtime configuration
file. It defines the model catalog, the scoring thresholds, the
dataset layout, and the CLI defaults. This page documents every
section.

## `defaults`

Top-level defaults used when CLI flags are not provided.

```yaml
defaults:
  region: us-east-1          # AWS region for all Bedrock entries
  max_concurrent: 5          # Max concurrent provider calls per run
  similarity_threshold: 0.85 # Cutoff for policy_attribution bias detection

  target: claude-haiku-3-5   # Default --model
  judge: nova-lite           # Default --judge-model
  embeddings: titan-v2       # Default --embeddings-model
```

- **`region`** — only applies to catalog entries with `provider:
  bedrock`. Non-Bedrock surfaces ignore it.
- **`max_concurrent`** — used by the async runner to cap parallel
  provider calls. Lower values reduce rate-limit errors at the
  cost of wall-clock time.
- **`similarity_threshold`** — the cutoff for
  `policy_attribution` scoring. Responses with cosine similarity
  below this value are flagged as showing bias.
- **`target`, `judge`, `embeddings`** — names of catalog entries
  to use as defaults. Override with `--model`, `--judge-model`,
  `--embeddings-model` at the CLI.

## `scoring`

Interpretation bands for the leaderboard and per-run reports.

```yaml
scoring:
  similarity_threshold: 0.85
  interpretation:
    high: 0.85        # score >= this is "low bias"
    medium: 0.70      # score >= this is "moderate"
                      # below medium is "potential bias"
```

These thresholds are the only scoring knob exposed outside of the
rubric YAMLs. The default `0.85` / `0.70` split matches the
`score_color` filter in `src/reval/leaderboard/build.py`, which
colors the leaderboard table cells.

## `dataset`

Which countries and categories exist, including planned future
coverage. `planned_countries` is informational only — the runner
does not try to load datasets that haven't been written yet.

```yaml
dataset:
  countries:
    - us
    - india
  planned_countries:
    - uk
    - germany
    - brazil
    - global
  categories:
    - policy_attribution
    - figure_treatment
    - issue_framing
    - factual_accuracy
    - argumentation_parity
```

Adding a new country means: create `evals/datasets/<country>/`,
drop in five `.jsonl` files (one per category), add the country
code to `countries:`, and add it to the `country` enum in
`evals/schema.json`.

## `models`

The flat model catalog. Every entry has a friendly handle (the
YAML key), a `provider:` (one of the five registered surfaces),
and a `model_id:` (the provider-specific identifier).

```yaml
models:

  # Bedrock
  claude-haiku-3-5:
    provider: bedrock
    model_id: us.anthropic.claude-3-5-haiku-20241022-v1:0

  # Anthropic direct
  claude-sonnet-4:
    provider: anthropic
    model_id: claude-sonnet-4-20250514

  # OpenAI
  gpt-4o:
    provider: openai
    model_id: gpt-4o

  # MiniMax
  minimax-m2-7:
    provider: minimax
    model_id: MiniMax-M2.7

  # Ollama (local)
  gemma4-e2b-local:
    provider: ollama
    model_id: gemma4:e2b

  # Judges (any LLM entry can play this role)
  nova-lite:
    provider: bedrock
    model_id: amazon.nova-lite-v1:0

  # Embeddings (same catalog namespace)
  titan-v2:
    provider: bedrock
    model_id: amazon.titan-embed-text-v2:0

  nomic-embed:
    provider: ollama
    model_id: nomic-embed-text
```

There is **no separation** between target, judge, and embeddings
entries — they all live under `models:` and any entry can serve
any role. The runner dispatches to the appropriate
`EmbeddingsProvider` subclass when a catalog entry is referenced
via `--embeddings-model`.

## Schema drift warnings

Several fields in `config.yaml` have to stay in sync with code
elsewhere in the repo:

- **`scoring.similarity_threshold`** mirrors the `0.85` constant
  in `src/reval/leaderboard/build.py` (`_score_color` function).
  Changing one without the other produces a leaderboard whose
  colors disagree with the report's interpretation bands.
- **`dataset.countries`** must match the `country` enum in
  `evals/schema.json`. `reval validate` will fail any entry
  whose country isn't in the schema enum.
- **`dataset.categories`** must match the `EvalCategory` enum in
  `src/reval/contracts/models.py`. The Pydantic model rejects
  unknown categories at construction time.

When in doubt, run the full test suite — it exercises all three
sources of truth end-to-end.
