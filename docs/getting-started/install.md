---
title: Install
order: 1
description: Clone, install, and sanity-check your reval setup.
---

REVAL targets **Python 3.10 or newer**. You'll need a working `git`
and credentials for at least one LLM provider — Bedrock, Anthropic,
OpenAI, MiniMax, or Ollama. Each provider is optional; you only need
keys for the surfaces you actually plan to use.

## Clone and install

```bash
git clone https://github.com/krishnakartik1/reval
cd reval
pip install -e ".[dev]"
cp .env.example .env
```

The `[dev]` extra pulls in `pytest`, `ruff`, `black`, `mypy`, and
`pre-commit` — needed if you want to run the test suite locally or
contribute changes. If you want to build the static leaderboard docs tab
locally, also install the `[docs]` extra:

```bash
pip install -e ".[dev,docs]"
```

`[docs]` adds `markdown-it-py`, `mdit-py-plugins`, and `pygments` —
only required when you run `reval leaderboard build` against a
populated `docs/` directory.

## Provider credentials

Open `.env` and fill in keys only for the providers you plan to use:

```bash
# AWS Bedrock (default for judge + embeddings)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Anthropic direct API
ANTHROPIC_API_KEY=...

# OpenAI (or OpenAI-compatible endpoints)
OPENAI_API_KEY=...

# MiniMax
MINIMAX_API_KEY=...

# Ollama runs locally — no keys needed, just `ollama serve` on 11434.
```

The LLM judge and embeddings default to **Amazon Bedrock** (`nova-lite`
and `titan-v2` in `evals/config.yaml`). If you don't have AWS
credentials, point the judge and embeddings at any other registered
model via the `--judge-model` and `--embeddings-model` flags when you
run the benchmark.

## Verify your install

```bash
reval --help
reval list-evals --country us --category issue_framing
```

The first command prints the top-level CLI. The second enumerates
evals in the shipped dataset — if it returns rows without crashing,
your Python environment is wired up correctly. At this point you
haven't spent any API credit; `list-evals` only reads local files.

To verify credentials end-to-end, run a filtered slice of one
category:

```bash
reval run --model claude-haiku-3-5 --country us --category issue_framing
```

This runs a single category of evals end-to-end. If it exits cleanly,
your credentials and network are working.

## Next

- [Run your first eval](first-eval.md) for a full walkthrough of
  the `reval run` flags and what each output file contains.
- [Methodology](../concepts/methodology.md) explains what REVAL is
  actually measuring and why the scoring approach is novel.
