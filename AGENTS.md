# AGENTS.md — Development Rules for AI Coding Agents

This file is loaded by Claude Code and other AI coding agents when working in this repo. It captures the project's invariants and conventions so agents don't drift from them.

## What is REVAL

REVAL (Robust Evaluation of Values and Alignment in LLMs) is a fact-aligned benchmark for political and ideological bias in LLMs, with international coverage (US, India today; UK, Germany, Brazil, Global planned).

**Key differentiator:** unlike symmetry-based benchmarks, REVAL scores against ground truth. A model stating "vaccines are safe" is rewarded for accuracy, not penalized as "biased".

## Eval Categories

1. **policy_attribution** — counterfactual prompt pairs, semantic similarity scoring
2. **figure_treatment** — paired prompts about opposing political figures, LLM judge with rubric, `treatment_parity` as the headline score
3. **issue_framing** — single neutral prompt, LLM judge with rubric
4. **factual_accuracy** — ground truth match via embeddings, plus a `counterfactual_prompt` probe for `framing_consistency`
5. **argumentation_parity** — parity judge compares argumentative effort on opposing positions

## Ground Truth Taxonomy

- **Level 1** — Empirical facts (match verified data)
- **Level 2** — Expert consensus (represent accurately)
- **Level 3** — Contested empirical (present evidence landscape fairly)
- **Level 4** — Value judgments (balance perspectives)

## Tech Stack

- Python 3.10+ with `asyncio`
- Pydantic v2 for all data models — no raw dicts crossing module boundaries
- Amazon Bedrock for target model calls, embeddings (Titan), and LLM judge (Nova)
- `aioboto3` for async Bedrock calls
- JSONL for dataset storage (one eval per line)
- Typer + Rich for CLI

## Data Model Invariants

These are enforced by the `EvalEntry` `model_validator` and must not be weakened:

- `policy_attribution` entries require `counterfactual_pair`
- `figure_treatment` entries require `figure_pair` and `rubric_id` (NOT a single `prompt`)
- `factual_accuracy` entries require `prompt`, `ground_truth`, AND `counterfactual_prompt`
- `issue_framing` entries require `prompt` and `rubric_id`
- `argumentation_parity` entries require `position_a` and `position_b`

The JSON schema at `evals/schema.json` mirrors these rules in `allOf` conditionals. Keep the Pydantic validators and the JSON schema in sync.

## Runner Invariants

- `_run_factual_accuracy` MUST call the counterfactual prompt and compute `counterfactual_similarity` + `framing_consistency`. Do not skip the counterfactual scoring.
- `_run_judge_eval` for `figure_treatment` MUST score both figures independently and set `treatment_parity = 1.0 - abs(score_a - score_b)`. The top-level `score` on a figure_treatment result IS the treatment parity.
- All Bedrock calls go through `reval.utils.bedrock.build_request_body` + `parse_response_text` so provider differences (Anthropic / Nova / Meta / Titan) are handled in one place.

## Branch & PR Workflow

1. **Never commit directly to `main` or `master`.**
2. Create a branch: `git checkout -b feat/your-feature-name`
3. Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
4. Push: `git push -u origin feat/your-feature-name`
5. Open PR via `gh pr create`
6. Krishna reviews and merges — never merge your own PR unless explicitly told.

## Python Best Practices

- Use `pyproject.toml` (NOT `setup.py` or `setup.cfg`)
- `ruff` for linting, `black` for formatting, line length 100
- `pathlib.Path` — never `os.path`
- `logging` — never `print()` in production code
- Type hints on every function signature
- Docstrings on public functions and classes
- `__init__.py` in every package directory
- No wildcard imports, no global mutable state

## Testing — Hard Requirements

Two tiers, kept strictly separate:

1. **`tests/`** — unit tests with every external call mocked. Run in CI on every PR.
   - `pytest tests/ --cov=reval --cov-fail-under=85`
2. **`evaluations/`** — integration tests (`@pytest.mark.eval`) that hit real Amazon Bedrock. Auto-skip when AWS credentials are unavailable.
   - `pytest -m eval evaluations/ -v`

Rules:
- **Every new class or module added must have corresponding tests in the same PR.**
- Mock ALL external API calls in `tests/` — no real network calls.
- Live-API tests belong in `evaluations/`, never in `tests/`.
- **Minimum 85% coverage required** on the `reval` package.
- When touching the runner's scoring paths, add an integration eval under `evaluations/` that verifies the new field is populated end-to-end.

## Linting & Formatting — Hard Gates

Before declaring any task complete:
```bash
ruff check src/ tests/ evaluations/
black --check src/ tests/ evaluations/
pytest tests/ --cov=reval --cov-fail-under=85 --tb=short
```

## Project Structure

```
reval/
├── pyproject.toml
├── README.md
├── AGENTS.md (this file)
├── src/
│   └── reval/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── runner.py
│       ├── report.py
│       ├── validate.py
│       ├── models/
│       │   └── eval.py              # Pydantic: EvalEntry, EvalResult, FigurePair, CounterfactualPair, ...
│       ├── scoring/
│       │   ├── similarity.py        # policy_attribution
│       │   ├── rubric.py
│       │   ├── judge.py             # figure_treatment, issue_framing
│       │   └── parity.py            # argumentation_parity
│       └── utils/
│           ├── bedrock.py           # provider-aware request/response helpers
│           └── embeddings.py        # Titan embeddings + cosine similarity
├── evals/
│   ├── schema.json                  # JSON Schema (kept in sync with Pydantic validators)
│   ├── config.yaml                  # model IDs, thresholds, defaults
│   ├── datasets/
│   │   ├── us/
│   │   └── india/
│   └── rubrics/
│       ├── figure_treatment.yaml
│       └── issue_framing.yaml
├── tests/                           # unit tests (mocked Bedrock)
├── evaluations/                     # live-Bedrock integration tests (@pytest.mark.eval)
│   ├── conftest.py                  # auto-skip when no AWS creds; eval_runner fixture
│   ├── eval_factual_accuracy.py
│   ├── eval_figure_treatment.py
│   └── eval_benchmark_run.py
└── scripts/
    ├── validate_dataset.py
    └── run_mock_benchmark.py
```

## Commands

```bash
# Validate dataset against schema
reval validate --dataset evals/datasets/

# Run a benchmark
reval run --model amazon.nova-pro-v1:0
reval run --model claude-haiku-3-5 --country us --category policy_attribution

# List available evals
reval list-evals --country india

# Unit tests
pytest tests/ --cov=reval --cov-fail-under=85

# Live-Bedrock evals (requires AWS credentials)
pytest -m eval evaluations/ -v
```

## Important Design Decisions

1. **Paired figure treatment, not single-prompt.** A `figure_treatment` entry contains a `FigurePair` with two prompts about opposing figures. The runner scores each prompt independently with the same rubric and computes `treatment_parity = 1.0 - abs(score_a - score_b)`. 1.0 = perfectly equal treatment, 0.0 = maximally biased.
2. **Counterfactual prompts are required for factual_accuracy.** Every `factual_accuracy` entry has a `counterfactual_prompt` (the same fact asked differently). The runner scores both and reports `framing_consistency` — the similarity between the two responses — so we can detect models that flip their answer based on phrasing.
3. **Counterfactual pairs use 0.85 semantic similarity threshold** for policy_attribution bias detection.
4. **LLM judge requires structured rubrics** (YAML under `evals/rubrics/`) for reproducibility. Judge model is configurable via `evals/config.yaml`.
5. **All eval entries validated against the JSON schema** before inclusion. The schema and the Pydantic `model_validator` are kept in sync.
6. **Dataset is JSONL** for easy version control, diffing, and line-by-line append semantics.
7. **Async execution** with a semaphore for parallel Bedrock calls bounded by `max_concurrent`.

## Owner

Krishna Kartik — krishnakartik1@gmail.com
