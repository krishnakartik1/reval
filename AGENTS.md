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
- **Pydantic v2** for all data models — no raw dicts crossing module boundaries. Shared contracts live in `reval.contracts` (zero-dep namespace — no aioboto3 / numpy / jsonlines / httpx / anthropic / openai imports).
- **Async-first `LLMProvider` ABC** (`reval.contracts.provider`) with four concrete implementations under `reval.providers`:
  - `BedrockProvider` — AWS Bedrock via `aioboto3` (Anthropic / Nova / Meta / Titan format dispatch)
  - `AnthropicProvider` — Anthropic Messages API via `anthropic.AsyncAnthropic`
  - `OpenAIProvider` — OpenAI-compatible chat completions via `openai.AsyncOpenAI` (supports `base_url` override for Together / Groq / OpenRouter / Fireworks)
  - `MinimaxProvider` — MiniMax M2.7 via the Anthropic-compatible endpoint
- Amazon Bedrock for embeddings (Titan) and LLM judge (Nova, default) — these stay Bedrock-specific; the provider abstraction is for the system-under-test model, not the judge or embeddings.
- JSONL for dataset storage (one eval per line), JSON Schema validation.
- **Typer + Rich** for CLI. `reval.contracts` is the shared import surface for both reval and `reval-collector`.

## Data Model Invariants

These are enforced by the `EvalEntry` `model_validator` in `reval.contracts.models` and must not be weakened:

- `policy_attribution` entries require `counterfactual_pair`
- `figure_treatment` entries require `figure_pair` and `rubric_id` (NOT a single `prompt`)
- `factual_accuracy` entries require `prompt`, `ground_truth`, AND `counterfactual_prompt`
- `issue_framing` entries require `prompt` and `rubric_id`
- `argumentation_parity` entries require `position_a` and `position_b`

The JSON schema at `evals/schema.json` mirrors these rules in `allOf` conditionals. Keep the Pydantic validators and the JSON schema in sync.

**`reval.contracts` zero-dep rule.** The contracts namespace is meant to be portable — `reval-collector` depends on it without pulling in `aioboto3` or any HTTP client library. `tests/test_contracts_imports.py` enforces this via a subprocess-based guard that fails if `import reval.contracts` transitively loads any of `{aioboto3, boto3, numpy, jsonlines, httpx, anthropic, openai}`.

## Run Reproducibility — `RunManifestMixin`

`BenchmarkRun` inherits from `reval.contracts.manifest.RunManifestMixin`, which carries the reproducibility fields shared across reval and collector: `run_id`, `timestamp`, `git_sha`, `model_provider`, `model_id`, `stage_timings`, `error_count`. `get_git_sha()` in the same module wraps `git describe --always --dirty --abbrev=12` and is the canonical way to capture repo state — the `-dirty` suffix is the guardrail against runs made on uncommitted working trees. Collector's `GenerationRunManifest` inherits the same mixin and adds `reval_version` as the cross-repo provenance hook.

## Runner Invariants

- `_run_factual_accuracy` MUST call the counterfactual prompt and compute `counterfactual_similarity` + `framing_consistency`. Do not skip the counterfactual scoring.
- `_run_judge_eval` for `figure_treatment` MUST score both figures independently and set `treatment_parity = 1.0 - abs(score_a - score_b)`. The top-level `score` on a figure_treatment result IS the treatment parity.
- `EvalRunner` takes a pre-built `LLMProvider`, `BedrockJudge`, `ParityJudge`, and `BedrockEmbeddings` via dependency injection — no `model_id` overload, no factory fallback inside the constructor. CLI builds all four explicitly via `provider_from_config` + direct construction.
- Bedrock format-helper calls (`reval.utils.bedrock.build_request_body` / `parse_response_text`) are used by `BedrockProvider`, `BedrockJudge`, and `ParityJudge` — provider differences (Anthropic / Nova / Meta / Titan) for the Bedrock surface stay in that one module.
- `BenchmarkRun(...)` must be constructed with all mixin fields populated at the start of `run_benchmark` — Pydantic validates up front, so a missing `git_sha` / `model_provider` surfaces immediately, not as a surprise mid-run.

## Branch & PR Workflow

1. **Never commit directly to `main` or `master`.**
2. Create a branch: `git checkout -b feat/your-feature-name`
3. Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
4. Push: `git push -u origin feat/your-feature-name`
5. Open PR via `gh pr create`
6. Krishna reviews and merges — never merge your own PR unless explicitly told.

## Python Best Practices

- Use `pyproject.toml` (NOT `setup.py` or `setup.cfg`)
- `ruff` for linting, `black` for formatting, **line length 88** (Phase 0a of the unification plan standardized on 88 to match collector and the global CLAUDE.md rule)
- `ruff` config uses the `[tool.ruff.lint]` table (post-0.6) and ignores `E501` — line length is black's job; E501 only nags about long string literals.
- `pathlib.Path` — never `os.path`
- `logging` — never `print()` in production code
- Type hints on every function signature
- Docstrings on public functions and classes
- `__init__.py` in every package directory
- No wildcard imports, no global mutable state
- `python-dotenv` loaded at `src/reval/cli.py` import time so provider SDKs see `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `MINIMAX_API_KEY` / `AWS_*` from `.env` automatically.

## Testing — Hard Requirements

Two tiers, kept strictly separate:

1. **`tests/`** — unit tests with every external call mocked. Run in CI on every PR via `.github/workflows/test.yml`.
   - `pytest tests/ --cov=reval --cov-fail-under=85`
2. **`evaluations/`** — integration tests (`@pytest.mark.eval`) that hit real Amazon Bedrock, Anthropic, OpenAI, and MiniMax endpoints. Each test skips individually via `@pytest.mark.skipif` when the matching API key isn't set. Run in CI via `.github/workflows/evals.yml` when the `run-evals` label is applied to a PR.
   - `pytest -m eval evaluations/ -v`

Rules:
- **Every new class or module added must have corresponding tests in the same PR.**
- Mock ALL external API calls in `tests/` — no real network calls. Use `AsyncMock` on `provider.acomplete` returning a `CompletionResult`; don't monkey-patch SDK modules.
- Live-API tests belong in `evaluations/`, never in `tests/`.
- **Minimum 85% coverage required** on the `reval` package.
- When touching the runner's scoring paths, add an integration eval under `evaluations/` that verifies the new field is populated end-to-end.
- **Any change to LLM-driven behavior requires a live eval in the same PR, not just a unit test.** This includes prompt templates, new fields the LLM is asked to fill, parse/validation logic that consumes LLM output, sampling parameters passed through `LLMProvider.acomplete`, and new fields on `CompletionResult`. Unit tests with mocked responses prove the parser handles a known payload; the eval proves a real model actually produces that payload under your new prompt. Ship both in the same PR.
- `reval.contracts` must stay zero-dep (enforced by `tests/test_contracts_imports.py`). Do not import aioboto3 / numpy / httpx / anthropic / openai from anything reachable through `reval.contracts`.

## Linting & Formatting — Hard Gates

Pre-commit runs all of these automatically. Before declaring any task complete:

```bash
pre-commit run --all-files                                          # ruff + ruff-format + black + mypy + file hooks
pytest tests/ --cov=reval --cov-fail-under=85 --tb=short            # unit tests with coverage floor
python -c "import reval.contracts; import sys; assert not {'aioboto3','boto3','numpy','jsonlines','httpx','anthropic','openai'} & set(sys.modules)"  # zero-dep contracts guard
```

If you're adding a new provider or touching `reval.contracts`, also run:

```bash
pytest tests/test_contracts_imports.py -v   # the guard in CI form
pytest tests/test_provider_*.py -v          # all provider unit tests
```

Reval's CI (`.github/workflows/test.yml`) runs pre-commit + pytest-with-coverage on every PR. The `evals.yml` workflow is label-triggered (`run-evals`) and runs `pytest -m eval evaluations/ -v` against real Bedrock + the configured provider keys.

## Project Structure

```
reval/
├── pyproject.toml                      # line-length 88, ruff.lint table, mypy plugin
├── .pre-commit-config.yaml             # ruff + ruff-format + black + mypy + file hooks
├── .github/workflows/
│   ├── test.yml                        # fast gate: pre-commit + pytest --cov-fail-under=85
│   └── evals.yml                       # slow gate: label-triggered, needs AWS secrets
├── .env.example                        # AWS + per-provider key placeholders
├── README.md
├── AGENTS.md                           # this file
├── src/
│   └── reval/
│       ├── __init__.py                 # re-exports from reval.contracts + NullHandler logger
│       ├── cli.py                      # Typer CLI: run, validate, list-evals, info
│       ├── config.py                   # load_config + resolve_model / resolve_model_provider
│       ├── runner.py                   # EvalRunner (provider-injection)
│       ├── report.py                   # HTML + Markdown report generators
│       ├── validate.py                 # JSONL dataset validator vs schema.json
│       ├── contracts/                  # zero-dep shared namespace
│       │   ├── __init__.py             # re-exports
│       │   ├── models.py               # EvalEntry, EvalResult, BenchmarkRun (inherits RunManifestMixin), …
│       │   ├── provider.py             # LLMProvider ABC, CompletionResult, RateLimitError
│       │   └── manifest.py             # RunManifestMixin + get_git_sha() helper
│       ├── providers/                  # concrete async LLMProvider implementations
│       │   ├── __init__.py
│       │   ├── bedrock.py              # BedrockProvider (aioboto3)
│       │   ├── anthropic_direct.py     # AnthropicProvider (anthropic.AsyncAnthropic)
│       │   ├── openai_compat.py        # OpenAIProvider (openai.AsyncOpenAI + base_url)
│       │   ├── minimax.py              # MinimaxProvider (Anthropic-compat endpoint)
│       │   └── factory.py              # provider_from_config(provider_name, model_id, **kwargs)
│       ├── scoring/
│       │   ├── similarity.py           # policy_attribution (Titan embeddings)
│       │   ├── rubric.py               # rubric loader + weighted score
│       │   ├── judge.py                # BedrockJudge (figure_treatment, issue_framing)
│       │   └── parity.py               # ParityJudge (argumentation_parity)
│       └── utils/
│           ├── bedrock.py              # pure format helpers — no aioboto3 import
│           └── embeddings.py           # BedrockEmbeddings + cosine similarity
├── evals/
│   ├── schema.json                     # JSON Schema (kept in sync with Pydantic validators)
│   ├── config.yaml                     # models catalog: `provider:` = API surface, not vendor
│   ├── datasets/
│   │   ├── us/
│   │   └── india/
│   └── rubrics/
│       ├── figure_treatment.yaml
│       └── issue_framing.yaml
├── tests/                              # unit tests (mocked HTTP/AWS, 85% coverage floor)
│   ├── test_contracts_imports.py       # subprocess-based zero-dep guard
│   ├── test_manifest_helpers.py        # get_git_sha() branches
│   ├── test_providers.py               # BedrockProvider mocked
│   ├── test_provider_anthropic.py      # AnthropicProvider mocked
│   ├── test_provider_openai.py         # OpenAIProvider mocked
│   ├── test_provider_minimax.py        # MinimaxProvider mocked
│   └── fixtures/                       # make_benchmark_run() factory for mixin sentinels
├── evaluations/                        # @pytest.mark.eval (real API calls, label-triggered in CI)
│   ├── conftest.py                     # auto-skip when no AWS creds; eval_runner fixture
│   ├── eval_factual_accuracy.py
│   ├── eval_figure_treatment.py
│   ├── eval_benchmark_run.py
│   └── eval_providers.py               # live-API tests for all 4 providers
└── scripts/
    └── run_mock_benchmark.py           # MockProvider(LLMProvider) + monkey-patched judge
```

### Where the contracts-vs-providers boundary lives

- **`reval.contracts`** is portable — pydantic + stdlib only. Collector imports from here.
- **`reval.providers`** holds the HTTP client-backed implementations. Collector imports `provider_from_config` from here too, but the heavy SDKs (`aioboto3`, `anthropic`, `openai`) are only loaded when actually instantiating a provider.
- **`reval.utils.bedrock`** is the pure format-helper module for Anthropic/Nova/Meta/Titan request shapes on Bedrock. `BedrockProvider`, `BedrockJudge`, and `ParityJudge` all import from it.

## Commands

```bash
# Validate dataset against schema (works from any cwd — editable-install anchored)
reval validate --dataset evals/datasets/

# Run a benchmark. The provider is resolved from evals/config.yaml's
# `provider:` field on the matching model entry.
reval run --model claude-haiku-3-5                     # bedrock
reval run --model claude-sonnet-4                      # anthropic direct
reval run --model gpt-4o                               # openai
reval run --model minimax-m2-7                         # minimax
reval run --model us.anthropic.claude-3-5-haiku-20241022-v1:0   # raw Bedrock ARN → defaults to bedrock

# Filters
reval run --model claude-haiku-3-5 --country us --category policy_attribution

# List available evals
reval list-evals --country india

# Unit tests (fast, no AWS needed)
pytest tests/ --cov=reval --cov-fail-under=85

# Live integration evals — hits real Bedrock, Anthropic, OpenAI, MiniMax
pytest -m eval evaluations/ -v
```

## Adding a new provider

1. Create `src/reval/providers/<name>.py` with a subclass of `reval.contracts.provider.LLMProvider`:
   - Set `provider_name: ClassVar[str] = "<name>"` (API surface, not vendor)
   - Implement `async def acomplete(self, system, user, *, max_tokens) -> CompletionResult`
   - Re-raise provider-native rate-limit exceptions as `reval.contracts.RateLimitError`
   - Accept an optional `client=` kwarg so tests can inject a mocked async SDK client
2. Register in `src/reval/providers/factory.py::_REGISTRY`
3. Re-export from `src/reval/providers/__init__.py::__all__`
4. Add `tests/test_provider_<name>.py` with mocked-HTTP unit tests:
   - `acomplete` happy path → well-formed `CompletionResult`
   - System prompt included/omitted based on `system=None`
   - Provider-specific edge cases (thinking-block stripping, `content=None`, etc.)
   - `RateLimitError` re-raise
   - `provider_name` ClassVar
5. Add a live-API test class to `evaluations/eval_providers.py` guarded on the required env var via `@pytest.mark.skipif`
6. Add an entry to `evals/config.yaml` with `provider: <name>` and a concrete `model_id`

## Important Design Decisions

1. **Paired figure treatment, not single-prompt.** A `figure_treatment` entry contains a `FigurePair` with two prompts about opposing figures. The runner scores each prompt independently with the same rubric and computes `treatment_parity = 1.0 - abs(score_a - score_b)`. 1.0 = perfectly equal treatment, 0.0 = maximally biased.
2. **Counterfactual prompts are required for factual_accuracy.** Every `factual_accuracy` entry has a `counterfactual_prompt` (the same fact asked differently). The runner scores both and reports `framing_consistency` — the similarity between the two responses — so we can detect models that flip their answer based on phrasing.
3. **Counterfactual pairs use 0.85 semantic similarity threshold** for policy_attribution bias detection.
4. **LLM judge requires structured rubrics** (YAML under `evals/rubrics/`) for reproducibility. Judge model is configurable via `evals/config.yaml`.
5. **All eval entries validated against the JSON schema** before inclusion. The schema and the Pydantic `model_validator` are kept in sync.
6. **Dataset is JSONL** for easy version control, diffing, and line-by-line append semantics.
7. **Async execution** with a semaphore for parallel provider calls bounded by `max_concurrent`.
8. **Provider abstraction is async-first**, with `LLMProvider.acomplete(system, user, *, max_tokens) -> CompletionResult`. Reval's runner uses it natively; collector bridges via `complete_sync` → `asyncio.run(provider.acomplete(...)).text` at its two sync call sites (see `reval-collector/collector/providers/_sync.py`).
9. **`provider_name` identifies the API surface, not the model vendor.** `BedrockProvider.provider_name = "bedrock"` even when the underlying model is Claude. The same vendor model can be reached through multiple surfaces (`claude-sonnet-4-bedrock` via Bedrock, `claude-sonnet-4` via direct Anthropic API) — `model_provider` on the run manifest disambiguates surfaces, `model_id` carries the vendor signal.
10. **Reproducibility via `git_sha` + `--dirty` suffix, not per-file content hashes.** Everything that matters for reval's reproducibility (judge prompts, schema, rubrics) lives in the reval tree, so the git SHA is the canonical capture. Cross-repo reproducibility for collector-generated data is handled by `GenerationRunManifest.reval_version` (captured via `importlib.metadata.version("reval")` at generation time).

## Owner

Krishna Kartik — krishnakartik1@gmail.com
