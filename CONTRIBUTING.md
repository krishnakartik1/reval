# Contributing to REVAL

Thanks for your interest in contributing! Here's how to get started.

## Ways to Contribute

- **Dataset expansion** — new evals for underrepresented countries or topics
- **Rubric refinement** — improved scoring criteria for figure treatment / issue framing
- **Ground truth curation** — verified facts with authoritative sources
- **Bug fixes and code improvements**

## Getting Started

```bash
git clone https://github.com/krishnakartik1/reval
cd reval
pip install -e ".[dev]"
```

## Adding Evals

1. Follow the schema in `evals/schema.json`
2. Add your `.jsonl` file to the appropriate `evals/datasets/{country}/` directory
3. Validate before submitting:

```bash
reval validate --dataset evals/datasets/
```

4. Open a pull request with a brief description of what the evals cover and why they're valuable

## Code Changes

- Keep changes focused — one concern per PR
- **Every new module or public class must ship tests in the same PR.** There is no "add tests later" — the review bar is that code and its tests land together.
- **Coverage floor is 85%.** CI runs `pytest tests/ --cov=reval --cov-fail-under=85`; PRs that drop coverage below the floor are blocked.
- Before pushing, run:
  ```bash
  ruff check .
  black --check .
  pytest tests/                  # unit tests only — no network, fast
  ```
  All three must pass.

### Which test suite to run

Reval has **two** test trees, configured via `testpaths = ["tests", "evaluations"]` in `pyproject.toml`:

- `tests/` — **unit tests, mocked providers, no network.** This is the default suite for iteration and pre-push gating. Always invoke as `pytest tests/`.
- `evaluations/` — **live-API integration tests** marked `@pytest.mark.eval`. These hit real Bedrock/Anthropic/OpenAI/MiniMax endpoints and cost money to run. They auto-skip when credentials are missing, but you still pay collection + probe overhead on every invocation. **Run these only when you deliberately want a live-eval pass**, with `pytest -m eval evaluations/ -v`.

**Do not run bare `pytest`** during normal development — because `testpaths` includes both trees, bare `pytest` walks `evaluations/` too and triggers the live-eval collection path. Always qualify the path (`pytest tests/`) or the marker (`pytest -m "not eval"`).

## Questions

Open a [GitHub Discussion](https://github.com/krishnakartik1/reval/discussions) for anything that doesn't fit as an issue.
