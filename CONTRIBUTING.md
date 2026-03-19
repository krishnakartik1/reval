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

- Run tests before submitting: `pytest`
- Keep changes focused — one concern per PR
- New functionality should include tests

## Questions

Open a [GitHub Discussion](https://github.com/krishnakartik1/reval/discussions) for anything that doesn't fit as an issue.
