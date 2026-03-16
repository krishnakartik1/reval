# REVAL Project Context

## What is this?

REVAL (Robust Evaluation of Values and Alignment in LLMs) is a benchmark for measuring political bias in LLMs. It's being submitted to the AWS $10,000 AI Ideas hackathon.

## Key Differentiator

Unlike existing bias benchmarks that treat "balanced" as equidistant from all positions, REVAL uses fact-aligned scoring. A model stating "vaccines are safe" isn't penalized for bias - it's rewarded for accuracy.

## Hackathon Scope (Conservative)

- 150-200 evaluations total
- Focus on US and India only
- Single LLM judge via Amazon Bedrock
- Framework + initial dataset, not a complete benchmark

## Long-term Vision

- 500+ evaluations
- 6 countries: US, India, UK, Germany, Brazil, Global
- Multi-model judge consensus
- Public leaderboard

## Eval Categories

1. Policy Attribution - counterfactual pairs, semantic similarity scoring
2. Figure Treatment - LLM judge with rubric
3. Issue Framing - LLM judge
4. Factual Accuracy - ground truth matching
5. Argumentation Parity - effort comparison metrics

## Ground Truth Taxonomy

- Level 1: Empirical facts (match verified data)
- Level 2: Expert consensus (represent accurately)
- Level 3: Contested empirical (present evidence landscape fairly)
- Level 4: Value judgments (balance perspectives)

## Tech Stack

- Python 3.10+ with asyncio
- Pydantic v2 for data models
- Amazon Bedrock for embeddings and LLM judge
- JSONL for dataset storage
- S3/DynamoDB for AWS deployment

## Target File Structure

```
reval/
├── pyproject.toml
├── requirements.txt
├── README.md
├── AWS_SUBMISSION.md
├── CLAUDE.md (this file)
├── src/
│   └── reval/
│       ├── __init__.py
│       ├── cli.py
│       ├── runner.py
│       ├── validate.py
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── similarity.py
│       │   ├── rubric.py
│       │   ├── judge.py
│       │   └── parity.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── eval.py
│       └── utils/
│           ├── __init__.py
│           └── embeddings.py
├── evals/
│   ├── schema.json
│   ├── config.yaml
│   ├── datasets/
│   │   ├── us/
│   │   └── india/
│   └── rubrics/
├── tests/           (not yet created)
└── scripts/
    └── validate_dataset.py
```

## Current Status

DONE:
- README.md created with full project documentation
- AWS_SUBMISSION.md created with hackathon submission text
- pyproject.toml and requirements.txt initialized
- JSON schema for eval validation (evals/schema.json)
- Pydantic models (src/reval/models/eval.py)
- Scoring functions: similarity, rubric, judge, parity
- Async eval runner (src/reval/runner.py)
- CLI entrypoint (src/reval/cli.py)
- Dataset validation (src/reval/validate.py, scripts/validate_dataset.py)
- Dataset directory structure with sample eval entries for US and India
- Rubric templates (evals/rubrics/)
- Eval config (evals/config.yaml)

TODO:
- Write unit tests (tests/ directory not yet created)
- Create scripts/run_benchmark.py
- Create evals/ground_truth/ directory and content

## Commands

```bash
# Validate dataset
reval validate --dataset evals/

# Run benchmark
reval run --model claude-3-opus --output results/

# Run specific category
reval run --model gpt-4 --category policy_attribution
```

## Important Design Decisions

1. Counterfactual pairs use 0.85 semantic similarity threshold
2. LLM judge requires structured rubrics for reproducibility
3. All eval entries validated against JSON schema before inclusion
4. Dataset is JSONL for easy version control and diffing
5. Async execution for parallel API calls
