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
├── README.md
├── AWS_SUBMISSION.md
├── CLAUDE.md (this file)
├── src/
│   └── reval/
│       ├── __init__.py
│       ├── runner.py
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
│   ├── ground_truth/
│   └── rubrics/
├── tests/
└── scripts/
    ├── run_benchmark.py
    └── validate_dataset.py
```

## Current Status

DONE:
- README.md created with full project documentation
- AWS_SUBMISSION.md created with hackathon submission text

TODO:
- Initialize pyproject.toml
- Create JSON schema for eval validation
- Create Pydantic models
- Implement scoring functions (similarity, rubric, judge, parity)
- Build async eval runner
- Create dataset directory structure
- Create rubric templates
- Write validation scripts
- Write unit tests
- Create sample eval entries

## Commands

When implementation is done:

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
