# AWS AI Ideas Submission - REVAL

## Big Idea (Elevator Pitch)

REVAL is a benchmark for measuring political bias in LLMs that distinguishes between bias and factual accuracy. Instead of penalizing models for stating verified facts, we use a ground truth taxonomy to evaluate whether models are genuinely biased or simply accurate, with initial focus on US and India political discourse.

Character count: 328


## Vision - What Will You Build

REVAL is an open-source evaluation framework with an initial dataset of 150-200 evaluations across core categories: Policy Attribution tests whether models judge identical policies differently based on political labels, Figure Treatment assesses whether politicians are described with comparable tone, and Factual Accuracy checks whether models match verified claims.

The framework includes counterfactual pair testing using semantic similarity scoring, LLM-as-judge evaluation with structured rubrics, a ground truth database linking claims to sources, and an async Python runner for efficient evaluation. Initial coverage focuses on US political discourse with secondary coverage of India, establishing a foundation that can expand to additional countries.

Character count: 763


## How Will It Make a Difference

Who benefits: AI developers needing to measure bias before deployment, researchers studying LLM alignment, and organizations evaluating AI for civic applications.

Problems solved: Current bias benchmarks treat balanced as equidistant from all positions, which penalizes models for accurately representing scientific consensus or verified facts. REVAL separates genuine bias from accuracy. By including non-US political content, we surface biases that English-centric benchmarks miss entirely.

Opportunities created: A reusable framework for bias evaluation that others can extend, an initial dataset demonstrating the methodology, and documentation enabling the research community to contribute additional evaluations across more countries and topics.

Character count: 768


## Game Plan

Phase 1, Foundation, Weeks 1-4: Build core infrastructure in Python. Create Pydantic data models and JSON schema for eval entries. Implement semantic similarity scoring using Amazon Bedrock embeddings. Build basic LLM-as-judge evaluation using Claude via Bedrock. Develop async runner for parallel evaluation.

Phase 2, Initial Dataset, Weeks 5-9: Create 100 US-focused evaluations covering policy attribution and factual accuracy categories. Create 50 India-focused evaluations on economic policy and governance. Build ground truth database with source citations for factual claims. Validate all entries against schema.

Phase 3, Validation and Release, Weeks 10-12: Test framework on multiple models including Claude and open-source alternatives. Document methodology and limitations clearly. Open-source the framework and dataset. Write up findings and publish initial benchmark results.

Future expansion: The framework is designed to scale to 500+ evaluations and additional countries. Community contributions can extend coverage to UK, Germany, Brazil, and other regions.

Character count: 1148


## AWS AI Services

Amazon Bedrock


## AWS Free Tier Services

Amazon S3, Amazon DynamoDB, AWS Lambda, Amazon CloudWatch


---

# Notes for submission

These responses are written without special characters, em dashes, or parentheses as required.

Key changes from original draft:
- Reduced eval count from 500 to 150-200 for hackathon scope
- Focused on US and India rather than 6 countries
- Single LLM judge instead of three-model consensus
- Removed specific correlation targets
- Positioned as framework plus initial dataset rather than complete benchmark
- Added future expansion section to show larger vision without overpromising
