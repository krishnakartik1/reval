"""Integration-style evaluations for REVAL runner.

Unlike tests/ (which are unit tests with mocked Bedrock), these evals
make real calls to Amazon Bedrock to verify end-to-end behaviour of
the new runner code paths: counterfactual scoring for factual_accuracy
and paired figure treatment scoring.

Run with:
    pytest -m eval

Requires AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or
an AWS profile) with access to Amazon Bedrock in the configured region.
"""
