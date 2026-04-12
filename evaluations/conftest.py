"""Shared fixtures for REVAL integration evaluations.

These tests hit real Amazon Bedrock. They skip automatically when
AWS credentials are not available so CI can safely run `pytest -m eval`
in environments without Bedrock access.
"""

import os

import pytest


def _has_aws_creds() -> bool:
    """Return True if AWS credentials are available in the environment."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    # Fall back to checking for an AWS profile/session
    try:
        import boto3

        session = boto3.Session()
        creds = session.get_credentials()
        return creds is not None
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def aws_region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


@pytest.fixture(scope="session")
def bedrock_available(aws_region: str) -> bool:
    """Skip eval tests if AWS credentials are not available."""
    if not _has_aws_creds():
        pytest.skip("No AWS credentials found — skipping Bedrock evals")
    return True


@pytest.fixture
def eval_runner(bedrock_available, aws_region):
    """Build an EvalRunner pointed at the evals/rubrics directory."""
    from pathlib import Path

    from reval.runner import EvalRunner

    # Small, fast model for eval assertions — users can override via env
    model_id = os.environ.get("REVAL_EVAL_MODEL", "amazon.nova-lite-v1:0")
    rubrics_dir = Path(__file__).parent.parent / "evals" / "rubrics"

    return EvalRunner(
        model_id=model_id,
        rubrics_dir=rubrics_dir,
        region=aws_region,
        max_concurrent=2,
    )
