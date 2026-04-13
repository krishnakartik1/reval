"""Shared fixtures for REVAL integration evaluations.

These tests hit real Amazon Bedrock (and, optionally, Ollama on
localhost). They skip individually when the matching dependency
isn't available so `pytest -m eval` still exercises the providers
you can reach.
"""

import json
import os
import socket
import urllib.request

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


def _is_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """Cheap TCP probe — Ollama listens on 11434 by default."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _ollama_has_model(
    model_name: str, host: str = "localhost", port: int = 11434
) -> bool:
    """Return True if Ollama has `model_name` pulled locally.

    Queries `/api/tags` and checks the installed model list. Tolerates
    Ollama's two tagging quirks:

    1. **Default `:latest` tag.** `ollama pull nomic-embed-text` installs
       the model as `nomic-embed-text:latest`. A query for `nomic-embed-text`
       should match.
    2. **Quantization suffixes.** `gemma4:e2b` may appear literally or as
       `gemma4:e2b-q4_0`. A query for `gemma4:e2b` should match either.

    The match rules:
    - Exact match wins.
    - If the query has no explicit tag (`":"` not in `model_name`),
      also match `name == f"{model_name}:latest"`.
    - Always also match `name.startswith(f"{model_name}-")` to cover the
      quantization-suffix case, which Ollama uses for tagged variants.
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            data = json.load(response)
    except Exception:  # noqa: BLE001
        return False
    installed = {m.get("name", "") for m in data.get("models", [])}
    if model_name in installed:
        return True
    if ":" not in model_name and f"{model_name}:latest" in installed:
        return True
    return any(name.startswith(f"{model_name}-") for name in installed)


@pytest.fixture(scope="session")
def aws_region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


@pytest.fixture(scope="session")
def bedrock_available(aws_region: str) -> bool:
    """Skip eval tests if AWS credentials are not available."""
    if not _has_aws_creds():
        pytest.skip("No AWS credentials found — skipping Bedrock evals")
    return True


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Skip eval tests if Ollama isn't running on localhost:11434."""
    if not _is_ollama_running():
        pytest.skip("Ollama not running on localhost:11434 — skipping Ollama evals")
    return True


@pytest.fixture
def require_ollama_model():
    """Return a `(name) -> None` helper that skips the test if the model isn't pulled.

    Usage inside an eval test:

        def test_ollama_thing(self, ollama_available, require_ollama_model):
            require_ollama_model("gemma4:e2b")
            # ... use the model ...

    Ollama doesn't auto-pull models at inference time, so a missing
    model surfaces as a 404 from the server mid-test. This fixture
    converts that into a clean skip with an actionable message telling
    the user which `ollama pull` to run.
    """

    def _require(model_name: str) -> None:
        if not _ollama_has_model(model_name):
            pytest.skip(
                f"Ollama model {model_name!r} not pulled; "
                f"run `ollama pull {model_name}` to enable this eval"
            )

    return _require


@pytest.fixture
def eval_runner(bedrock_available, aws_region):
    """Build an EvalRunner pointed at the evals/rubrics directory.

    Uses the all-Bedrock path (target + judge + embeddings) so the
    existing benchmark_run / factual_accuracy / figure_treatment evals
    keep exercising Bedrock exactly like before. A parallel
    fully-local-Ollama fixture would need its own eval file.
    """
    from pathlib import Path

    from reval.providers.factory import provider_from_config
    from reval.runner import EvalRunner
    from reval.scoring.judge import LLMJudge
    from reval.scoring.parity import LLMParityJudge
    from reval.utils.embeddings import BedrockEmbeddingsProvider

    # Small, fast model for eval assertions — users can override via env
    model_id = os.environ.get("REVAL_EVAL_MODEL", "amazon.nova-lite-v1:0")
    rubrics_dir = Path(__file__).parent.parent / "evals" / "rubrics"

    target_provider = provider_from_config(
        "bedrock", model_id=model_id, region=aws_region
    )
    # Judge + parity share one Bedrock provider; parity_judge is just
    # a thin wrapper around the same acomplete() call path.
    judge_provider = provider_from_config(
        "bedrock", model_id="amazon.nova-lite-v1:0", region=aws_region
    )
    return EvalRunner(
        provider=target_provider,
        judge=LLMJudge(provider=judge_provider),
        parity_judge=LLMParityJudge(provider=judge_provider),
        embeddings=BedrockEmbeddingsProvider(region=aws_region),
        rubrics_dir=rubrics_dir,
        max_concurrent=2,
    )
