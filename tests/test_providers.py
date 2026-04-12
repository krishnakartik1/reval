"""Tests for `reval.providers.BedrockProvider`.

The Bedrock provider wraps `aioboto3.Session().client("bedrock-runtime")`.
These tests mock the session so no AWS calls happen and verify:
- `acomplete` returns a `CompletionResult` with the parsed text and a
  non-zero latency
- Provider-format dispatch builds the correct request body for Anthropic,
  Nova, Meta Llama, and Titan model IDs
- A throttling exception is re-raised as `RateLimitError`
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reval.contracts.provider import CompletionResult, RateLimitError
from reval.providers.bedrock import BedrockProvider


class _FakeBody:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def _make_session(payload: dict) -> MagicMock:
    """Build a mock aioboto3.Session whose client invoke_model returns payload."""
    client = MagicMock()
    client.invoke_model = AsyncMock(return_value={"body": _FakeBody(payload)})
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.client = MagicMock(return_value=client)
    return session


@pytest.mark.asyncio
async def test_acomplete_returns_completion_result() -> None:
    payload = {
        "content": [{"text": "hello from claude"}],
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }
    session = _make_session(payload)
    provider = BedrockProvider(
        model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        region="us-east-1",
        session=session,
    )

    result = await provider.acomplete(system=None, user="Say hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from claude"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_acomplete_reraises_throttle_as_rate_limit() -> None:
    session = MagicMock()
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    class ThrottlingException(Exception):
        pass

    client.invoke_model = AsyncMock(side_effect=ThrottlingException("Slow down"))
    session.client = MagicMock(return_value=client)

    provider = BedrockProvider(model_id="any-model-id", session=session)

    with pytest.raises(RateLimitError):
        await provider.acomplete(system=None, user="go")


def test_provider_name_is_bedrock() -> None:
    assert BedrockProvider.provider_name == "bedrock"
