"""Tests for `reval.providers.AnthropicProvider`.

Uses `anthropic.AsyncAnthropic` under the hood. We mock the client's
`messages.create` to avoid real API calls and verify that `acomplete`
returns a properly-constructed `CompletionResult` and re-raises
`anthropic.RateLimitError` as `reval.contracts.provider.RateLimitError`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from reval.contracts.provider import CompletionResult, RateLimitError
from reval.providers.anthropic_direct import AnthropicProvider


def _mock_message_response(text: str, input_tokens: int, output_tokens: int):
    """Build an anthropic-like Message object with .content and .usage."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _mock_client(response) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_acomplete_returns_completion_result() -> None:
    client = _mock_client(_mock_message_response("hello from claude", 11, 3))
    provider = AnthropicProvider(model_id="claude-sonnet-4", client=client)

    result = await provider.acomplete(system="be helpful", user="Say hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from claude"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_acomplete_without_system_prompt() -> None:
    client = _mock_client(_mock_message_response("ok", 5, 1))
    provider = AnthropicProvider(model_id="claude-sonnet-4", client=client)

    await provider.acomplete(system=None, user="say ok")

    # The SDK call should not include `system` when it's None.
    call = client.messages.create.call_args
    assert "system" not in call.kwargs


@pytest.mark.asyncio
async def test_acomplete_filters_non_text_blocks() -> None:
    client = _mock_client(
        SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", text="ignored"),
                SimpleNamespace(type="text", text="kept"),
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
    )
    provider = AnthropicProvider(model_id="claude-sonnet-4", client=client)

    result = await provider.acomplete(system=None, user="hi")

    assert result.text == "kept"


@pytest.mark.asyncio
async def test_acomplete_reraises_rate_limit() -> None:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(
        side_effect=anthropic.RateLimitError(
            "rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
    )
    provider = AnthropicProvider(model_id="claude-sonnet-4", client=client)

    with pytest.raises(RateLimitError):
        await provider.acomplete(system=None, user="hi")


def test_provider_name_is_anthropic() -> None:
    assert AnthropicProvider.provider_name == "anthropic"
