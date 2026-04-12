"""Tests for `reval.providers.MinimaxProvider`.

MinimaxProvider is a thin wrapper around `anthropic.AsyncAnthropic`
with `base_url` overridden to MiniMax's Anthropic-compatible endpoint.
The test shape mirrors `test_provider_anthropic.py` — we mock the
client and verify the returned `CompletionResult`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reval.contracts.provider import CompletionResult
from reval.providers.minimax import MinimaxProvider


def _mock_client_with_text(text: str) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        )
    )
    return client


@pytest.mark.asyncio
async def test_acomplete_returns_completion_result() -> None:
    client = _mock_client_with_text("hello from minimax")
    provider = MinimaxProvider(model_id="MiniMax-M2.7", client=client)

    result = await provider.acomplete(system="be helpful", user="Say hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from minimax"
    assert result.input_tokens == 8
    assert result.output_tokens == 4


@pytest.mark.asyncio
async def test_acomplete_strips_thinking_blocks() -> None:
    """MiniMax M2.7 returns a thinking block before the text block.

    The provider should keep only the text block, dropping the thinking
    content — that is the whole reason this code exists.
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", text="internal reasoning..."),
                SimpleNamespace(type="text", text="final answer"),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
    )
    provider = MinimaxProvider(model_id="MiniMax-M2.7", client=client)

    result = await provider.acomplete(system=None, user="explain")

    assert result.text == "final answer"
    assert "reasoning" not in result.text


def test_provider_name_is_minimax() -> None:
    assert MinimaxProvider.provider_name == "minimax"
