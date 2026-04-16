"""Tests for `reval.providers.OpenAIProvider`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from reval.contracts.provider import CompletionResult, RateLimitError
from reval.providers.openai_compat import OpenAIProvider


def _mock_chat_response(text: str, prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            ),
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _mock_client(response) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_acomplete_returns_completion_result() -> None:
    client = _mock_client(_mock_chat_response("hello from gpt", 11, 3))
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    result = await provider.acomplete(system="be helpful", user="Say hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from gpt"
    assert result.input_tokens == 11
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_acomplete_includes_system_message_when_set() -> None:
    client = _mock_client(_mock_chat_response("ok", 5, 1))
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    await provider.acomplete(system="be terse", user="hi")

    call = client.chat.completions.create.call_args
    messages = call.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_acomplete_skips_system_message_when_none() -> None:
    client = _mock_client(_mock_chat_response("ok", 5, 1))
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    await provider.acomplete(system=None, user="hi")

    call = client.chat.completions.create.call_args
    messages = call.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_acomplete_handles_none_content() -> None:
    """OpenAI returns `content=None` on some edge cases (tool calls etc.)."""
    client = _mock_client(
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=0),
        )
    )
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    result = await provider.acomplete(system=None, user="hi")

    assert result.text == ""


@pytest.mark.asyncio
async def test_acomplete_reraises_rate_limit() -> None:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError(
            "rate limited",
            response=MagicMock(status_code=429, request=MagicMock()),
            body=None,
        )
    )
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    with pytest.raises(RateLimitError):
        await provider.acomplete(system=None, user="hi")


def test_provider_name_is_openai() -> None:
    assert OpenAIProvider.provider_name == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["gpt-5", "gpt-5.4", "gpt-5.4-mini"])
async def test_acomplete_uses_max_completion_tokens_for_gpt5(model_id: str) -> None:
    """GPT-5 family rejects max_tokens; we must send max_completion_tokens."""
    client = _mock_client(_mock_chat_response("ok", 5, 1))
    provider = OpenAIProvider(model_id=model_id, client=client)

    await provider.acomplete(system=None, user="hi", max_tokens=512)

    call = client.chat.completions.create.call_args
    assert "max_completion_tokens" in call.kwargs
    assert call.kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in call.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"])
async def test_acomplete_uses_max_completion_tokens_for_o_series(
    model_id: str,
) -> None:
    """o-series reasoning models also require max_completion_tokens."""
    client = _mock_client(_mock_chat_response("ok", 5, 1))
    provider = OpenAIProvider(model_id=model_id, client=client)

    await provider.acomplete(system=None, user="hi", max_tokens=1024)

    call = client.chat.completions.create.call_args
    assert "max_completion_tokens" in call.kwargs
    assert call.kwargs["max_completion_tokens"] == 1024
    assert "max_tokens" not in call.kwargs


@pytest.mark.asyncio
async def test_acomplete_uses_max_tokens_for_legacy_models() -> None:
    """Legacy models (gpt-4o, gpt-3.5-turbo) still use max_tokens."""
    client = _mock_client(_mock_chat_response("ok", 5, 1))
    provider = OpenAIProvider(model_id="gpt-4o", client=client)

    await provider.acomplete(system=None, user="hi", max_tokens=2048)

    call = client.chat.completions.create.call_args
    assert "max_tokens" in call.kwargs
    assert call.kwargs["max_tokens"] == 2048
    assert "max_completion_tokens" not in call.kwargs
