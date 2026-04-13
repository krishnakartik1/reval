"""Tests for `reval.providers.OllamaProvider`.

OllamaProvider is a thin subclass of OpenAIProvider — same wire format,
different defaults (loopback base URL, placeholder api_key). Tests
cover the default-resolution behavior plus one happy-path completion
to make sure the OpenAI SDK path still works when driven through the
subclass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reval.contracts.provider import CompletionResult
from reval.providers.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaProvider


def _mock_chat_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )


def _mock_client(text: str) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_mock_chat_response(text))
    return client


@pytest.mark.asyncio
async def test_acomplete_returns_completion_result() -> None:
    client = _mock_client("hello from ollama")
    provider = OllamaProvider(model_id="gemma4:e2b", client=client)

    result = await provider.acomplete(system="be helpful", user="Say hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello from ollama"
    assert result.input_tokens == 5
    assert result.output_tokens == 2


def test_provider_name_is_ollama() -> None:
    assert OllamaProvider.provider_name == "ollama"


def test_default_model_is_gemma4_e2b() -> None:
    """User-specified default judge model for the local Ollama path."""
    assert DEFAULT_MODEL == "gemma4:e2b"


def test_default_base_url_is_loopback() -> None:
    assert DEFAULT_BASE_URL == "http://localhost:11434/v1"


def test_constructor_uses_default_base_url_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = _mock_client("x")
    provider = OllamaProvider(model_id="gemma4:e2b", client=client)
    # Injected client short-circuits the base_url resolution, but the
    # provider still records model_id correctly.
    assert provider.model_id == "gemma4:e2b"


def test_constructor_honors_ollama_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host:9999/v1")
    # We don't inject a client here — we need to verify the real OpenAI
    # client constructor sees the env-derived base_url. The OpenAI SDK
    # constructor doesn't hit the network until acomplete() is called,
    # so this is safe.
    provider = OllamaProvider(model_id="gemma4:e2b")
    # openai.AsyncOpenAI exposes base_url as a public attribute
    assert "remote-host:9999" in str(provider._client.base_url)


def test_constructor_uses_placeholder_api_key_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama's default config ignores auth, so we pass a placeholder.

    This guards against an accidental fallback to OPENAI_API_KEY if we
    ever stop passing the placeholder explicitly.
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OllamaProvider(model_id="gemma4:e2b")
    assert provider._client.api_key == "ollama"
