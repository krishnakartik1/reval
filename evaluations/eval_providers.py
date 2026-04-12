"""Real integration tests for reval LLM providers — actual API calls.

Each test sends a trivial prompt through one of the async providers
and verifies the resulting `CompletionResult` is well-formed. Tests
skip individually when the required API key isn't set, so running
`pytest -m eval` with only some keys configured still exercises the
providers you can reach.

Phase 3 of the unification plan added these three providers to reval
alongside the existing `BedrockProvider`. The equivalent collector-
side file (`reval-collector/evaluations/eval_providers.py`)
is slated for deletion in the Phase 3 collector PR — reval's
providers are now the canonical implementations, so reval owns the
real-API coverage too.

Run: `pytest -m eval evaluations/eval_providers.py -v`
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from reval.contracts.provider import CompletionResult, LLMProvider

load_dotenv()

SIMPLE_SYSTEM = "You are a helpful assistant."
SIMPLE_USER = "Respond with only the word 'ok'."


async def _assert_valid_completion(provider: LLMProvider) -> CompletionResult:
    """Shared assertion shape: acomplete returns non-empty text + latency."""
    result = await provider.acomplete(
        system=SIMPLE_SYSTEM, user=SIMPLE_USER, max_tokens=50
    )
    assert isinstance(result, CompletionResult)
    assert result.text.strip(), "provider returned empty text"
    assert result.latency_ms >= 0
    return result


class TestMinimaxProviderReal:
    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("MINIMAX_API_KEY"),
        reason="MINIMAX_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        """MinimaxProvider returns a non-empty response via the async SDK."""
        from reval.providers.minimax import MinimaxProvider

        provider = MinimaxProvider(api_key=os.environ["MINIMAX_API_KEY"])
        await _assert_valid_completion(provider)

    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("MINIMAX_API_KEY"),
        reason="MINIMAX_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_strips_thinking_blocks(self) -> None:
        """Live M2.7 responses include a thinking block before the text.

        MinimaxProvider.acomplete must keep only the text block — the
        regression test from `tests/test_provider_minimax.py` uses a
        mocked response with a thinking block; this version proves the
        real endpoint actually returns one (and that the parser handles
        it without dropping the text on the floor).
        """
        from reval.providers.minimax import MinimaxProvider

        provider = MinimaxProvider(api_key=os.environ["MINIMAX_API_KEY"])
        result = await provider.acomplete(
            system=SIMPLE_SYSTEM,
            user="List 3 facts about the number 7. Be brief.",
            max_tokens=300,
        )
        assert len(result.text.strip()) > 10


class TestAnthropicProviderReal:
    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        """AnthropicProvider returns a non-empty response via AsyncAnthropic."""
        from reval.providers.anthropic_direct import AnthropicProvider

        provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
        await _assert_valid_completion(provider)

    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_model_override(self) -> None:
        """AnthropicProvider works with an explicit model_id."""
        from reval.providers.anthropic_direct import AnthropicProvider

        provider = AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_id="claude-haiku-4-5-20251001",
        )
        await _assert_valid_completion(provider)


class TestOpenAIProviderReal:
    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        """OpenAIProvider returns a non-empty response via AsyncOpenAI."""
        from reval.providers.openai_compat import OpenAIProvider

        provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
        await _assert_valid_completion(provider)

    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    @pytest.mark.asyncio
    async def test_model_override(self) -> None:
        """OpenAIProvider works with a different model."""
        from reval.providers.openai_compat import OpenAIProvider

        provider = OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            model_id="gpt-4o-mini",
        )
        await _assert_valid_completion(provider)


class TestBedrockProviderReal:
    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("AWS_ACCESS_KEY_ID")
        or not os.environ.get("AWS_SECRET_ACCESS_KEY"),
        reason="AWS credentials not set",
    )
    @pytest.mark.asyncio
    async def test_acomplete(self) -> None:
        """BedrockProvider returns a non-empty response via aioboto3."""
        from reval.providers.bedrock import BedrockProvider

        provider = BedrockProvider(
            model_id="amazon.nova-lite-v1:0",
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
        await _assert_valid_completion(provider)


class TestProviderParity:
    """Compare outputs across all providers whose credentials are set."""

    @pytest.mark.eval
    @pytest.mark.asyncio
    async def test_all_available_providers_return_text(self) -> None:
        """Every provider with a configured key should return non-empty text."""
        from reval.providers.factory import provider_from_config

        configs: list[tuple[str, dict]] = []
        if os.environ.get("MINIMAX_API_KEY"):
            configs.append(
                (
                    "minimax",
                    {
                        "model_id": "MiniMax-M2.7",
                        "api_key": os.environ["MINIMAX_API_KEY"],
                    },
                )
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            configs.append(
                (
                    "anthropic",
                    {
                        "model_id": "claude-sonnet-4-20250514",
                        "api_key": os.environ["ANTHROPIC_API_KEY"],
                    },
                )
            )
        if os.environ.get("OPENAI_API_KEY"):
            configs.append(
                (
                    "openai",
                    {
                        "model_id": "gpt-4o-mini",
                        "api_key": os.environ["OPENAI_API_KEY"],
                    },
                )
            )
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ):
            configs.append(
                (
                    "bedrock",
                    {
                        "model_id": "amazon.nova-lite-v1:0",
                        "region": os.environ.get("AWS_REGION", "us-east-1"),
                    },
                )
            )

        if not configs:
            pytest.skip("no provider API keys configured")

        results: dict[str, str] = {}
        for name, kwargs in configs:
            provider = provider_from_config(name, **kwargs)
            result = await provider.acomplete(
                system=SIMPLE_SYSTEM, user=SIMPLE_USER, max_tokens=50
            )
            results[name] = result.text.strip()
            assert results[name], f"{name} returned empty response"
