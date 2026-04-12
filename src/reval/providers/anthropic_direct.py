"""Anthropic API provider — direct async access via the official SDK."""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import anthropic

from reval.contracts.provider import (
    CompletionResult,
    LLMProvider,
    RateLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(LLMProvider):
    """Async Anthropic Messages API provider.

    Phase 3 added this provider so reval benchmarks can target Claude
    directly (not via Bedrock). `provider_name = "anthropic"` identifies
    the API surface — the same Claude model reached via
    `BedrockProvider(provider_name="bedrock")` is a different surface
    with different throttling, auth, and pricing characteristics.
    """

    provider_name: ClassVar[str] = "anthropic"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def acomplete(
        self,
        system: str | None,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        start = time.perf_counter()
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc

        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return CompletionResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=getattr(response.usage, "input_tokens", None),
            output_tokens=getattr(response.usage, "output_tokens", None),
        )
