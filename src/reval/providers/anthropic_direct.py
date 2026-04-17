"""Anthropic API provider — direct async access via the official SDK."""

from __future__ import annotations

import asyncio
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

# Delays (seconds) between successive retries when the Anthropic API returns 429.
# Three retries → total wait of up to 420 s before surfacing RateLimitError.
# NOTE: reval-collector/collector/agents/base.py has its own retry layer
# (MAX_RATE_LIMIT_RETRIES=3, delays 1/2/4 s). If both layers exhaust, total
# elapsed time can reach ~21 min. Tune together if you change either constant.
_RETRY_DELAYS = (60.0, 120.0, 240.0)

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
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                response = await self._client.messages.create(**kwargs)
                break
            except anthropic.RateLimitError as exc:
                if delay is None:
                    raise RateLimitError(str(exc)) from exc
                logger.warning(
                    "rate-limited on %s (attempt %d/%d) — retrying in %.0fs",
                    self.model_id,
                    attempt + 1,
                    len(_RETRY_DELAYS),
                    delay,
                )
                await asyncio.sleep(delay)

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
