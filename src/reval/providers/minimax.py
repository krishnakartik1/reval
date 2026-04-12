"""MiniMax M2.7 provider via the Anthropic-compatible endpoint.

MiniMax exposes an Anthropic Messages-compatible API at
`api.minimax.io/anthropic`, so we reuse the async Anthropic SDK with
`base_url` pointed at MiniMax. This keeps the provider implementation
tiny while still giving collector's generator and reval's runner the
same async contract.
"""

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

ANTHROPIC_COMPAT_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_MODEL = "MiniMax-M2.7"


class MinimaxProvider(LLMProvider):
    """MiniMax via the Anthropic-compatible Messages API."""

    provider_name: ClassVar[str] = "minimax"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = ANTHROPIC_COMPAT_BASE_URL,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

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

        # MiniMax M2.7 returns thinking blocks alongside text; keep only text.
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
