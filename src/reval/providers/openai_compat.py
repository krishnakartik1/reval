"""OpenAI-compatible async provider.

Works with the OpenAI API and any OpenAI-compatible backend that accepts
`base_url` override (Together AI, Groq, OpenRouter, Fireworks, ...).
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import openai

from reval.contracts.provider import (
    CompletionResult,
    LLMProvider,
    RateLimitError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    """Async OpenAI Chat Completions provider with optional `base_url`.

    `provider_name = "openai"` covers both native OpenAI and
    OpenAI-compatible endpoints (Together, Groq, etc.). The underlying
    vendor is carried by `model_id`.
    """

    provider_name: ClassVar[str] = "openai"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client or openai.AsyncOpenAI(
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=messages,
            )
        except openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc

        text = response.choices[0].message.content or ""
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = response.usage
        return CompletionResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
