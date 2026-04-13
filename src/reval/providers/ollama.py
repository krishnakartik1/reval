"""Ollama provider via its OpenAI-compatible endpoint.

Ollama (https://ollama.com) serves local LLMs at `localhost:11434` and
ships an OpenAI-compatible chat completions endpoint at `/v1`. That
means `OllamaProvider` is just `OpenAIProvider` with different defaults
— same wire format, same parsing, same rate-limit mapping. The subclass
exists to give Ollama a distinct `provider_name` for the catalog and
factory, plus sensible defaults (loopback base URL, placeholder api_key
since Ollama's default config ignores auth).
"""

from __future__ import annotations

import os
from typing import ClassVar

import openai

from reval.providers.openai_compat import OpenAIProvider

#: Ollama's OpenAI-compatible endpoint on loopback. Override via
#: `OLLAMA_BASE_URL` if Ollama is running on another host/port.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: Default target model when no `model_id` is passed. Picked for being
#: small enough to run on a laptop (effective 2B params) while still
#: being capable enough to follow JSON-rubric instructions for judging.
DEFAULT_MODEL = "gemma4:e2b"


class OllamaProvider(OpenAIProvider):
    """LLMProvider for Ollama via the OpenAI-compatible Messages API.

    Inherits from :class:`OpenAIProvider` because the wire format is
    identical — only the `base_url` and auth defaults differ. Override
    the default endpoint via the `OLLAMA_BASE_URL` env var when running
    Ollama somewhere other than `localhost:11434`.
    """

    provider_name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        # Ollama's default config doesn't check auth, but the OpenAI SDK
        # refuses to construct a client without an api_key, so we pass
        # a placeholder. Users running Ollama behind a reverse proxy
        # that enforces auth can pass a real key.
        resolved_api_key = api_key or "ollama"
        resolved_base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        )
        super().__init__(
            model_id=model_id,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            client=client,
        )
