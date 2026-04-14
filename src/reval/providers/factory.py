"""Provider factory — resolves a provider-name string to a concrete LLMProvider.

Registered providers: `bedrock`, `anthropic`, `minimax`, `openai`, `ollama`.
"""

from __future__ import annotations

from typing import Any

from reval.contracts.provider import LLMProvider
from reval.providers.anthropic_direct import AnthropicProvider
from reval.providers.bedrock import BedrockProvider
from reval.providers.minimax import MinimaxProvider
from reval.providers.ollama import OllamaProvider
from reval.providers.openai_compat import OpenAIProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "bedrock": BedrockProvider,
    "anthropic": AnthropicProvider,
    "minimax": MinimaxProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def provider_from_config(
    provider_name: str,
    model_id: str,
    **kwargs: Any,
) -> LLMProvider:
    """Construct an LLMProvider by its `provider_name` string.

    Args:
        provider_name: One of the registered provider identifiers
            (`"bedrock"`, `"anthropic"`, `"minimax"`, `"openai"`,
            `"ollama"`). Matches `LLMProvider.provider_name`.
        model_id: Model identifier passed through to the concrete provider.
        **kwargs: Additional provider-specific kwargs (e.g. `region` for
            Bedrock). Forwarded verbatim.

    Raises:
        ValueError: If `provider_name` is not registered.
    """
    try:
        cls = _REGISTRY[provider_name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown provider {provider_name!r}; registered: {known}"
        ) from exc
    return cls(model_id=model_id, **kwargs)
