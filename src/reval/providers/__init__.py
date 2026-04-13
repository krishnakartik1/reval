"""Concrete LLMProvider implementations for reval.

Providers live here (not in `reval.contracts`) because they pull in
concrete HTTP client libraries (`aioboto3`, `anthropic`, `openai`).
The contracts namespace must stay zero-dep.
"""

from reval.providers.anthropic_direct import AnthropicProvider
from reval.providers.bedrock import BedrockProvider
from reval.providers.factory import provider_from_config
from reval.providers.minimax import MinimaxProvider
from reval.providers.ollama import OllamaProvider
from reval.providers.openai_compat import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "BedrockProvider",
    "MinimaxProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "provider_from_config",
]
