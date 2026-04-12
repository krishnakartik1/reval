"""Concrete LLMProvider implementations for reval.

Providers live here (not in `reval.contracts`) because they pull in
concrete HTTP client libraries (`aioboto3`, and in Phase 3 `httpx`,
`anthropic`, `openai`). The contracts namespace must stay zero-dep.
"""

from reval.providers.bedrock import BedrockProvider
from reval.providers.factory import provider_from_config

__all__ = ["BedrockProvider", "provider_from_config"]
