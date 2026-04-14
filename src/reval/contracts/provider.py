"""Abstract LLM provider interface used by the reval runner and collector.

Concrete providers live under `reval.providers` and do import their
concrete SDKs. This module itself must not — `reval.contracts` is a
zero-dep namespace. The forbidden-imports set is enforced by
`tests/test_contracts_imports.py` and is:
`{aioboto3, boto3, numpy, jsonlines, httpx, anthropic, openai}`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


class RateLimitError(Exception):
    """Raised when a provider's backend returns a rate-limit / throttle error.

    Callers can catch this to implement back-off without having to know which
    concrete provider raised it (Bedrock `ThrottlingException`, OpenAI 429,
    Anthropic `overloaded_error`, etc.).
    """


@dataclass
class CompletionResult:
    """A provider-agnostic completion response.

    Strict superset of what reval's old `ModelClient.generate` returned
    (`tuple[str, int]`) and collector's old `LLMProvider.complete` returned
    (`str`). Both callers gain token-count telemetry for free.
    """

    text: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    """Async completion interface.

    `provider_name` is a class-level identifier for the **API surface**, not
    the model vendor. `BedrockProvider.provider_name = "bedrock"` is correct
    even when the underlying model is Claude or Llama — the same Claude model
    can be reached through both `BedrockProvider` and (in Phase 3)
    `AnthropicProvider`, and these are different surfaces with different
    throttling, auth, and pricing characteristics. `model_id` carries the
    vendor signal (`us.anthropic.claude-sonnet-4-...` vs
    `claude-sonnet-4-20250514`).
    """

    provider_name: ClassVar[str]
    model_id: str

    # Subclasses define their own `__init__` (concrete providers have their
    # own kwargs like `region`, `session`, `api_key`, `base_url`). This base
    # `__init__` exists so the provider factory can type-check as
    # `cls(model_id=..., **kwargs)` without mypy complaining. The actual
    # subclass `__init__` is what runs at instantiation time.
    def __init__(self, model_id: str, **kwargs: object) -> None:
        self.model_id = model_id

    @abstractmethod
    async def acomplete(
        self,
        system: str | None,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Run a single completion. Must raise `RateLimitError` on throttle."""
