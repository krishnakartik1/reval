"""Amazon Bedrock async LLM provider.

Wraps `aioboto3.Session().client('bedrock-runtime')` behind the
`reval.contracts.LLMProvider` interface. Provider-format dispatch for
Anthropic/Nova/Meta/Titan lives in `reval.utils.bedrock` as pure stdlib
helpers and is imported here and by the scoring judges unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import ClassVar

import aioboto3

from reval.contracts.provider import (
    CompletionResult,
    LLMProvider,
    RateLimitError,
)
from reval.utils.bedrock import build_request_body, parse_response_text

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """LLMProvider backed by `aioboto3`'s `invoke_model` Bedrock Runtime API.

    The `region` and `session` kwargs exist for testability — callers may
    inject a mock `aioboto3.Session` or share one across `BedrockProvider`,
    `BedrockJudge`, and `BedrockEmbeddings` to reuse IAM context.
    """

    provider_name: ClassVar[str] = "bedrock"

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        session: aioboto3.Session | None = None,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._session = session or aioboto3.Session()

    async def acomplete(
        self,
        system: str | None,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        request_body = build_request_body(
            self.model_id,
            user,
            system_prompt=system,
            max_tokens=max_tokens,
        )
        start = time.perf_counter()
        try:
            async with self._session.client(
                "bedrock-runtime", region_name=self.region
            ) as client:
                response = await client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(await response["body"].read())
        except Exception as exc:  # noqa: BLE001 — re-raised below
            if _is_throttle(exc):
                raise RateLimitError(str(exc)) from exc
            raise

        text = parse_response_text(self.model_id, response_body)
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = response_body.get("usage") or {}
        return CompletionResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens") or usage.get("inputTokens"),
            output_tokens=usage.get("output_tokens") or usage.get("outputTokens"),
        )


def _is_throttle(exc: BaseException) -> bool:
    """Heuristic: treat Bedrock ThrottlingException as a rate-limit signal."""
    name = type(exc).__name__
    if name in {"ThrottlingException", "TooManyRequestsException"}:
        return True
    msg = str(exc).lower()
    return "throttling" in msg or "too many requests" in msg
