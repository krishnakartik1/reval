"""Async embeddings interface + Bedrock and Ollama implementations.

`Embeddings` is the ABC; `BedrockEmbeddingsProvider` and
`OllamaEmbeddingsProvider` are the concrete backends. Both return numpy
arrays for drop-in compatibility with `cosine_similarity` and
`compute_semantic_similarity` — the choice of backend is transparent to
the scoring helpers.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import aioboto3
import httpx
import numpy as np


class Embeddings(ABC):
    """Async embeddings interface.

    Subclasses must implement `get_embedding(text)` returning a
    `numpy.ndarray`. The default `get_embeddings(texts)` implementation
    calls `get_embedding` in parallel via `asyncio.gather`. Backends
    that support real batching can override it.

    `provider_name` is a class-level identifier for the API surface
    (`"bedrock"`, `"ollama"`), not the vendor. `model_id` carries the
    specific model identifier.
    """

    provider_name: ClassVar[str]
    model_id: str

    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """Return the embedding for a single text as a numpy array."""

    async def get_embeddings(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Return embeddings for multiple texts in parallel.

        Default implementation fans out to `get_embedding` concurrently.
        """
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)


class BedrockEmbeddingsProvider(Embeddings):
    """Amazon Bedrock backend (Titan by default)."""

    provider_name: ClassVar[str] = "bedrock"

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        session: aioboto3.Session | None = None,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self._session = session or aioboto3.Session()

    async def get_embedding(self, text: str) -> np.ndarray:
        async with self._session.client(
            "bedrock-runtime", region_name=self.region
        ) as client:
            response = await client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(await response["body"].read())
            return np.array(response_body["embedding"])


class OllamaEmbeddingsProvider(Embeddings):
    """Ollama backend — hits `/api/embeddings` on localhost:11434.

    Ollama's embeddings endpoint is one-at-a-time (no batch support as
    of Ollama 0.11), so `get_embeddings` parallelises via the default
    `asyncio.gather` implementation rather than a single batch call.

    Override the endpoint with `OLLAMA_BASE_URL` — supports both
    `http://host:11434` and `http://host:11434/v1` (the `/v1` suffix is
    the OpenAI-compat chat path, which is wrong for embeddings; this
    class strips it automatically so a single `OLLAMA_BASE_URL` env
    var can drive both the LLM provider and the embeddings provider).
    """

    provider_name: ClassVar[str] = "ollama"

    def __init__(
        self,
        model_id: str = "nomic-embed-text",
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_id = model_id
        raw_base = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        # Strip the OpenAI-compat `/v1` suffix if present — Ollama's
        # embeddings endpoint lives at /api/embeddings on the root.
        if raw_base.endswith("/v1"):
            raw_base = raw_base[:-3]
        self.base_url = raw_base.rstrip("/")
        self._client = client

    async def get_embedding(self, text: str) -> np.ndarray:
        payload = {"model": self.model_id, "prompt": text}
        if self._client is not None:
            response = await self._client.post(
                f"{self.base_url}/api/embeddings", json=payload
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings", json=payload
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])


def embeddings_from_config(
    provider_name: str,
    model_id: str,
    **kwargs: Any,
) -> Embeddings:
    """Construct an `Embeddings` backend by its `provider_name`.

    Mirror of `reval.providers.factory.provider_from_config` for the
    embeddings side. Two backends today: `bedrock` and `ollama`.
    """
    if provider_name == "bedrock":
        return BedrockEmbeddingsProvider(model_id=model_id, **kwargs)
    if provider_name == "ollama":
        # Filter out any Bedrock-only kwargs the CLI might pass
        # unconditionally (e.g. `region`).
        ollama_kwargs = {
            k: v for k, v in kwargs.items() if k in {"base_url", "client"}
        }
        return OllamaEmbeddingsProvider(model_id=model_id, **ollama_kwargs)
    raise ValueError(
        f"Unsupported embeddings provider {provider_name!r}; "
        "supported: bedrock, ollama"
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))


async def compute_semantic_similarity(
    text_a: str,
    text_b: str,
    embeddings_client: Embeddings,
) -> float:
    """Compute semantic similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.
        embeddings_client: An `Embeddings` backend — required. Callers
            (the runner, similarity scorer) inject this explicitly; there
            is no factory fallback.

    Returns:
        Similarity score between 0 and 1.
    """
    embeddings = await embeddings_client.get_embeddings([text_a, text_b])
    return cosine_similarity(embeddings[0], embeddings[1])
