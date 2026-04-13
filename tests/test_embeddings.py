"""Tests for the `Embeddings` ABC and its Bedrock + Ollama backends.

The Bedrock backend instantiation is smoke-tested only — its network
path is covered in `evaluations/eval_providers.py` as a live eval.
The Ollama backend gets a richer unit test because it's new in this
refactor and its httpx-based HTTP layer is easy to mock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reval.utils.embeddings import (
    BedrockEmbeddingsProvider,
    Embeddings,
    OllamaEmbeddingsProvider,
    embeddings_from_config,
)


def _mock_httpx_response(embedding: list[float]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value={"embedding": embedding})
    return response


def _mock_httpx_client(embedding: list[float]) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=_mock_httpx_response(embedding))
    return client


class TestBedrockEmbeddingsProvider:
    def test_subclasses_embeddings(self) -> None:
        assert issubclass(BedrockEmbeddingsProvider, Embeddings)

    def test_provider_name_is_bedrock(self) -> None:
        assert BedrockEmbeddingsProvider.provider_name == "bedrock"

    def test_default_model_is_titan_v2(self) -> None:
        provider = BedrockEmbeddingsProvider()
        assert provider.model_id == "amazon.titan-embed-text-v2:0"

    def test_respects_explicit_model_id(self) -> None:
        provider = BedrockEmbeddingsProvider(model_id="amazon.titan-embed-text-v1")
        assert provider.model_id == "amazon.titan-embed-text-v1"


class TestOllamaEmbeddingsProvider:
    def test_subclasses_embeddings(self) -> None:
        assert issubclass(OllamaEmbeddingsProvider, Embeddings)

    def test_provider_name_is_ollama(self) -> None:
        assert OllamaEmbeddingsProvider.provider_name == "ollama"

    def test_default_model_is_nomic(self) -> None:
        provider = OllamaEmbeddingsProvider()
        assert provider.model_id == "nomic-embed-text"

    def test_default_base_url_is_loopback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        provider = OllamaEmbeddingsProvider()
        assert provider.base_url == "http://localhost:11434"

    def test_strips_v1_suffix_from_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lets a single OLLAMA_BASE_URL drive both LLM and embeddings paths.

        The LLM provider wants `/v1` (OpenAI-compat endpoint); the
        embeddings backend wants the root because its endpoint is
        `/api/embeddings`. Strip the `/v1` suffix so users can set one
        env var and both work.
        """
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote:9999/v1")
        provider = OllamaEmbeddingsProvider()
        assert provider.base_url == "http://remote:9999"

    def test_base_url_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ignored:1111")
        provider = OllamaEmbeddingsProvider(base_url="http://explicit:2222")
        assert provider.base_url == "http://explicit:2222"

    def test_strips_trailing_slash(self) -> None:
        provider = OllamaEmbeddingsProvider(base_url="http://h:1/")
        assert provider.base_url == "http://h:1"

    @pytest.mark.asyncio
    async def test_get_embedding_returns_numpy_array(self) -> None:
        client = _mock_httpx_client([0.1, 0.2, 0.3])
        provider = OllamaEmbeddingsProvider(model_id="nomic-embed-text", client=client)
        result = await provider.get_embedding("hello")
        assert isinstance(result, np.ndarray)
        assert result.tolist() == pytest.approx([0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_get_embedding_hits_api_embeddings_endpoint(self) -> None:
        client = _mock_httpx_client([1.0, 0.0])
        provider = OllamaEmbeddingsProvider(
            model_id="nomic-embed-text",
            base_url="http://localhost:11434",
            client=client,
        )
        await provider.get_embedding("some text")

        call = client.post.call_args
        assert call.args[0] == "http://localhost:11434/api/embeddings"
        assert call.kwargs["json"] == {
            "model": "nomic-embed-text",
            "prompt": "some text",
        }

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_via_default_impl(self) -> None:
        """`get_embeddings` fans out via `get_embedding` per the ABC default."""
        client = _mock_httpx_client([0.5, 0.5])
        provider = OllamaEmbeddingsProvider(model_id="nomic-embed-text", client=client)
        results = await provider.get_embeddings(["one", "two", "three"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
        # Three POSTs — one per text
        assert client.post.await_count == 3


class TestEmbeddingsFromConfig:
    def test_bedrock_dispatch(self) -> None:
        instance = embeddings_from_config(
            "bedrock", model_id="amazon.titan-embed-text-v2:0"
        )
        assert isinstance(instance, BedrockEmbeddingsProvider)
        assert instance.model_id == "amazon.titan-embed-text-v2:0"

    def test_bedrock_passes_region_kwarg(self) -> None:
        instance = embeddings_from_config(
            "bedrock",
            model_id="amazon.titan-embed-text-v2:0",
            region="us-west-2",
        )
        assert instance.region == "us-west-2"

    def test_ollama_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        instance = embeddings_from_config("ollama", model_id="nomic-embed-text")
        assert isinstance(instance, OllamaEmbeddingsProvider)
        assert instance.model_id == "nomic-embed-text"

    def test_ollama_ignores_region_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI passes `region` unconditionally; Ollama backend must drop it.

        Regression guard: the old behavior would have crashed with an
        unexpected kwarg. `embeddings_from_config` filters bedrock-only
        kwargs out before dispatching to the Ollama constructor.
        """
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        instance = embeddings_from_config(
            "ollama",
            model_id="nomic-embed-text",
            region="us-east-1",  # should be silently ignored
        )
        assert isinstance(instance, OllamaEmbeddingsProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported embeddings provider"):
            embeddings_from_config("unknown", model_id="anything")
