"""Embedding utilities using Amazon Bedrock."""

import asyncio
import json
from collections.abc import Sequence

import aioboto3
import numpy as np


class BedrockEmbeddings:
    """Client for generating embeddings via Amazon Bedrock."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        session: aioboto3.Session | None = None,
    ):
        self.model_id = model_id
        self.region = region
        self._session = session or aioboto3.Session()

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
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

    async def get_embeddings(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Get embeddings for multiple texts in parallel."""
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)


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
    embeddings_client: BedrockEmbeddings | None = None,
) -> float:
    """Compute semantic similarity between two texts.

    Args:
        text_a: First text
        text_b: Second text
        embeddings_client: Optional client to reuse

    Returns:
        Similarity score between 0 and 1
    """
    client = embeddings_client or BedrockEmbeddings()
    embeddings = await client.get_embeddings([text_a, text_b])
    return cosine_similarity(embeddings[0], embeddings[1])
