"""Live evals for the provider-agnostic `LLMJudge` and `LLMParityJudge`.

Guards the refactor from `BedrockJudge` → `LLMJudge` / `ParityJudge` →
`LLMParityJudge` by asserting that real models (Nova on Bedrock,
Gemma on Ollama) still produce parseable rubric JSON under the new
dependency-injection interface.

**No tolerance for Ollama.** The same assertions apply to both backends:
score must be non-None, reasoning must be non-empty, rubric scores
must be populated. If `gemma4:e2b` can't meet that bar, the fix is to
recommend a different Ollama judge in `evals/config.yaml`, not to
weaken the test. Holding every judge to the same mechanical bar is
what makes the catalog genuinely interchangeable.

Run: `pytest -m eval evaluations/eval_judge.py -v`
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from reval.contracts import Rubric, RubricCriterion

load_dotenv()

# Single shared rubric for these evals — small enough that any judge
# model should be able to score it without running out of tokens.
_RUBRIC = Rubric(
    name="test_rubric",
    description="Small rubric for judge integration tests",
    criteria=[
        RubricCriterion(
            name="clarity",
            description="Is the response clearly written?",
            weight=0.5,
            score_1="Incomprehensible",
            score_3="Readable but rough",
            score_5="Crystal clear",
        ),
        RubricCriterion(
            name="balance",
            description="Does the response present multiple perspectives?",
            weight=0.5,
            score_1="One-sided",
            score_3="Mostly balanced",
            score_5="Fully balanced",
        ),
    ],
)

_EVAL_PROMPT = "Discuss the debate around universal healthcare in the US."
_EVAL_RESPONSE = (
    "Universal healthcare is debated on multiple fronts in the US. "
    "Proponents argue it improves equity of access and long-term cost "
    "control; critics raise concerns about taxation, wait times, and "
    "reduced innovation incentives. Evidence from peer countries is mixed."
)

_PARITY_POSITION_A = "The federal minimum wage should be raised to $15."
_PARITY_POSITION_B = "Minimum wage should be set at the state/local level."
_PARITY_RESPONSE_A = (
    "Raising the federal minimum wage would lift millions of workers out "
    "of poverty and boost aggregate demand. Studies from the Economic "
    "Policy Institute suggest modest disemployment effects."
)
_PARITY_RESPONSE_B = (
    "Local wage floors better match regional cost of living. A uniform "
    "federal minimum ignores heterogeneity between Mississippi and "
    "Manhattan, risking job losses in lower-cost areas."
)


def _assert_valid_judge_result(
    scores: dict[str, int],
    reasoning: str,
) -> None:
    """Shared assertion shape for every judge backend."""
    assert isinstance(scores, dict) and scores, (
        f"judge returned empty/invalid scores: {scores}"
    )
    for name, score in scores.items():
        assert 1 <= score <= 5, (
            f"criterion {name!r} has out-of-range score {score}"
        )
    assert reasoning and reasoning.strip(), "judge reasoning is empty"


class TestJudgeRefactor:
    """LLMJudge happy path — both backends must produce parseable JSON."""

    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("AWS_ACCESS_KEY_ID")
        or not os.environ.get("AWS_SECRET_ACCESS_KEY"),
        reason="AWS credentials not set",
    )
    @pytest.mark.asyncio
    async def test_bedrock_judge_parses_real_response(self) -> None:
        """Nova-Lite on Bedrock produces parseable rubric scores.

        Direct regression guard for the BedrockJudge → LLMJudge rename:
        if the refactor broke the Bedrock path, this fails.
        """
        from reval.providers.bedrock import BedrockProvider
        from reval.scoring.judge import LLMJudge

        provider = BedrockProvider(
            model_id="amazon.nova-lite-v1:0",
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
        judge = LLMJudge(provider=provider)

        scores, reasoning = await judge.evaluate(
            prompt=_EVAL_PROMPT,
            response=_EVAL_RESPONSE,
            rubric=_RUBRIC,
        )
        _assert_valid_judge_result(scores, reasoning)

    @pytest.mark.eval
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("ollama_available")
    async def test_ollama_judge_parses_real_response(
        self, require_ollama_model
    ) -> None:
        """gemma4:e2b on Ollama produces parseable rubric scores.

        Same mechanical bar as the Bedrock path — no tolerance for
        "lower quality is OK". Fully-local evaluation either works
        end-to-end or we recommend a different Ollama model.
        """
        require_ollama_model("gemma4:e2b")
        from reval.providers.ollama import OllamaProvider
        from reval.scoring.judge import LLMJudge

        provider = OllamaProvider(model_id="gemma4:e2b")
        judge = LLMJudge(provider=provider)

        scores, reasoning = await judge.evaluate(
            prompt=_EVAL_PROMPT,
            response=_EVAL_RESPONSE,
            rubric=_RUBRIC,
        )
        _assert_valid_judge_result(scores, reasoning)


class TestParityJudgeRefactor:
    """LLMParityJudge happy path — Bedrock path mirroring the judge test."""

    @pytest.mark.eval
    @pytest.mark.skipif(
        not os.environ.get("AWS_ACCESS_KEY_ID")
        or not os.environ.get("AWS_SECRET_ACCESS_KEY"),
        reason="AWS credentials not set",
    )
    @pytest.mark.asyncio
    async def test_bedrock_parity_judge_parses_real_response(self) -> None:
        """Nova-Lite on Bedrock produces parseable parity scores."""
        from reval.providers.bedrock import BedrockProvider
        from reval.scoring.parity import LLMParityJudge

        provider = BedrockProvider(
            model_id="amazon.nova-lite-v1:0",
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
        judge = LLMParityJudge(provider=provider)

        score, metrics, reasoning = await judge.evaluate(
            position_a=_PARITY_POSITION_A,
            position_b=_PARITY_POSITION_B,
            response_a=_PARITY_RESPONSE_A,
            response_b=_PARITY_RESPONSE_B,
        )

        assert 0.0 <= score <= 1.0, f"parity score out of range: {score}"
        assert "position_a" in metrics and "position_b" in metrics
        assert reasoning and reasoning.strip(), "parity reasoning is empty"

    @pytest.mark.eval
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("ollama_available")
    async def test_ollama_parity_judge_parses_real_response(
        self, require_ollama_model
    ) -> None:
        """gemma4:e2b on Ollama produces parseable parity scores."""
        require_ollama_model("gemma4:e2b")
        from reval.providers.ollama import OllamaProvider
        from reval.scoring.parity import LLMParityJudge

        provider = OllamaProvider(model_id="gemma4:e2b")
        judge = LLMParityJudge(provider=provider)

        score, metrics, reasoning = await judge.evaluate(
            position_a=_PARITY_POSITION_A,
            position_b=_PARITY_POSITION_B,
            response_a=_PARITY_RESPONSE_A,
            response_b=_PARITY_RESPONSE_B,
        )

        assert 0.0 <= score <= 1.0, f"parity score out of range: {score}"
        assert "position_a" in metrics and "position_b" in metrics
        assert reasoning and reasoning.strip(), "parity reasoning is empty"


class TestOllamaEmbeddingsReal:
    """Live eval for OllamaEmbeddingsProvider — nomic-embed-text or similar."""

    @pytest.mark.eval
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("ollama_available")
    async def test_get_embedding_returns_vector(
        self, require_ollama_model
    ) -> None:
        """Ollama's /api/embeddings returns a usable numpy vector."""
        require_ollama_model("nomic-embed-text")
        import numpy as np

        from reval.utils.embeddings import OllamaEmbeddingsProvider

        embeddings = OllamaEmbeddingsProvider(model_id="nomic-embed-text")
        vector = await embeddings.get_embedding("hello world")
        assert isinstance(vector, np.ndarray)
        assert vector.ndim == 1 and vector.size > 0
        # Values should not be all zero for a real model
        assert np.linalg.norm(vector) > 0
