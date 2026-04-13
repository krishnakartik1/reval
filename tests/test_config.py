"""Tests for config loader and model name resolution."""

import pytest
import yaml

from reval.config import (
    RevalConfig,
    load_config,
    resolve_model,
    resolve_model_id,
    resolve_model_provider,
)


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config.yaml for testing.

    The `provider` values here are API surfaces (bedrock / anthropic /
    openai / minimax), not model vendors. Phase 3 of the unification
    plan repurposed the field to disambiguate surfaces for the same
    underlying vendor model.
    """
    config_data = {
        "defaults": {
            "region": "us-west-2",
            "max_concurrent": 10,
            "similarity_threshold": 0.90,
            "target": "claude-sonnet-4",
            "judge": "nova-pro",
            "embeddings": "titan-v2",
        },
        "models": {
            "claude-haiku-bedrock": {
                "provider": "bedrock",
                "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            },
            "claude-sonnet-4": {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-20250514",
            },
            "gpt-4o": {
                "provider": "openai",
                "model_id": "gpt-4o",
            },
            "minimax-m2-7": {
                "provider": "minimax",
                "model_id": "MiniMax-M2.7",
            },
            "gemma4-e2b-local": {
                "provider": "ollama",
                "model_id": "gemma4:e2b",
            },
            "nova-pro": {
                "provider": "bedrock",
                "model_id": "amazon.nova-pro-v1:0",
            },
            "titan-v2": {
                "provider": "bedrock",
                "model_id": "amazon.titan-embed-text-v2:0",
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path


class TestLoadConfig:
    def test_load_valid_config(self, config_file):
        config = load_config(config_file)
        assert config.region == "us-west-2"
        assert config.max_concurrent == 10
        assert config.similarity_threshold == 0.90
        assert config.default_target == "claude-sonnet-4"
        assert config.default_judge == "nova-pro"
        assert config.default_embeddings == "titan-v2"
        assert "claude-haiku-bedrock" in config.models
        assert "claude-sonnet-4" in config.models
        assert "gpt-4o" in config.models
        assert "minimax-m2-7" in config.models
        assert "gemma4-e2b-local" in config.models

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.region == "us-east-1"
        assert config.max_concurrent == 5
        assert config.similarity_threshold == 0.85
        assert config.default_target == "claude-haiku-3-5"
        assert config.default_judge == "nova-lite"
        assert config.default_embeddings == "titan-v2"
        assert config.models == {}

    def test_partial_config(self, tmp_path):
        """Config with only some fields still returns defaults for the rest."""
        config_path = tmp_path / "partial.yaml"
        config_path.write_text(yaml.dump({"defaults": {"judge": "nova-pro"}}))
        config = load_config(config_path)
        assert config.default_judge == "nova-pro"
        assert config.region == "us-east-1"  # default
        assert config.default_embeddings == "titan-v2"  # default

    def test_empty_file(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(config_path)
        assert config == RevalConfig()


class TestResolveModelId:
    def test_short_name_resolved(self, config_file):
        config = load_config(config_file)
        result = resolve_model_id("claude-haiku-bedrock", config)
        assert result == "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_full_id_passthrough(self, config_file):
        config = load_config(config_file)
        result = resolve_model_id("amazon.nova-lite-v1:0", config)
        assert result == "amazon.nova-lite-v1:0"

    def test_unknown_short_name_passthrough(self, config_file):
        config = load_config(config_file)
        result = resolve_model_id("some-unknown-name", config)
        assert result == "some-unknown-name"

    def test_empty_models_catalog(self):
        config = RevalConfig()
        result = resolve_model_id("anything", config)
        assert result == "anything"


class TestResolveModelProvider:
    def test_bedrock_short_name(self, config_file):
        config = load_config(config_file)
        assert resolve_model_provider("claude-haiku-bedrock", config) == "bedrock"

    def test_anthropic_short_name(self, config_file):
        config = load_config(config_file)
        assert resolve_model_provider("claude-sonnet-4", config) == "anthropic"

    def test_openai_short_name(self, config_file):
        config = load_config(config_file)
        assert resolve_model_provider("gpt-4o", config) == "openai"

    def test_minimax_short_name(self, config_file):
        config = load_config(config_file)
        assert resolve_model_provider("minimax-m2-7", config) == "minimax"

    def test_unknown_name_falls_back_to_bedrock(self, config_file):
        """Unknown names (e.g. full ARNs on the CLI) default to bedrock.

        Callers that pass a raw Bedrock model ID on the command line
        without a catalog entry still get routed through BedrockProvider,
        which matches pre-Phase-3 behavior.
        """
        config = load_config(config_file)
        assert resolve_model_provider("some.unknown.arn", config) == "bedrock"

    def test_empty_models_catalog_defaults_to_bedrock(self):
        config = RevalConfig()
        assert resolve_model_provider("anything", config) == "bedrock"


class TestResolveModel:
    def test_returns_provider_and_model_id_tuple(self, config_file):
        config = load_config(config_file)
        provider, model_id = resolve_model("claude-sonnet-4", config)
        assert provider == "anthropic"
        assert model_id == "claude-sonnet-4-20250514"

    def test_unknown_name(self, config_file):
        config = load_config(config_file)
        provider, model_id = resolve_model("some.unknown.arn", config)
        assert provider == "bedrock"
        assert model_id == "some.unknown.arn"
