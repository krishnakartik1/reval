"""Tests for config loader and model name resolution."""

from pathlib import Path

import pytest
import yaml

from reval.config import RevalConfig, load_config, resolve_model_id


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config.yaml for testing."""
    config_data = {
        "defaults": {
            "region": "us-west-2",
            "max_concurrent": 10,
            "similarity_threshold": 0.90,
        },
        "models": {
            "claude-haiku": {
                "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "provider": "anthropic",
            },
            "nova-lite": {
                "model_id": "amazon.nova-lite-v1:0",
                "provider": "amazon",
            },
        },
        "judge": {"model_id": "amazon.nova-pro-v1:0"},
        "embeddings": {"model_id": "amazon.titan-embed-text-v2:0"},
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
        assert config.judge_model_id == "amazon.nova-pro-v1:0"
        assert config.embeddings_model_id == "amazon.titan-embed-text-v2:0"
        assert "claude-haiku" in config.models
        assert "nova-lite" in config.models

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.region == "us-east-1"
        assert config.max_concurrent == 5
        assert config.similarity_threshold == 0.85
        assert config.judge_model_id == "amazon.nova-lite-v1:0"
        assert config.embeddings_model_id == "amazon.titan-embed-text-v2:0"
        assert config.models == {}

    def test_partial_config(self, tmp_path):
        """Config with only some fields still returns defaults for the rest."""
        config_path = tmp_path / "partial.yaml"
        config_path.write_text(yaml.dump({"judge": {"model_id": "amazon.nova-pro-v1:0"}}))
        config = load_config(config_path)
        assert config.judge_model_id == "amazon.nova-pro-v1:0"
        assert config.region == "us-east-1"  # default
        assert config.embeddings_model_id == "amazon.titan-embed-text-v2:0"  # default

    def test_empty_file(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(config_path)
        assert config == RevalConfig()


class TestResolveModelId:
    def test_short_name_resolved(self, config_file):
        config = load_config(config_file)
        result = resolve_model_id("claude-haiku", config)
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
