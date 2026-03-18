"""Configuration loader for REVAL benchmark."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RevalConfig:
    """Benchmark configuration loaded from config.yaml."""

    region: str = "us-east-1"
    max_concurrent: int = 5
    similarity_threshold: float = 0.85
    judge_model_id: str = "amazon.nova-lite-v1:0"
    embeddings_model_id: str = "amazon.titan-embed-text-v2:0"
    models: dict[str, dict] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> RevalConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to config file. Defaults to evals/config.yaml.

    Returns:
        RevalConfig with values from file, falling back to defaults
        if file is missing or fields are absent.
    """
    config_path = Path(path) if path else Path("evals/config.yaml")

    if not config_path.exists():
        return RevalConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {})
    judge = data.get("judge", {})
    embeddings = data.get("embeddings", {})

    return RevalConfig(
        region=defaults.get("region", "us-east-1"),
        max_concurrent=defaults.get("max_concurrent", 5),
        similarity_threshold=defaults.get("similarity_threshold", 0.85),
        judge_model_id=judge.get("model_id", "amazon.nova-lite-v1:0"),
        embeddings_model_id=embeddings.get("model_id", "amazon.titan-embed-text-v2:0"),
        models=data.get("models", {}),
    )


def resolve_model_id(name: str, config: RevalConfig) -> str:
    """Resolve a model short name to a full Bedrock model ID.

    If ``name`` matches a key in the config's models catalog, returns
    the corresponding ``model_id``. Otherwise returns ``name`` unchanged.

    Args:
        name: Short name or full model ID.
        config: Loaded config with models catalog.

    Returns:
        Full Bedrock model ID.
    """
    if name in config.models:
        return config.models[name]["model_id"]
    return name
