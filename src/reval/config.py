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
    """Resolve a model short name to a full provider model ID.

    If ``name`` matches a key in the config's models catalog, returns
    the corresponding ``model_id``. Otherwise returns ``name`` unchanged.

    Args:
        name: Short name or full model ID.
        config: Loaded config with models catalog.

    Returns:
        Full provider-specific model ID.
    """
    if name in config.models:
        return config.models[name]["model_id"]
    return name


def resolve_model_provider(name: str, config: RevalConfig) -> str:
    """Resolve a model short name to its `provider` (API surface) string.

    `provider` in the YAML entry identifies the API surface, not the
    model vendor — `"bedrock"` / `"anthropic"` / `"openai"` /
    `"minimax"`. The same vendor model can appear under multiple
    surfaces (`claude-sonnet-4-bedrock` vs `claude-sonnet-4-direct`),
    and `provider` is what disambiguates.

    If ``name`` is not in the catalog, falls back to ``"bedrock"``:
    callers that pass a full Bedrock ARN on the command line still get
    routed through `BedrockProvider`.
    """
    if name in config.models:
        return config.models[name].get("provider", "bedrock")
    return "bedrock"


def resolve_model(name: str, config: RevalConfig) -> tuple[str, str]:
    """One-shot helper returning `(provider, model_id)`.

    Convenience wrapper around `resolve_model_provider` +
    `resolve_model_id` for CLI code that needs both values.
    """
    return resolve_model_provider(name, config), resolve_model_id(name, config)
