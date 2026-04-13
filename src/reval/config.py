"""Configuration loader for REVAL benchmark.

`evals/config.yaml` is a flat model catalog. Every entry under `models:`
has a `provider` + `model_id` pair, and any entry can be used as the
**target** (system under test), **judge**, or **embeddings** backend —
role is determined by which CLI flag references it, not by which
section of the YAML it sits in. The `defaults:` section just names
which catalog keys to use when `--model`, `--judge-model`, or
`--embeddings-model` aren't passed.

This replaces the old shape where `judge:` and `embeddings:` were
top-level blocks carrying bare `model_id` strings without provider
information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RevalConfig:
    """Benchmark configuration loaded from config.yaml."""

    region: str = "us-east-1"
    max_concurrent: int = 5
    similarity_threshold: float = 0.85
    #: Default catalog key for the system-under-test model.
    default_target: str = "claude-haiku-3-5"
    #: Default catalog key for the judge model.
    default_judge: str = "nova-lite"
    #: Default catalog key for the embeddings model.
    default_embeddings: str = "titan-v2"
    #: Flat catalog: `{key: {provider, model_id}}`.
    models: dict[str, dict] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> RevalConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to config file. Defaults to `evals/config.yaml`.

    Returns:
        `RevalConfig` with values from file, falling back to dataclass
        defaults if the file is missing or fields are absent.
    """
    config_path = Path(path) if path else Path("evals/config.yaml")

    if not config_path.exists():
        return RevalConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {})

    return RevalConfig(
        region=defaults.get("region", "us-east-1"),
        max_concurrent=defaults.get("max_concurrent", 5),
        similarity_threshold=defaults.get("similarity_threshold", 0.85),
        default_target=defaults.get("target", "claude-haiku-3-5"),
        default_judge=defaults.get("judge", "nova-lite"),
        default_embeddings=defaults.get("embeddings", "titan-v2"),
        models=data.get("models", {}),
    )


def resolve_model_id(name: str, config: RevalConfig) -> str:
    """Resolve a catalog key (or raw model id) to a full provider model id.

    If `name` matches a key in `config.models`, returns the entry's
    `model_id`. Otherwise returns `name` unchanged so callers can pass
    raw Bedrock ARNs or other provider-specific identifiers on the
    command line without a catalog entry.
    """
    if name in config.models:
        return config.models[name]["model_id"]
    return name


def resolve_model_provider(name: str, config: RevalConfig) -> str:
    """Resolve a catalog key to its `provider` (API surface) string.

    `provider` identifies the API surface — `bedrock`, `anthropic`,
    `openai`, `minimax`, `ollama` — not the model vendor. If `name` is
    not in the catalog, falls back to `"bedrock"` so raw Bedrock ARNs
    passed on the command line still route through `BedrockProvider`.
    """
    if name in config.models:
        return config.models[name].get("provider", "bedrock")
    return "bedrock"


def resolve_model(name: str, config: RevalConfig) -> tuple[str, str]:
    """One-shot helper returning `(provider, model_id)` for a catalog key."""
    return resolve_model_provider(name, config), resolve_model_id(name, config)
