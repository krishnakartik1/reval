---
title: Providers & models
order: 1
description: The five supported provider surfaces and how to register a new model.
---

REVAL supports five LLM provider surfaces, all hidden behind a single
`LLMProvider` async ABC:

| Surface     | Module                                      | Auth                          |
|-------------|---------------------------------------------|-------------------------------|
| Bedrock     | `reval.providers.bedrock`                   | AWS IAM (env vars or profile) |
| Anthropic   | `reval.providers.anthropic_direct`          | `ANTHROPIC_API_KEY`           |
| OpenAI      | `reval.providers.openai_compat`             | `OPENAI_API_KEY`              |
| MiniMax     | `reval.providers.minimax`                   | `MINIMAX_API_KEY`              |
| Ollama      | `reval.providers.ollama`                    | none — local daemon on :11434 |

The provider registry lives in `reval/src/reval/providers/factory.py`:

```python
_REGISTRY: dict[str, type[LLMProvider]] = {
    "bedrock": BedrockProvider,
    "anthropic": AnthropicProvider,
    "minimax": MinimaxProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}
```

A sixth entry (`openai_compat`) reuses `OpenAIProvider` against
third-party endpoints like Together, Groq, OpenRouter, and
Fireworks — any service that speaks the OpenAI API shape. Point
`base_url` at your endpoint via environment variables.

## Choosing a judge and embeddings

Judge and embeddings are just more entries in the same
`evals/config.yaml` catalog. Any registered model can play any
role — target, judge, or embeddings — depending on which CLI flag
references it:

```bash
# Bedrock target, Bedrock judge, Bedrock embeddings (the defaults)
reval run --model claude-haiku-3-5

# Anthropic target, Bedrock judge (explicit), Bedrock embeddings
reval run --model claude-sonnet-4 --judge-model nova-pro

# OpenAI target, Anthropic judge, Ollama embeddings
reval run --model gpt-4o \
          --judge-model claude-opus-4 \
          --embeddings-model nomic-embed
```

The defaults are `nova-lite` (judge) and `titan-v2` (embeddings),
both on Bedrock — so running without any judge/embeddings flags
needs AWS credentials. Override both with non-Bedrock entries if
you want to run fully off-AWS.

## Adding a new model

Edit `reval/evals/config.yaml` and add a handle under `models:`:

```yaml
models:
  # … existing entries …

  my-new-model:
    provider: openai          # must match a _REGISTRY key
    model_id: gpt-5-turbo     # the string the provider's SDK accepts
```

The `provider:` field must be one of the keys in `_REGISTRY` above.
The `model_id:` value is passed verbatim to the provider — for
Bedrock, that's a Bedrock ARN (e.g. `amazon.nova-lite-v1:0`); for
Anthropic, it's a model name (`claude-sonnet-4-20250514`); for
Ollama, it's whatever tag you've pulled locally (`gemma4:e2b`).

Once the handle is in the catalog, it's usable immediately:

```bash
reval run --model my-new-model
```

No code changes required — the factory wires up `LLMProvider`
instances lazily based on the catalog entry.

## Adding a new provider surface

If you need a surface that's not in the registry (a new cloud,
a vendor-specific API), you'll need to write a provider
implementation:

1. Subclass `reval.contracts.provider.LLMProvider`.
2. Implement the async `acomplete(self, system: str | None, user: str, *, max_tokens: int = 4096) -> CompletionResult` method.
3. Register the class in `_REGISTRY` in
   `reval/src/reval/providers/factory.py`.
4. Add a config entry in `evals/config.yaml` that uses your new
   `provider:` key.
5. Add a test under `reval/tests/test_provider_<name>.py`.

The five existing providers are good templates — they all share
the same async error-handling and retry pattern via the base
class. The shortest one to crib from is `providers/minimax.py`,
which delegates to the Anthropic-compatible wire format and adds
~50 lines of provider-specific request shaping.

## Fully local (no cloud credentials)

If you want to run REVAL with zero cloud credits, use Ollama for
target, judge, AND embeddings:

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text

reval run --model gemma4-e2b-local \
          --judge-model gemma4-e2b-local \
          --embeddings-model nomic-embed
```

Everything runs on localhost. The judge quality is markedly lower
than `nova-lite` or `claude-opus-4`, so local scores should be
compared against other local runs, not against cloud-judged runs.
