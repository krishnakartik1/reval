"""Provider-aware request/response helpers for Amazon Bedrock models.

Centralises the per-provider format differences so that callers only need
to supply model_id, prompt, and optional parameters.
"""


def build_request_body(
    model_id: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Build the InvokeModel request body for the given model provider.

    Args:
        model_id: Bedrock model identifier
        prompt: User message text
        system_prompt: Optional system message (ignored by providers that
            don't support it)
        max_tokens: Maximum tokens to generate

    Returns:
        JSON-serialisable request body dict
    """
    if _is_anthropic(model_id):
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        return body

    if _is_nova(model_id):
        body = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]
        return body

    if _is_meta(model_id):
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
        }
        if system_prompt:
            # Llama uses a combined prompt with system prefix
            body["prompt"] = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            )
        return body

    if _is_titan(model_id):
        body = {
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": max_tokens},
        }
        if system_prompt:
            body["inputText"] = f"{system_prompt}\n\n{prompt}"
        return body

    # Fallback: Anthropic format
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt
    return body


def parse_response_text(model_id: str, response_body: dict) -> str:
    """Extract the generated text from an InvokeModel response body.

    Args:
        model_id: Bedrock model identifier (same one used for the request)
        response_body: Parsed JSON response from InvokeModel

    Returns:
        The generated text string
    """
    if _is_anthropic(model_id):
        return response_body["content"][0]["text"]

    if _is_nova(model_id):
        return response_body["output"]["message"]["content"][0]["text"]

    if _is_meta(model_id):
        return response_body["generation"]

    if _is_titan(model_id):
        return response_body["results"][0]["outputText"]

    # Fallback: Anthropic format
    return response_body.get("content", [{}])[0].get("text", "")


# ── provider detection helpers ──────────────────────────────────────────


def _is_anthropic(model_id: str) -> bool:
    return "anthropic" in model_id


def _is_nova(model_id: str) -> bool:
    return "nova" in model_id


def _is_meta(model_id: str) -> bool:
    return model_id.startswith("meta.")


def _is_titan(model_id: str) -> bool:
    # Titan models start with amazon. but are NOT nova
    return model_id.startswith("amazon.") and "nova" not in model_id
