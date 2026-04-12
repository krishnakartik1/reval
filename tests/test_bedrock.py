"""Tests for provider-aware Bedrock request/response helpers."""

from reval.utils.bedrock import build_request_body, parse_response_text


class TestBuildRequestBody:
    """Tests for build_request_body across providers."""

    def test_anthropic_model(self):
        body = build_request_body(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "Hello",
            system_prompt="Be helpful.",
            max_tokens=1024,
        )
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 1024
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["system"] == "Be helpful."

    def test_anthropic_cross_region(self):
        body = build_request_body(
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "Hello",
        )
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert "system" not in body

    def test_nova_model(self):
        body = build_request_body(
            "amazon.nova-lite-v1:0",
            "Hello",
            system_prompt="Be helpful.",
            max_tokens=2048,
        )
        assert body["messages"] == [{"role": "user", "content": [{"text": "Hello"}]}]
        assert body["system"] == [{"text": "Be helpful."}]
        assert body["inferenceConfig"] == {"maxTokens": 2048}
        assert "anthropic_version" not in body

    def test_nova_no_system(self):
        body = build_request_body("amazon.nova-pro-v1:0", "Hello")
        assert "system" not in body

    def test_meta_model(self):
        body = build_request_body(
            "meta.llama3-70b-instruct-v1:0",
            "Hello",
            max_tokens=4096,
        )
        assert body["prompt"] == "Hello"
        assert body["max_gen_len"] == 4096

    def test_meta_with_system_prompt(self):
        body = build_request_body(
            "meta.llama3-70b-instruct-v1:0",
            "Hello",
            system_prompt="Be helpful.",
        )
        assert "<<SYS>>" in body["prompt"]
        assert "Be helpful." in body["prompt"]
        assert "Hello" in body["prompt"]

    def test_titan_model(self):
        body = build_request_body(
            "amazon.titan-text-express-v1",
            "Hello",
            max_tokens=512,
        )
        assert body["inputText"] == "Hello"
        assert body["textGenerationConfig"]["maxTokenCount"] == 512

    def test_titan_with_system_prompt(self):
        body = build_request_body(
            "amazon.titan-text-express-v1",
            "Hello",
            system_prompt="Be helpful.",
        )
        assert "Be helpful." in body["inputText"]
        assert "Hello" in body["inputText"]

    def test_unknown_model_defaults_to_anthropic(self):
        body = build_request_body("some.unknown-model-v1", "Hello")
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    def test_system_prompt_omitted_when_none(self):
        for model_id in [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.nova-lite-v1:0",
            "meta.llama3-70b-instruct-v1:0",
            "amazon.titan-text-express-v1",
        ]:
            body = build_request_body(model_id, "Hello")
            assert "system" not in body, f"system key present for {model_id}"


class TestParseResponseText:
    """Tests for parse_response_text across providers."""

    def test_anthropic(self):
        body = {"content": [{"text": "response text"}]}
        assert (
            parse_response_text("anthropic.claude-3-haiku-v1:0", body)
            == "response text"
        )

    def test_anthropic_cross_region(self):
        body = {"content": [{"text": "response text"}]}
        assert (
            parse_response_text("us.anthropic.claude-3-5-haiku-v1:0", body)
            == "response text"
        )

    def test_nova(self):
        body = {"output": {"message": {"content": [{"text": "nova response"}]}}}
        assert parse_response_text("amazon.nova-lite-v1:0", body) == "nova response"

    def test_meta(self):
        body = {"generation": "llama response"}
        assert (
            parse_response_text("meta.llama3-70b-instruct-v1:0", body)
            == "llama response"
        )

    def test_titan(self):
        body = {"results": [{"outputText": "titan response"}]}
        assert (
            parse_response_text("amazon.titan-text-express-v1", body)
            == "titan response"
        )

    def test_unknown_fallback(self):
        body = {"content": [{"text": "fallback response"}]}
        assert parse_response_text("some.unknown-model", body) == "fallback response"
