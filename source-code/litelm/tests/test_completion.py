"""Request bodies, response parsing, and the completion / embedding entry points."""

from __future__ import annotations

from typing import cast

import pytest

import litelm
from tests.conftest import FakeTransport, chat_body, tool_call_body


def weather_tool() -> litelm.Tool:
    return litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        lambda args: f"sunny in {args['location']}",
    )


class TestBuildChatPayload:
    def test_minimal_payload_only_carries_model_and_messages(self) -> None:
        payload = litelm.build_chat_payload("m", [{"role": "user", "content": "hi"}])
        assert payload == {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_sampling_options_are_omitted_when_unset(self) -> None:
        payload = litelm.build_chat_payload("m", [])
        for key in ("temperature", "max_tokens", "top_p", "stream", "tools"):
            assert key not in payload

    def test_sampling_options_are_included_when_set(self) -> None:
        payload = litelm.build_chat_payload(
            "m", [], temperature=0.0, max_tokens=16, top_p=0.5, stream=True
        )
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 16
        assert payload["top_p"] == 0.5
        assert payload["stream"] is True

    def test_tools_add_a_tool_choice(self) -> None:
        payload = litelm.build_chat_payload("m", [], tools=[weather_tool()])
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "get_weather"

    def test_tool_choice_can_be_disabled(self) -> None:
        payload = litelm.build_chat_payload(
            "m", [], tools=[weather_tool()], tool_choice=None
        )
        assert "tool_choice" not in payload

    def test_tool_choice_without_tools_is_not_sent(self) -> None:
        assert "tool_choice" not in litelm.build_chat_payload("m", [])

    def test_a_misspelled_tool_choice_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="tool_choice"):
            litelm.build_chat_payload(
                "m",
                [],
                tools=[weather_tool()],
                tool_choice=cast(litelm.ToolChoice, "requried"),
            )

    def test_a_bad_tool_choice_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="tool_choice"):
            litelm.build_chat_payload(
                "m",
                [],
                tools=[weather_tool()],
                tool_choice=cast(litelm.ToolChoice, 42),
            )

    def test_extra_is_merged_last_and_wins(self) -> None:
        payload = litelm.build_chat_payload(
            "m",
            [],
            temperature=0.0,
            extra={"temperature": 1.5, "thinking": {"type": "enabled"}},
        )
        assert payload["temperature"] == 1.5
        assert payload["thinking"] == {"type": "enabled"}


class TestParseChatResponse:
    def test_string_content(self) -> None:
        response = litelm.parse_chat_response(chat_body("4"), "m")
        assert response.content == "4"
        assert response.finish_reason == "stop"
        assert response.model == "test-model"
        assert response.tool_calls == []
        assert response.usage == litelm.Usage(3, 5, 8)

    def test_content_block_lists_are_joined(self) -> None:
        body = chat_body(None)
        body["choices"][0]["message"]["content"] = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert litelm.parse_chat_response(body, "m").content == "Hello world"

    def test_null_content_stays_none(self) -> None:
        assert litelm.parse_chat_response(chat_body(None), "m").content is None

    def test_tool_calls_are_parsed(self) -> None:
        body = chat_body(
            content=None,
            tool_calls=[tool_call_body("get_weather", '{"location": "Paris"}')],
            finish_reason="tool_calls",
        )
        (call,) = litelm.parse_chat_response(body, "m").tool_calls
        assert call.id == "call_1"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Paris"}
        assert call.arguments_raw == '{"location": "Paris"}'

    def test_malformed_tool_call_arguments_keep_the_raw_text(self) -> None:
        body = chat_body(content=None, tool_calls=[tool_call_body("t", "{oops")])
        (call,) = litelm.parse_chat_response(body, "m").tool_calls
        assert call.arguments == {}
        assert call.arguments_raw == "{oops"

    def test_total_tokens_comes_from_the_provider_not_the_sum(self) -> None:
        """A provider that reports its own total must be believed."""
        body = chat_body(
            "x",
            usage={
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 99,
            },
        )
        assert litelm.parse_chat_response(body, "m").usage == litelm.Usage(3, 5, 99)

    def test_a_textless_content_block_list_is_an_empty_string(self) -> None:
        """Matching the Racket port: [] joins to "", while null stays None."""
        body = chat_body(None)
        body["choices"][0]["message"]["content"] = []
        assert litelm.parse_chat_response(body, "m").content == ""

    def test_a_number_in_place_of_tool_arguments_is_not_silently_dropped(
        self,
    ) -> None:
        body = chat_body(content=None)
        body["choices"][0]["message"]["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": 5}}
        ]
        (call,) = litelm.parse_chat_response(body, "m").tool_calls
        assert call.arguments == {}
        assert call.arguments_raw == "5"

    def test_usage_falls_back_to_input_and_output_tokens(self) -> None:
        body = chat_body("x", usage={"input_tokens": 2, "output_tokens": 4})
        usage = litelm.parse_chat_response(body, "m").usage
        assert usage == litelm.Usage(
            prompt_tokens=2, completion_tokens=4, total_tokens=6
        )

    def test_cached_tokens_are_read_from_prompt_tokens_details(self) -> None:
        body = chat_body(
            "x",
            usage={
                "prompt_tokens": 2033,
                "completion_tokens": 5,
                "total_tokens": 2038,
                "prompt_tokens_details": {"cached_tokens": 2023},
            },
        )
        usage = litelm.parse_chat_response(body, "m").usage
        assert usage is not None
        assert usage.cached_tokens == 2023

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt_cache_hit_tokens", 7),  # DeepSeek
            ("cache_read_input_tokens", 7),  # Anthropic
            ("cached_tokens", 7),
        ],
    )
    def test_cached_tokens_accept_other_provider_spellings(
        self, field: str, value: int
    ) -> None:
        body = chat_body("x", usage={"prompt_tokens": 9, field: value})
        usage = litelm.parse_chat_response(body, "m").usage
        assert usage is not None
        assert usage.cached_tokens == 7

    def test_absent_cache_numbers_stay_none_not_zero(self) -> None:
        """A server that reports no cache fields is not reporting a cache miss."""
        body = chat_body("x", usage={"prompt_tokens": 9, "completion_tokens": 1})
        usage = litelm.parse_chat_response(body, "m").usage
        assert usage is not None
        assert usage.cached_tokens is None

    def test_missing_usage_is_none(self) -> None:
        body = chat_body("x")
        del body["usage"]
        assert litelm.parse_chat_response(body, "m").usage is None

    def test_separate_reasoning_field_is_surfaced(self) -> None:
        body = chat_body("answer", reasoning="because")
        assert litelm.parse_chat_response(body, "m").reasoning == "because"

    def test_model_name_falls_back_to_the_request(self) -> None:
        body = chat_body("x")
        del body["model"]
        assert litelm.parse_chat_response(body, "requested").model == "requested"

    def test_an_error_body_becomes_a_litelm_error(self) -> None:
        # Assert on the *branch*, not just on the body text: the "no choices"
        # message also embeds the body, so a looser assertion would pass even
        # if the error branch were removed.
        with pytest.raises(litelm.LitelmError) as excinfo:
            litelm.parse_chat_response({"error": {"message": "rate limited"}}, "m")
        assert str(excinfo.value).startswith("LLM API error:")
        assert "rate limited" in str(excinfo.value)

    def test_a_body_without_choices_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="no choices"):
            litelm.parse_chat_response({"choices": []}, "m")


class TestCompletionTransport:
    def test_sends_the_routed_url_and_bearer_token(
        self, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake_transport.json_results = [chat_body("hi")]

        litelm.completion("openai/gpt-5.4-nano", "hello")

        assert fake_transport.last_request["url"] == (
            "https://api.openai.com/v1/chat/completions"
        )
        assert (
            fake_transport.last_request["headers"]["Authorization"] == "Bearer sk-test"
        )
        assert fake_transport.last_payload["model"] == "gpt-5.4-nano"

    def test_local_provider_needs_no_key(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [chat_body("hi")]

        litelm.completion("ollama/llama3.2:3b", "hello")

        assert "Authorization" not in fake_transport.last_request["headers"]
        assert fake_transport.last_request["url"] == (
            "http://localhost:11434/v1/chat/completions"
        )

    def test_api_base_and_key_overrides(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [chat_body("hi")]

        litelm.completion(
            "openai/gpt-5.4-nano",
            "hello",
            api_base="http://localhost:9999/v1",
            api_key="explicit",
        )

        assert fake_transport.last_request["url"] == (
            "http://localhost:9999/v1/chat/completions"
        )
        assert (
            fake_transport.last_request["headers"]["Authorization"] == "Bearer explicit"
        )

    def test_extra_headers_are_added(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [chat_body("hi")]
        litelm.completion(
            "ollama/llama3.2:3b", "hello", extra_headers={"X-Trace": "abc"}
        )
        assert fake_transport.last_request["headers"]["X-Trace"] == "abc"

    def test_system_is_prepended_as_a_message(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [chat_body("hi")]
        litelm.completion("ollama/llama3.2:3b", "hello", system="Be terse.")
        messages = fake_transport.last_payload["messages"]
        assert messages[0] == {"role": "system", "content": "Be terse."}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_provider_override_keeps_the_full_model_name(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [chat_body("hi")]
        litelm.completion(
            "meta/llama-3.1-8b-instruct",
            "hello",
            provider="nvidia",
            api_key="test-key",
        )
        assert fake_transport.last_payload["model"] == "meta/llama-3.1-8b-instruct"
        assert "integrate.api.nvidia.com" in fake_transport.last_request["url"]

    def test_timeout_is_forwarded(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [chat_body("hi")]
        litelm.completion("ollama/llama3.2:3b", "hello", timeout=7.5)
        assert fake_transport.last_request["timeout"] == 7.5


class TestMaxTokensFallback:
    def test_retries_with_max_completion_tokens(
        self, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake_transport.errors = [
            litelm.ApiError(
                400,
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead.",
            )
        ]
        fake_transport.json_results = [chat_body("hi")]

        litelm.completion("openai/gpt-5.4-nano", "hello", max_tokens=64)

        assert len(fake_transport.requests) == 2
        retry = fake_transport.requests[1]["payload"]
        assert retry["max_completion_tokens"] == 64
        assert "max_tokens" not in retry

    def test_other_400s_propagate(self, fake_transport: FakeTransport) -> None:
        fake_transport.errors = [litelm.ApiError(400, "bad request")]
        with pytest.raises(litelm.ApiError, match="bad request"):
            litelm.completion("ollama/llama3.2:3b", "hello", max_tokens=64)


class TestAsk:
    def test_returns_the_reply_text(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [chat_body("4")]
        assert litelm.ask("ollama/llama3.2:3b", "What is 2+2?") == "4"

    def test_a_tool_only_answer_is_an_empty_string(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [
            chat_body(content=None, tool_calls=[tool_call_body("t", "{}")])
        ]
        assert litelm.ask("ollama/llama3.2:3b", "hi", tools=[weather_tool()]) == ""


class TestEmbedding:
    def test_string_input_is_wrapped_in_a_list(
        self, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        fake_transport.json_results = [
            {"data": [{"embedding": [0.5, 0.25]}, {"embedding": [1, 2]}]}
        ]

        vectors = litelm.embedding("openai/text-embedding-3-small", ["a", "b"])

        assert vectors == [[0.5, 0.25], [1.0, 2.0]]
        assert fake_transport.last_payload["input"] == ["a", "b"]
        assert fake_transport.last_request["url"].endswith("/embeddings")

    def test_a_bare_string_is_accepted(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [{"data": [{"embedding": [1.0]}]}]
        assert litelm.embedding("ollama/nomic-embed-text", "hello") == [[1.0]]
        assert fake_transport.last_payload["input"] == ["hello"]

    def test_dimensions_are_forwarded(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [{"data": [{"embedding": [1.0]}]}]
        litelm.embedding("ollama/nomic-embed-text", "hello", dimensions=256)
        assert fake_transport.last_payload["dimensions"] == 256

    def test_empty_vectors_are_tolerated(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [{"data": [{}]}]
        assert litelm.embedding("ollama/nomic-embed-text", "hello") == [[]]

    def test_a_malformed_vector_is_an_error_not_a_shorter_vector(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [{"data": [{"embedding": [1, "x", 3]}]}]
        with pytest.raises(litelm.LitelmError, match="Malformed embedding vector"):
            litelm.embedding("ollama/nomic-embed-text", "hello")

    def test_bad_input_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="list of strings"):
            litelm.embedding("ollama/nomic-embed-text", cast(list[str], [1, 2]))

    def test_a_body_without_data_is_rejected(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [{"nope": True}]
        with pytest.raises(litelm.LitelmError, match="no data array"):
            litelm.embedding("ollama/nomic-embed-text", "hello")
