"""The Responses API entry point: request bodies and output-array parsing.

``responses`` is a second wire protocol, so it gets the same treatment as
``completion``: assert on the exact body litelm would POST, and on how every
shape of ``output`` item is folded into a :class:`litelm.Response`.
"""

from __future__ import annotations

from typing import Any

import pytest

import litelm
from tests.conftest import FakeTransport


def message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def responses_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "model": "test-model",
        "status": "completed",
        "output": [message_item("hello")],
        "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
    }
    body.update(overrides)
    return body


class TestResponsesRequest:
    def test_minimal_request_targets_the_responses_endpoint(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses("openai/gpt-5.4-nano", "hi", api_key="test-key")
        request = fake_transport.last_request
        assert request["url"] == "https://api.openai.com/v1/responses"
        assert request["headers"]["Authorization"].startswith("Bearer ")
        assert fake_transport.last_payload == {
            "model": "gpt-5.4-nano",
            "input": "hi",
        }

    def test_options_are_added_when_set(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses(
            "openai/gpt-5.4-nano",
            "hi",
            api_key="test-key",
            instructions="be terse",
            tools=[litelm.WEB_SEARCH],
            temperature=0.0,
            max_output_tokens=64,
        )
        payload = fake_transport.last_payload
        assert payload["instructions"] == "be terse"
        assert payload["tools"] == [{"type": "web_search_preview"}]
        assert payload["temperature"] == 0.0
        assert payload["max_output_tokens"] == 64

    def test_the_web_search_constant_is_copied_not_shared(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses(
            "openai/gpt-5.4-nano", "hi", api_key="test-key", tools=[litelm.WEB_SEARCH]
        )
        tool = fake_transport.last_payload["tools"][0]
        tool["type"] = "mutated"
        assert litelm.WEB_SEARCH == {"type": "web_search_preview"}

    def test_message_lists_pass_through(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses(
            "openai/gpt-5.4-nano",
            [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
            api_key="test-key",
        )
        assert fake_transport.last_payload["input"] == [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]

    def test_a_single_message_dict_is_wrapped(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses(
            "openai/gpt-5.4-nano",
            {"role": "user", "content": "one"},
            api_key="test-key",
        )
        assert fake_transport.last_payload["input"] == [
            {"role": "user", "content": "one"}
        ]

    def test_non_dict_input_entries_are_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="entries must be dicts"):
            litelm.responses("openai/gpt-5.4-nano", ["hi"], api_key="test-key")  # type: ignore[list-item]

    def test_a_bad_input_type_is_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="input must be"):
            litelm.responses("openai/gpt-5.4-nano", 42, api_key="test-key")  # type: ignore[arg-type]

    def test_extra_is_merged_last_and_wins(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [responses_body()]
        litelm.responses(
            "openai/gpt-5.4-nano",
            "hi",
            api_key="test-key",
            temperature=0.0,
            extra={"temperature": 1.0, "reasoning": {"effort": "low"}},
        )
        assert fake_transport.last_payload["temperature"] == 1.0
        assert fake_transport.last_payload["reasoning"] == {"effort": "low"}


class TestResponsesParsing:
    def test_message_items_are_joined(self) -> None:
        body = responses_body(output=[message_item("Hello "), message_item("world")])
        response = litelm.parse_responses_body(body, "m")
        assert response.content == "Hello world"
        assert response.finish_reason == "completed"
        assert response.model == "test-model"
        assert response.usage == litelm.Usage(3, 5, 8)

    def test_the_output_text_shortcut_wins(self) -> None:
        body = responses_body(output_text="from the shortcut")
        assert litelm.parse_responses_body(body, "m").content == "from the shortcut"

    def test_an_empty_output_leaves_content_none(self) -> None:
        assert (
            litelm.parse_responses_body(responses_body(output=[]), "m").content is None
        )

    def test_a_missing_output_is_not_an_error(self) -> None:
        body = responses_body()
        del body["output"]
        assert litelm.parse_responses_body(body, "m").content is None

    def test_reasoning_summaries_are_collected(self) -> None:
        body = responses_body(
            output=[
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "two plus two"}],
                },
                message_item("4"),
            ]
        )
        response = litelm.parse_responses_body(body, "m")
        assert response.reasoning == "two plus two"
        assert response.content == "4"

    def test_function_calls_become_tool_calls(self) -> None:
        body = responses_body(
            output=[
                {
                    "type": "function_call",
                    "call_id": "call_9",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            ]
        )
        (call,) = litelm.parse_responses_body(body, "m").tool_calls
        assert call.id == "call_9"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Paris"}
        assert call.arguments_raw == '{"location": "Paris"}'

    def test_malformed_function_arguments_keep_the_raw_text(self) -> None:
        body = responses_body(
            output=[
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "t",
                    "arguments": "{oops",
                }
            ]
        )
        (call,) = litelm.parse_responses_body(body, "m").tool_calls
        assert call.arguments == {}
        assert call.arguments_raw == "{oops"

    def test_an_incomplete_response_reports_its_reason(self) -> None:
        body = responses_body(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        response = litelm.parse_responses_body(body, "m")
        assert response.finish_reason == "max_output_tokens"

    def test_a_failed_response_raises(self) -> None:
        body = responses_body(status="failed", error={"message": "boom"})
        with pytest.raises(litelm.LitelmError, match="boom"):
            litelm.parse_responses_body(body, "m")

    def test_an_error_field_raises(self) -> None:
        body = responses_body(error={"message": "bad key"})
        with pytest.raises(litelm.LitelmError, match="bad key"):
            litelm.parse_responses_body(body, "m")

    def test_a_failed_status_without_an_error_still_raises(self) -> None:
        with pytest.raises(litelm.LitelmError, match="failed"):
            litelm.parse_responses_body(responses_body(status="failed"), "m")
