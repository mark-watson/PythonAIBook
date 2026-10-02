"""Streaming: folding server-sent events into chunks."""

from __future__ import annotations

from typing import Any

import pytest

import litelm
from tests.conftest import FakeTransport, chat_body, delta, tool_call_body


def test_text_deltas_arrive_in_order(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [[delta("Hel"), delta("lo"), delta("!")]]

    chunks = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert [chunk.text for chunk in chunks] == ["Hel", "lo", "!"]
    assert all(chunk.model == "test-model" for chunk in chunks)


def test_finish_reason_rides_on_the_last_chunk(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [[delta("done", finish_reason="stop")]]

    (chunk,) = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert chunk.finish_reason == "stop"


def test_the_streaming_request_sets_stream_true(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [[delta("x")]]

    list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert fake_transport.last_payload["stream"] is True


def test_events_without_choices_are_skipped(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [
        [{"model": "m"}, {"model": "m", "choices": []}, delta("real")]
    ]

    chunks = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert [chunk.text for chunk in chunks] == ["real"]


def test_content_block_deltas_are_joined(fake_transport: FakeTransport) -> None:
    event: dict[str, Any] = {
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"content": [{"type": "text", "text": "block"}]},
            }
        ],
    }
    fake_transport.sse_results = [[event]]

    (chunk,) = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert chunk.text == "block"


def test_reasoning_deltas_are_separate_from_text(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [
        [{"model": "m", "choices": [{"index": 0, "delta": {"reasoning": "hmm"}}]}]
    ]

    (chunk,) = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert chunk.text == ""
    assert chunk.reasoning == "hmm"


def test_tool_call_fragments_are_accumulated(
    fake_transport: FakeTransport,
) -> None:
    def fragment(
        index: int,
        *,
        call_id: str | None = None,
        name: str | None = None,
        args: str = "",
    ) -> dict[str, Any]:
        function: dict[str, Any] = {}
        if name is not None:
            function["name"] = name
        if args:
            function["arguments"] = args
        entry: dict[str, Any] = {"index": index, "function": function}
        if call_id is not None:
            entry["id"] = call_id
        return {
            "model": "m",
            "choices": [{"index": 0, "delta": {"tool_calls": [entry]}}],
        }

    fake_transport.sse_results = [
        [
            fragment(0, call_id="call_1", name="get_weather", args='{"loc'),
            fragment(0, args='ation": "Paris"}'),
            delta("", finish_reason="tool_calls"),
        ]
    ]

    chunks = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    # The first fragment already names the call, but its arguments are partial.
    assert chunks[0].tool_calls[0].name == "get_weather"
    assert chunks[0].tool_calls[0].arguments == {}
    assert chunks[0].tool_calls[0].arguments_raw is None

    assert chunks[1].tool_calls[0].arguments == {"location": "Paris"}
    (call,) = chunks[-1].tool_calls
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.arguments_raw == '{"location": "Paris"}'


def test_a_partial_tool_call_fragment_is_not_marked_as_raw_json(
    fake_transport: FakeTransport,
) -> None:
    event = {
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": '{"loc'},
                        }
                    ]
                },
            }
        ],
    }
    fake_transport.sse_results = [[event]]

    (chunk,) = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))

    assert chunk.tool_calls[0].arguments_raw is None


def test_chunks_stringify_to_their_text(fake_transport: FakeTransport) -> None:
    fake_transport.sse_results = [[delta("hi")]]
    (chunk,) = list(litelm.completion("ollama/llama3.2:3b", "x", stream=True))
    assert str(chunk) == "hi"


def test_non_streaming_response_stringifies_to_its_content() -> None:
    assert str(litelm.parse_chat_response(chat_body("answer"), "m")) == "answer"


def test_tool_calls_survive_a_non_streaming_round_trip() -> None:
    body = chat_body(content=None, tool_calls=[tool_call_body("t", "{}")])
    (call,) = litelm.parse_chat_response(body, "m").tool_calls
    assert str(call) == "t({})"


def test_the_streaming_path_also_retries_max_completion_tokens(
    fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README promises this for every call, not just non-streaming ones."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_transport.errors = [
        litelm.ApiError(
            400,
            "Unsupported parameter: 'max_tokens' is not supported with this "
            "model. Use 'max_completion_tokens' instead.",
        )
    ]
    fake_transport.sse_results = [[delta("ok", finish_reason="stop")]]

    chunks = list(
        litelm.completion("openai/gpt-5.4-nano", "hi", max_tokens=64, stream=True)
    )

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert len(fake_transport.requests) == 2
    retry = fake_transport.requests[1]["payload"]
    assert retry["max_completion_tokens"] == 64
    assert "max_tokens" not in retry


def test_a_streaming_error_event_is_raised_not_swallowed(
    fake_transport: FakeTransport,
) -> None:
    fake_transport.sse_results = [
        [delta("partial"), {"error": {"message": "upstream boom"}}]
    ]

    with pytest.raises(litelm.LitelmError, match="upstream boom"):
        list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))


def test_interleaved_tool_call_indices_stay_separate(
    fake_transport: FakeTransport,
) -> None:
    """Providers stream several tool calls at once, keyed by `index`; merging
    them would concatenate one call's arguments onto another's."""

    def fragment(index: int, call_id: str, name: str, args: str) -> dict[str, Any]:
        return {
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "function": {"name": name, "arguments": args},
                            }
                        ]
                    },
                }
            ],
        }

    fake_transport.sse_results = [
        [
            fragment(0, "call_a", "get_weather", '{"location":'),
            fragment(1, "call_b", "calculator", '{"expression":'),
            fragment(0, "", "", ' "Paris"}'),
            fragment(1, "", "", ' "2+2"}'),
            delta("", finish_reason="tool_calls"),
        ]
    ]

    chunks = list(litelm.completion("ollama/llama3.2:3b", "hi", stream=True))
    calls = chunks[-1].tool_calls

    assert [call.name for call in calls] == ["get_weather", "calculator"]
    assert calls[0].arguments == {"location": "Paris"}
    assert calls[1].arguments == {"expression": "2+2"}
