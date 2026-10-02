"""Message normalization, tool-call rendering, and image attachment."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import cast

import pytest

import litelm

PNG = b"\x89PNG\r\n\x1a\n" + b"fake png payload"
JPEG = b"\xff\xd8\xff" + b"fake jpeg payload"


def test_string_is_shorthand_for_one_user_message() -> None:
    assert litelm.normalize_messages("hello") == [{"role": "user", "content": "hello"}]


def test_single_dict_is_accepted() -> None:
    assert litelm.normalize_messages({"role": "user", "content": "hi"}) == [
        {"role": "user", "content": "hi"}
    ]


def test_roles_are_lowercased() -> None:
    assert litelm.normalize_messages({"role": "SYSTEM", "content": "s"}) == [
        {"role": "system", "content": "s"}
    ]


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(litelm.LitelmError, match="unknown message role"):
        litelm.normalize_messages({"role": "wizard", "content": "hi"})


def test_non_string_content_is_rejected() -> None:
    with pytest.raises(litelm.LitelmError, match="content must be a string"):
        litelm.normalize_messages({"role": "user", "content": 42})


def test_tool_message_needs_a_tool_call_id() -> None:
    with pytest.raises(litelm.LitelmError, match="tool_call_id"):
        litelm.normalize_messages({"role": "tool", "content": "sunny"})


def test_assistant_message_needs_content_or_tool_calls() -> None:
    with pytest.raises(litelm.LitelmError, match="content or tool_calls"):
        litelm.normalize_messages({"role": "assistant"})


def test_assistant_message_with_only_tool_calls_has_no_content_key() -> None:
    call = litelm.ToolCall(id="c1", name="get_weather", arguments={"location": "Paris"})
    (message,) = litelm.normalize_messages({"role": "assistant", "tool_calls": [call]})
    assert "content" not in message
    assert message["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
        }
    ]


def test_the_original_argument_text_is_resent_verbatim() -> None:
    """Re-encoding the parsed dict would change spacing and key order, and the
    model may be matching on the exact text it produced."""
    raw = '{"location":"Paris","units":"celsius"}'
    call = litelm.ToolCall(
        id="c1", name="get_weather", arguments={"location": "Paris"}, arguments_raw=raw
    )
    (message,) = litelm.normalize_messages({"role": "assistant", "tool_calls": [call]})
    assert message["tool_calls"][0]["function"]["arguments"] == raw


def test_none_content_is_not_forwarded_to_the_wire() -> None:
    call = litelm.ToolCall(id="c1", name="get_weather", arguments={})
    (message,) = litelm.normalize_messages(
        {"role": "assistant", "content": None, "tool_calls": [call]}
    )
    assert "content" not in message


def test_extra_keys_are_not_forwarded() -> None:
    (message,) = litelm.normalize_messages(
        {"role": "user", "content": "hi", "images": None, "tool_calls": None}
    )
    assert message == {"role": "user", "content": "hi"}


def test_assistant_and_tool_messages_close_the_loop() -> None:
    response = litelm.Response(
        content=None,
        tool_calls=[litelm.ToolCall(id="c1", name="get_weather", arguments={})],
    )
    assistant = litelm.assistant_message(response)
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "content" not in assistant

    tool = litelm.tool_message(
        litelm.ToolResult(call_id="c1", name="get_weather", result="sunny")
    )
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "sunny"}


def test_empty_content_is_dropped_from_a_tool_calling_assistant_turn() -> None:
    """`"content": ""` next to `tool_calls` reads as an empty answer to some
    servers, so the key is omitted when the turn produced only tool calls."""
    response = litelm.Response(
        content="",
        tool_calls=[litelm.ToolCall(id="c1", name="get_weather", arguments={})],
    )
    assert "content" not in litelm.assistant_message(response)


def test_text_alongside_tool_calls_is_kept() -> None:
    response = litelm.Response(
        content="Let me check.",
        tool_calls=[litelm.ToolCall(id="c1", name="get_weather", arguments={})],
    )
    assert litelm.assistant_message(response)["content"] == "Let me check."


def test_a_plain_reply_keeps_its_content() -> None:
    assert litelm.assistant_message(litelm.Response(content="4")) == {
        "role": "assistant",
        "content": "4",
    }


def test_an_empty_plain_reply_stays_a_valid_message() -> None:
    """Without tool calls the empty string must survive, or re-sending the
    history would fail validation."""
    assert litelm.assistant_message(litelm.Response(content="")) == {
        "role": "assistant",
        "content": "",
    }


def test_assistant_message_rejects_a_non_response() -> None:
    with pytest.raises(litelm.LitelmError):
        litelm.assistant_message(cast(litelm.Response, "not a response"))


class TestImages:
    def test_bytes_become_a_data_url_with_a_sniffed_mime_type(self) -> None:
        (message,) = litelm.normalize_messages(
            {"role": "user", "content": "what is this?", "images": [PNG]}
        )
        parts = message["content"]
        assert parts[0] == {"type": "text", "text": "what is this?"}
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG

    def test_jpeg_is_detected(self) -> None:
        (message,) = litelm.normalize_messages(
            {"role": "user", "content": None, "images": [JPEG]}
        )
        assert message["content"][0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )

    def test_a_file_path_is_read_from_disk(self, tmp_path: Path) -> None:
        image = tmp_path / "ticket.png"
        image.write_bytes(PNG)
        (message,) = litelm.normalize_messages(
            {"role": "user", "content": "describe", "images": [image]}
        )
        assert message["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    def test_http_urls_pass_through_untouched(self) -> None:
        (message,) = litelm.normalize_messages(
            {
                "role": "user",
                "content": "describe",
                "images": ["https://example.test/cat.png"],
            }
        )
        assert message["content"][1]["image_url"]["url"] == (
            "https://example.test/cat.png"
        )

    def test_text_part_is_omitted_when_there_is_no_content(self) -> None:
        (message,) = litelm.normalize_messages(
            {"role": "user", "content": None, "images": [PNG]}
        )
        assert len(message["content"]) == 1

    def test_images_on_a_system_message_are_rejected(self) -> None:
        with pytest.raises(litelm.LitelmError, match="only user messages"):
            litelm.normalize_messages(
                {"role": "system", "content": "s", "images": [PNG]}
            )
