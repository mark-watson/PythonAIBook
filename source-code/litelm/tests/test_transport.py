"""The HTTP layer itself: status mapping, JSON decoding, and SSE framing."""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Callable, Iterator
from email.message import Message
from typing import Any, Self, override

import pytest

import litelm
from litelm import transport

URL = "http://localhost:11434/v1/chat/completions"


class FakeHTTPResponse:
    """Just enough of ``http.client.HTTPResponse`` for the transport."""

    def __init__(self, body: bytes) -> None:
        self._lines = body.splitlines(keepends=True)
        self.closed = False

    def read(self) -> bytes:
        return b"".join(self._lines)

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False

    def close(self) -> None:
        self.closed = True


class FakeURLopener:
    def __init__(self, result: bytes | Exception) -> None:
        self.result = result
        self.calls: list[Any] = []

    def __call__(self, request: Any, timeout: float | None = None) -> FakeHTTPResponse:
        self.calls.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return FakeHTTPResponse(self.result)


def http_error(status: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        URL, status, "error", Message(), io.BytesIO(body.encode("utf-8"))
    )


@pytest.fixture
def opener(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[bytes | Exception], FakeURLopener]:
    def install(result: bytes | Exception) -> FakeURLopener:
        fake = FakeURLopener(result)
        monkeypatch.setattr(transport.urllib.request, "urlopen", fake)
        return fake

    return install


class TestPostJson:
    def test_decodes_a_json_object(self, opener: Any) -> None:
        opener(json.dumps({"ok": True}).encode())
        assert transport.post_json(URL, {}, {"model": "m"}) == {"ok": True}

    def test_sends_the_payload_as_json(self, opener: Any) -> None:
        fake = opener(b"{}")
        transport.post_json(URL, {"Authorization": "Bearer k"}, {"model": "m"})
        request = fake.calls[0]
        assert request.method == "POST"
        assert json.loads(request.data) == {"model": "m"}
        assert request.get_header("Authorization") == "Bearer k"

    @pytest.mark.parametrize(
        ("status", "error_type"),
        [
            (401, litelm.AuthenticationError),
            (403, litelm.AuthenticationError),
            (404, litelm.NotFoundError),
            (429, litelm.RateLimitError),
        ],
    )
    def test_http_errors_map_onto_the_hierarchy(
        self, opener: Any, status: int, error_type: type[litelm.ApiError]
    ) -> None:
        opener(http_error(status, '{"error": "nope"}'))
        with pytest.raises(error_type) as excinfo:
            transport.post_json(URL, {}, {})
        assert excinfo.value.status == status
        assert "nope" in excinfo.value.body

    def test_context_window_errors_are_detected(self, opener: Any) -> None:
        opener(http_error(400, "maximum context length exceeded"))
        with pytest.raises(litelm.ContextWindowExceededError):
            transport.post_json(URL, {}, {})

    def test_a_connection_failure_is_a_litelm_error(self, opener: Any) -> None:
        opener(urllib.error.URLError("connection refused"))
        with pytest.raises(litelm.LitelmError, match="Could not reach"):
            transport.post_json(URL, {}, {})

    def test_a_socket_timeout_is_a_litelm_error(self, opener: Any) -> None:
        opener(TimeoutError("timed out"))
        with pytest.raises(litelm.LitelmError, match="timed out after 5s"):
            transport.post_json(URL, {}, {}, timeout=5)

    def test_other_os_errors_are_litelm_errors(self, opener: Any) -> None:
        opener(ConnectionResetError("reset by peer"))
        with pytest.raises(litelm.LitelmError, match="Could not reach"):
            transport.post_json(URL, {}, {})

    def test_a_non_json_body_is_a_litelm_error(self, opener: Any) -> None:
        opener(b"<html>gateway</html>")
        with pytest.raises(litelm.LitelmError, match="non-JSON body"):
            transport.post_json(URL, {}, {})

    def test_json_that_is_not_an_object_is_rejected(self, opener: Any) -> None:
        opener(b"[1, 2, 3]")
        with pytest.raises(litelm.LitelmError, match="not an object"):
            transport.post_json(URL, {}, {})


class TestPostSse:
    def test_yields_each_data_event(self, opener: Any) -> None:
        body = (
            b'data: {"choices": [{"delta": {"content": "a"}}]}\n'
            b"\n"
            b": keep-alive comment\n"
            b'data: {"choices": [{"delta": {"content": "b"}}]}\n'
            b"data: [DONE]\n"
        )
        opener(body)
        events = list(transport.post_sse(URL, {}, {}))
        assert [event["choices"][0]["delta"]["content"] for event in events] == [
            "a",
            "b",
        ]

    def test_stops_at_the_done_sentinel(self, opener: Any) -> None:
        body = b'data: [DONE]\ndata: {"choices": []}\n'
        opener(body)
        assert list(transport.post_sse(URL, {}, {})) == []

    def test_an_empty_data_field_does_not_end_the_stream(self, opener: Any) -> None:
        body = b'data:\ndata: {"ok": 1}\n'
        opener(body)
        assert list(transport.post_sse(URL, {}, {})) == [{"ok": 1}]

    def test_undecodable_events_are_skipped(self, opener: Any) -> None:
        body = b'data: {not json}\ndata: {"ok": 1}\n'
        opener(body)
        assert list(transport.post_sse(URL, {}, {})) == [{"ok": 1}]

    def test_an_http_error_fails_before_the_first_event(self, opener: Any) -> None:
        opener(http_error(429, "slow down"))
        with pytest.raises(litelm.RateLimitError):
            list(transport.post_sse(URL, {}, {}))

    def test_a_connection_failure_is_a_litelm_error(self, opener: Any) -> None:
        opener(urllib.error.URLError("refused"))
        with pytest.raises(litelm.LitelmError, match="Could not reach"):
            list(transport.post_sse(URL, {}, {}))

    def test_a_socket_timeout_is_a_litelm_error(self, opener: Any) -> None:
        opener(TimeoutError("timed out"))
        with pytest.raises(litelm.LitelmError, match="timed out after 5s"):
            list(transport.post_sse(URL, {}, {}, timeout=5))

    def test_a_timeout_mid_stream_is_a_litelm_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class TimingOutResponse(FakeHTTPResponse):
            @override
            def __iter__(self) -> Iterator[bytes]:
                yield b'data: {"ok": 1}\n'
                raise TimeoutError("timed out")

        def urlopen_stub(
            request: Any, timeout: float | None = None
        ) -> FakeHTTPResponse:
            del request, timeout
            return TimingOutResponse(b"")

        monkeypatch.setattr(transport.urllib.request, "urlopen", urlopen_stub)
        with pytest.raises(litelm.LitelmError, match="timed out after 5s"):
            list(transport.post_sse(URL, {}, {}, timeout=5))
