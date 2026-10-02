"""End-to-end tests over a real socket.

Everything else in the suite replaces ``litelm.transport`` with a fake, which
proves the *payloads* are right but never proves the HTTP layer works. These
tests start a throwaway OpenAI-compatible server on a loopback port and drive
litelm against it, so ``urllib``, JSON encoding, the ``Authorization`` header, SSE
framing and the error mapping are all exercised for real — minus the third-party
provider.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, override

import pytest

import litelm

STREAM_TEXT = "streamed reply"


class LoopbackServer(ThreadingHTTPServer):
    """A loopback server that records every request it is sent."""

    requests: list[dict[str, Any]]


def _last_user_text(payload: dict[str, Any]) -> str:
    texts = [
        message.get("content")
        for message in payload.get("messages", [])
        if message.get("role") == "user"
    ]
    return str(texts[-1]) if texts else ""


def _tool_result_present(payload: dict[str, Any]) -> bool:
    return any(message.get("role") == "tool" for message in payload.get("messages", []))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Keep the test output clean."""

    # No @override: BaseHTTPRequestHandler dispatches do_POST dynamically.
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload: dict[str, Any] = json.loads(raw or b"{}")
        server = self.server
        assert isinstance(server, LoopbackServer)
        server.requests.append(
            {"path": self.path, "headers": dict(self.headers), "payload": payload}
        )

        if self.path.endswith("/embeddings"):
            self._json({"data": [{"embedding": [1.0, 2.0, 3.0]}]})
            return
        if self.path.endswith("/chat/completions"):
            self._chat(payload)
            return
        self._json({"error": {"message": "unknown path"}}, status=404)

    def _chat(self, payload: dict[str, Any]) -> None:
        if _last_user_text(payload) == "rate limit me":
            self._json({"error": {"message": "slow down"}}, status=429)
            return
        if payload.get("stream"):
            closing_delta: dict[str, Any] = {}
            events = [
                {"choices": [{"index": 0, "delta": {"content": "streamed "}}]},
                {"choices": [{"index": 0, "delta": {"content": "reply"}}]},
                {
                    "choices": [
                        {"index": 0, "delta": closing_delta, "finish_reason": "stop"}
                    ]
                },
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            body += "data: [DONE]\n\n"
            self._send(body.encode(), "text/event-stream")
            return
        if "tools" in payload and not _tool_result_present(payload):
            self._json(
                {
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"location": "Paris"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            )
            return
        text = (
            "It is sunny and 22C in Paris."
            if _tool_result_present(payload)
            else f"echo: {_last_user_text(payload)}"
        )
        self._json(
            {
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            }
        )

    def _json(self, body: dict[str, Any], status: int = 200) -> None:
        self._send(json.dumps(body).encode(), "application/json", status)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server() -> Iterator[LoopbackServer]:
    httpd = LoopbackServer(("127.0.0.1", 0), Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def api_base(server: LoopbackServer) -> str:
    host, port = server.server_address[0], server.server_address[1]
    return f"http://{host}:{port}/v1"


def test_completion_round_trip(server: LoopbackServer, api_base: str) -> None:
    response = litelm.completion(
        "openai/echo-model", "hello", api_base=api_base, api_key="test-key"
    )

    assert response.content == "echo: hello"
    assert response.model == "echo-model"
    assert response.finish_reason == "stop"
    assert response.usage == litelm.Usage(5, 7, 12)

    request = server.requests[-1]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["payload"]["model"] == "echo-model"
    assert request["payload"]["messages"] == [{"role": "user", "content": "hello"}]


def test_system_and_sampling_options_reach_the_server(
    server: LoopbackServer, api_base: str
) -> None:
    litelm.completion(
        "openai/echo-model",
        "hello",
        system="Be terse.",
        temperature=0.25,
        max_tokens=32,
        top_p=0.9,
        api_base=api_base,
        api_key="test-key",
    )

    payload = server.requests[-1]["payload"]
    assert payload["messages"][0] == {"role": "system", "content": "Be terse."}
    assert payload["temperature"] == 0.25
    assert payload["max_tokens"] == 32
    assert payload["top_p"] == 0.9


def test_streaming_round_trip(api_base: str) -> None:
    chunks = list(
        litelm.completion(
            "openai/echo-model", "hi", stream=True, api_base=api_base, api_key="k"
        )
    )

    assert "".join(chunk.text for chunk in chunks) == STREAM_TEXT
    assert chunks[-1].finish_reason == "stop"


def test_embeddings_round_trip(api_base: str) -> None:
    vectors = litelm.embedding(
        "openai/text-embedding-3-small", ["a", "b"], api_base=api_base, api_key="k"
    )
    assert vectors == [[1.0, 2.0, 3.0]]


def test_http_errors_map_onto_the_hierarchy(api_base: str) -> None:
    with pytest.raises(litelm.RateLimitError) as excinfo:
        litelm.completion(
            "openai/echo-model", "rate limit me", api_base=api_base, api_key="k"
        )
    assert excinfo.value.status == 429
    assert "slow down" in excinfo.value.body


def test_unknown_paths_become_not_found_errors(api_base: str) -> None:
    # `completion` always appends /chat/completions, so hit the transport
    # directly to exercise the 404 mapping over a real socket.
    with pytest.raises(litelm.NotFoundError) as excinfo:
        litelm.transport.post_json(f"{api_base}/nope", {}, {})
    assert excinfo.value.status == 404


def test_the_tool_loop_runs_end_to_end(server: LoopbackServer, api_base: str) -> None:
    def get_weather(args: dict[str, Any]) -> str:
        return f"sunny and 22C in {args['location']}"

    tool = litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        get_weather,
    )

    response = litelm.chat_with_tools(
        "openai/echo-model",
        "What is the weather in Paris?",
        [tool],
        api_base=api_base,
        api_key="k",
    )

    assert response.content == "It is sunny and 22C in Paris."
    assert response.tool_calls == []

    # The second request must carry the assistant's tool call and the result.
    follow_up = server.requests[-1]["payload"]["messages"]
    assert follow_up[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert follow_up[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "sunny and 22C in Paris",
    }
