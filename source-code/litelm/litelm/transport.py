"""HTTP transport: the only place litelm touches the network.

Both entry points are deliberately tiny and module level so tests can replace
them without a server::

    monkeypatch.setattr(litelm.transport, "post_json", fake_post_json)

Any HTTP failure is mapped onto the :mod:`litelm.errors` hierarchy, so callers
never see ``urllib`` exceptions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .errors import LitelmError, error_for_status

__all__ = ["DEFAULT_TIMEOUT", "post_json", "post_sse"]

#: Seconds to wait for a response. Generous on purpose: the local reasoning
#: models this book uses (DeepSeek-R1, Qwen with thinking enabled) can spend
#: minutes inside one non-streamed answer, and a timeout is worse than a wait.
#: The Common Lisp and Racket ports default to 120s; pass ``timeout=`` to
#: tighten this per call.
DEFAULT_TIMEOUT = 300.0

_SSE_DATA = "data:"


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON object."""
    request = _request(url, headers, payload)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise error_for_status(exc.code, _body_of(exc)) from exc
    except urllib.error.URLError as exc:
        raise LitelmError(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise _timeout_error(url, timeout) from exc
    except OSError as exc:
        raise LitelmError(f"Could not reach {url}: {exc}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LitelmError(f"{url} returned a non-JSON body: {body[:500]}") from exc
    if not isinstance(decoded, dict):
        raise LitelmError(f"{url} returned JSON that is not an object: {body[:500]}")
    return decoded


def post_sse(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[dict[str, Any]]:
    """POST ``payload`` and yield each decoded server-sent event object.

    Stops at the ``data: [DONE]`` sentinel every OpenAI-compatible provider
    sends at the end of a stream.
    """
    request = _request(url, headers, payload)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise error_for_status(exc.code, _body_of(exc)) from exc
    except urllib.error.URLError as exc:
        raise LitelmError(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise _timeout_error(url, timeout) from exc
    except OSError as exc:
        raise LitelmError(f"Could not reach {url}: {exc}") from exc

    with response:
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith(_SSE_DATA):
                    continue
                data = line[len(_SSE_DATA) :].strip()
                if data == "[DONE]":
                    return
                if not data:
                    # An empty data field is an empty event, not end-of-stream.
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
        except urllib.error.URLError as exc:
            raise LitelmError(f"Stream from {url} failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _timeout_error(url, timeout) from exc
        except OSError as exc:
            raise LitelmError(f"Stream from {url} failed: {exc}") from exc


def _request(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> urllib.request.Request:
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=data, headers=dict(headers), method="POST")


def _body_of(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _timeout_error(url: str, timeout: float) -> LitelmError:
    """A socket timeout is a LitelmError too -- local models are slow.

    Pass a larger ``timeout=`` for a big local model, or a smaller one to fail
    fast; the default is comfortably above a cloud API's latency.
    """
    return LitelmError(f"{url} timed out after {timeout:g}s")
