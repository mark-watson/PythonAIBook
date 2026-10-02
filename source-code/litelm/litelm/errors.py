"""Error hierarchy for litelm.

Mirrors the condition hierarchy of the Common Lisp ``litelm`` library (and its
Racket port)::

    LitelmError
    └── ApiError                      attributes: status, body
        ├── AuthenticationError       401, 403
        ├── RateLimitError            429
        ├── NotFoundError             404
        └── ContextWindowExceededError 400 whose body mentions "context"

Catch ``ApiError`` to handle every HTTP failure, or one of the subclasses to
react to a specific one::

    try:
        litelm.completion("openai/gpt-5.4-nano", "hi")
    except litelm.RateLimitError:
        ...          # back off and retry
    except litelm.AuthenticationError:
        ...          # bad or missing API key
"""

from __future__ import annotations


class LitelmError(Exception):
    """Base class for every error litelm raises."""


class ApiError(LitelmError):
    """An LLM provider answered with an HTTP error status."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"LLM API error {status}: {body}")


class AuthenticationError(ApiError):
    """401 or 403: the API key is missing, wrong, or not permitted."""


class RateLimitError(ApiError):
    """429: the provider is throttling us; back off before retrying."""


class NotFoundError(ApiError):
    """404: usually an unknown model name for that provider."""


class ContextWindowExceededError(ApiError):
    """400 naming the context window: the prompt no longer fits the model."""


def error_for_status(status: int, body: str) -> ApiError:
    """Return the ``ApiError`` subclass matching an HTTP ``status``."""
    if status in (401, 403):
        return AuthenticationError(status, body)
    if status == 429:
        return RateLimitError(status, body)
    if status == 404:
        return NotFoundError(status, body)
    if status == 400 and "context" in body.lower():
        return ContextWindowExceededError(status, body)
    return ApiError(status, body)
