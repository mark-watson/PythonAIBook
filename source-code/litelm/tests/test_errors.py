"""The HTTP status -> exception mapping."""

from __future__ import annotations

import pytest

import litelm
from litelm.errors import error_for_status


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, litelm.AuthenticationError),
        (403, litelm.AuthenticationError),
        (429, litelm.RateLimitError),
        (404, litelm.NotFoundError),
        (400, litelm.ApiError),
        (500, litelm.ApiError),
    ],
)
def test_status_mapping(status: int, expected: type[litelm.ApiError]) -> None:
    error = error_for_status(status, "boom")
    assert type(error) is expected
    assert error.status == status
    assert error.body == "boom"


def test_context_window_is_detected_case_insensitively() -> None:
    error = error_for_status(400, "This model's maximum Context Length is 8192")
    assert isinstance(error, litelm.ContextWindowExceededError)


def test_every_api_error_is_a_litelm_error() -> None:
    for cls in (
        litelm.ApiError,
        litelm.AuthenticationError,
        litelm.RateLimitError,
        litelm.NotFoundError,
        litelm.ContextWindowExceededError,
    ):
        assert issubclass(cls, litelm.LitelmError)


def test_api_error_message_carries_status_and_body() -> None:
    error = litelm.ApiError(500, "server on fire")
    assert "500" in str(error)
    assert "server on fire" in str(error)
