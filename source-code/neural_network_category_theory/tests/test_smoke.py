"""Smoke tests for the categorical NN reference implementation.

Imports the module (which guards its demo under `if __name__ == "__main__":`)
and exercises the small pure math helpers used everywhere else in the file.
"""

import math

import pytest

import neural_network_category_theory as nnct


def test_sigmoid_at_zero() -> None:
    assert nnct.sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_deriv_from_output() -> None:
    a = nnct.sigmoid(0.5)
    assert nnct.sigmoid_deriv(a) == pytest.approx(a * (1 - a))


def test_dot_matches_hypot_squared() -> None:
    v = [3.0, 4.0]
    assert nnct.dot(v, v) == pytest.approx(math.hypot(*v) ** 2)
