"""Unit tests for the pure-Python math helpers.

These are the primitive building blocks the rest of the categorical framework
composes on top of, so if any of them regresses the whole file breaks.
"""

import math

import pytest

from deep_learning_category_theory import (
    dot,
    matT_vec,
    matvec,
    outer,
    relu,
    relu_deriv,
    scalar_vec_mul,
    sigmoid,
    sigmoid_deriv,
    transpose,
    vec_add,
)


def test_sigmoid_at_zero() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_saturates() -> None:
    # The book implementation is `1 / (1 + exp(-z))`, which overflows past |z| ≈ 700.
    # Use values well inside the safe range so we're testing saturation, not numerics.
    assert sigmoid(50.0) == pytest.approx(1.0)
    assert sigmoid(-50.0) == pytest.approx(0.0)


def test_sigmoid_deriv_from_output() -> None:
    a = sigmoid(0.5)
    assert sigmoid_deriv(a) == pytest.approx(a * (1 - a))


def test_relu_and_deriv() -> None:
    assert relu(-3.0) == 0.0
    assert relu(0.0) == 0.0
    assert relu(2.5) == 2.5
    assert relu_deriv(-1.0) == 0.0
    assert relu_deriv(0.0) == 0.0
    assert relu_deriv(1.0) == 1.0


def test_dot_matches_math_hypot_squared() -> None:
    v = [3.0, 4.0]
    assert dot(v, v) == pytest.approx(math.hypot(*v) ** 2)


def test_matvec_identity() -> None:
    identity = [[1.0, 0.0], [0.0, 1.0]]
    assert matvec(identity, [7.0, -2.0]) == [7.0, -2.0]


def test_vec_add_and_scalar_mul() -> None:
    assert vec_add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
    assert scalar_vec_mul(2.5, [2.0, -4.0]) == [5.0, -10.0]


def test_outer_and_transpose_roundtrip() -> None:
    delta = [1.0, 2.0]
    x = [3.0, 4.0, 5.0]
    op = outer(delta, x)
    assert op == [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]
    assert transpose(transpose(op)) == op


def test_matT_vec_matches_manual_transpose_product() -> None:
    m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # shape (3, 2)
    v = [1.0, 1.0, 1.0]
    # Mᵀ · v where Mᵀ has shape (2, 3): row sums of columns of M.
    assert matT_vec(m, v) == [1.0 + 3.0 + 5.0, 2.0 + 4.0 + 6.0]
