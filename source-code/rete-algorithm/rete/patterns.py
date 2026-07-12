"""Pattern operators and condition elements (Cond) for rule LHS.

Operators
---------
``Var``, ``Eq``, ``Gt``, ``Lt``, ``Ge``, ``Le``, ``In``, ``Contains``,
``Match``, ``Test`` — each with a ``.test(value)`` predicate and an
optional ``.bind_name`` for variable capture.

``Cond`` represents a single compiled condition element (one row of the
LHS).  Build them with ``Pat(FactType, field=Op, ...)`` or negate with
``~cond``.
"""

from __future__ import annotations

import re
from typing import Any, Callable

__all__ = [
    "PatternOp",
    "Var",
    "Eq",
    "Gt",
    "Lt",
    "Ge",
    "Le",
    "In",
    "Contains",
    "Match",
    "Test",
    "Cond",
    "cond",
    "Pat",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PatternOp:
    """Abstract base for pattern operators."""

    @property
    def bind_name(self) -> str | None:
        """Variable name to bind, or None if this is a pure test."""
        return None

    def test(self, value: Any) -> bool:
        """Return True if *value* satisfies this operator."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class Var(PatternOp):
    """Bind a field to a named variable.

    ``Var("x")`` always passes (it captures, but does not filter).
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def bind_name(self) -> str:
        return self._name

    def test(self, value: Any) -> bool:  # noqa: D401
        return True

    def __repr__(self) -> str:
        return f"Var({self._name!r})"


class Eq(PatternOp):
    """Field must equal a constant."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return value == self._value

    def __repr__(self) -> str:
        return f"Eq({self._value!r})"


class Gt(PatternOp):
    """Field must be greater than a constant."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return value > self._value

    def __repr__(self) -> str:
        return f"Gt({self._value!r})"


class Lt(PatternOp):
    """Field must be less than a constant."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return value < self._value

    def __repr__(self) -> str:
        return f"Lt({self._value!r})"


class Ge(PatternOp):
    """Field must be greater than or equal to a constant."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return value >= self._value

    def __repr__(self) -> str:
        return f"Ge({self._value!r})"


class Le(PatternOp):
    """Field must be less than or equal to a constant."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return value <= self._value

    def __repr__(self) -> str:
        return f"Le({self._value!r})"


class In(PatternOp):
    """Field value must be a member of the given set."""

    def __init__(self, *values: Any) -> None:
        self._values = frozenset(values)

    def test(self, value: Any) -> bool:
        return value in self._values

    def __repr__(self) -> str:
        return f"In({', '.join(repr(v) for v in self._values)})"


class Contains(PatternOp):
    """Collection field must contain the given element."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def test(self, value: Any) -> bool:
        return self._value in value

    def __repr__(self) -> str:
        return f"Contains({self._value!r})"


class Match(PatternOp):
    """String field must match a regular expression."""

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._compiled = re.compile(pattern)

    def test(self, value: Any) -> bool:
        return bool(self._compiled.search(str(value)))

    def __repr__(self) -> str:
        return f"Match({self._pattern!r})"


class Test(PatternOp):
    """Arbitrary predicate on a field value."""

    def __init__(self, fn: Callable[[Any], bool]) -> None:
        self._fn = fn

    def test(self, value: Any) -> bool:
        return self._fn(value)

    def __repr__(self) -> str:
        return f"Test({self._fn!r})"


# ---------------------------------------------------------------------------
# Condition element
# ---------------------------------------------------------------------------


class Cond:
    """A compiled condition element for one row of a rule's LHS.

    Attributes
    ----------
    fact_type : type
        The fact class this condition matches against.
    field_constraints : dict[str, PatternOp]
        Mapping of field names to their operators.
    negated : bool
        If True, this is a Negated Condition Element (NCE): the
        condition succeeds when *no* matching WME exists.
    """

    __slots__ = ("fact_type", "field_constraints", "negated")

    def __init__(
        self,
        fact_type: type,
        field_constraints: dict[str, PatternOp],
        negated: bool = False,
    ) -> None:
        self.fact_type = fact_type
        self.field_constraints = field_constraints
        self.negated = negated

    def __invert__(self) -> Cond:
        """Return a negated copy of this condition."""
        return Cond(
            fact_type=self.fact_type,
            field_constraints=self.field_constraints,
            negated=not self.negated,
        )

    def var_bindings(self) -> dict[str, str]:
        """Return ``{variable_name: field_name}`` for all ``Var`` constraints."""
        return {
            op.bind_name: field
            for field, op in self.field_constraints.items()
            if isinstance(op, Var)
        }

    def __repr__(self) -> str:
        neg = "~" if self.negated else ""
        parts = ", ".join(f"{k}={v!r}" for k, v in self.field_constraints.items())
        return f"{neg}Cond({self.fact_type.__name__}, {parts})"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def cond(fact_type: type, negated: bool = False, **field_constraints: Any) -> Cond:
    """Create a Cond, auto-wrapping plain values in Eq."""
    wrapped = {
        k: v if isinstance(v, PatternOp) else Eq(v)
        for k, v in field_constraints.items()
    }
    return Cond(fact_type, wrapped, negated)


def Pat(fact_type: type, **constraints: Any) -> Cond:
    """Convenience factory for a positive Cond.

    Plain values are auto-wrapped in ``Eq``::

        Pat(Patient, name=Var("n"), temperature=Gt(38.0))
    """
    return cond(fact_type, negated=False, **constraints)
