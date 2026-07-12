"""Working Memory Element (WME) and Fact base class.

Facts are user-defined frozen dataclasses that inherit from ``Fact``.
A ``WME`` wraps a fact with a unique integer id and an integer timestamp
so the engine can track assertion order (for recency-based conflict
resolution) and efficiently hash/compare elements.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    """Base marker class for user-defined facts.

    User fact types should inherit from this and be frozen dataclasses::

        @dataclass(frozen=True)
        class Patient(Fact):
            name: str
            temperature: float
            symptoms: tuple[str, ...] = ()
    """


# ---------------------------------------------------------------------------
# WME id generation
# ---------------------------------------------------------------------------


class _WMEIdGenerator:
    """Thread-unsafe monotonic counter for assigning unique WME ids.

    Each call to ``next()`` returns a strictly increasing integer starting
    from 1.  The generator can be reset (mainly useful in tests).
    """

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def next(self) -> int:
        """Return the next unique id."""
        return next(self._counter)

    def reset(self) -> None:
        """Reset the counter back to 1."""
        self._counter = itertools.count(1)


# Module-level singleton used by ``WME.create``.
_wme_id_gen = _WMEIdGenerator()


# ---------------------------------------------------------------------------
# WME
# ---------------------------------------------------------------------------


class WME:
    """Working Memory Element — an immutable wrapper around a ``Fact``.

    Attributes:
        fact: The underlying user fact (a frozen dataclass).
        id:   A unique integer assigned at creation time.
        timestamp: An integer representing assertion order (higher = more
                   recent).  The engine typically uses the same counter as
                   *id*, but callers may supply an explicit value.

    Hashing and equality are based solely on *id* so that two ``WME``
    objects wrapping identical facts but asserted at different times are
    considered distinct.
    """

    __slots__ = ("fact", "id", "timestamp")

    def __init__(self, fact: Fact, wme_id: int, timestamp: int) -> None:
        object.__setattr__(self, "fact", fact)
        object.__setattr__(self, "id", wme_id)
        object.__setattr__(self, "timestamp", timestamp)

    # --- immutability -------------------------------------------------------

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("WME instances are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("WME instances are immutable")

    # --- hashing / equality by id -------------------------------------------

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WME):
            return NotImplemented
        return self.id == other.id

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"WME(id={self.id}, ts={self.timestamp}, fact={self.fact!r})"

    # --- factory ------------------------------------------------------------

    @classmethod
    def create(cls, fact: Fact, timestamp: int | None = None) -> WME:
        """Create a new WME with an auto-assigned unique id.

        If *timestamp* is ``None`` the id value is reused as the timestamp
        (so id order ≡ assertion order).
        """
        wme_id = _wme_id_gen.next()
        if timestamp is None:
            timestamp = wme_id
        return cls(fact, wme_id, timestamp)


def reset_wme_ids() -> None:
    """Reset the global WME id counter (useful in tests)."""
    _wme_id_gen.reset()
