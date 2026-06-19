"""Token — a partial match flowing through the beta network.

A Token is an immutable chain: each node holds one matched WME and a
cumulative variable-binding environment.  Tokens share prefixes via
parent pointers so the same partial match is never duplicated.
"""

from __future__ import annotations

from typing import Any

from .facts import WME

__all__ = ["Token", "EMPTY_TOKEN"]

Bindings = dict[str, Any]


class Token:
    """An immutable partial-match record in the Rete beta network.

    Attributes
    ----------
    parent : Token | None
        The preceding token in the chain (``None`` for the root).
    wme : WME | None
        The WME matched at this level (``None`` only for the root).
    bindings : frozenset[tuple[str, Any]]
        Variable bindings accumulated up to this point, stored as a
        frozenset of ``(name, value)`` pairs for hashability.
    """

    __slots__ = ("parent", "wme", "bindings")

    def __init__(
        self,
        parent: Token | None,
        wme: WME | None,
        bindings: frozenset[tuple[str, Any]] | None = None,
    ) -> None:
        self.parent = parent
        self.wme = wme
        self.bindings: frozenset[tuple[str, Any]] = bindings or frozenset()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def binding_dict(self) -> dict[str, Any]:
        """Return the bindings as a plain dictionary."""
        return dict(self.bindings)

    def wmes(self) -> list[WME]:
        """Collect all WMEs in the chain (root-first order)."""
        result: list[WME] = []
        tok: Token | None = self
        while tok is not None:
            if tok.wme is not None:
                result.append(tok.wme)
            tok = tok.parent
        result.reverse()
        return result

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def extend(self, wme: WME, new_bindings: dict[str, Any]) -> Token:
        """Create a child token adding *wme* and extra *new_bindings*."""
        merged = set(self.bindings)
        merged.update(new_bindings.items())
        return Token(parent=self, wme=wme, bindings=frozenset(merged))

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    def wme_ids(self) -> frozenset[int]:
        """Return the set of WME ids in this token chain."""
        return frozenset(w.id for w in self.wmes())

    def __repr__(self) -> str:
        bd = self.binding_dict
        wme_ids = [w.id for w in self.wmes()]
        return f"Token(wmes={wme_ids}, bindings={bd})"

    def __hash__(self) -> int:
        return hash((id(self.parent), self.wme, self.bindings))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return (
            self.parent is other.parent
            and self.wme is other.wme
            and self.bindings == other.bindings
        )


# Sentinel root token — used as the left input seed for the first join.
EMPTY_TOKEN = Token(parent=None, wme=None, bindings=frozenset())
