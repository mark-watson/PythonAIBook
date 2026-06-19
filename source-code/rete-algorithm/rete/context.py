"""RuleContext — the handle passed to rule RHS action functions.

All working-memory mutations requested during a rule's RHS execution are
**deferred** into internal queues.  The engine calls :meth:`_apply` after
the action returns to replay those mutations in order, which avoids
re-entrant network updates and matches OPS5 semantics.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .facts import Fact, WME

__all__ = ["RuleContext"]


class RuleContext:
    """Scoped working-memory handle for rule RHS functions.

    Parameters
    ----------
    engine : ReteEngine
        The engine instance (used for ``modify`` and deferred application).
    token_wmes : list[WME]
        The WMEs that matched this rule's LHS, exposed for introspection.
    """

    __slots__ = (
        "_engine",
        "_pending_asserts",
        "_pending_retracts",
        "_halted",
        "_trace_log",
        "token_wmes",
    )

    def __init__(self, engine: Any, token_wmes: list | None = None) -> None:
        self._engine = engine
        self._pending_asserts: list[Fact] = []
        self._pending_retracts: list[WME] = []
        self._halted: bool = False
        self._trace_log: list[str] = []
        self.token_wmes: list[WME] = token_wmes or []

    # ------------------------------------------------------------------
    # Public API (called from inside RHS actions)
    # ------------------------------------------------------------------

    def assert_fact(self, fact: Fact) -> None:
        """Queue a new fact for assertion after the current RHS completes."""
        self._pending_asserts.append(fact)

    def retract(self, wme: WME) -> None:
        """Queue a WME for retraction after the current RHS completes."""
        self._pending_retracts.append(wme)

    def modify(self, wme: WME, **changes: Any) -> None:
        """Queue a modify (retract old + assert modified copy).

        The modified fact is constructed via :func:`dataclasses.replace` on
        the WME's underlying fact, so *changes* must be valid field names
        for that fact type.
        """
        new_fact = dataclasses.replace(wme.fact, **changes)
        self._pending_retracts.append(wme)
        self._pending_asserts.append(new_fact)

    def halt(self) -> None:
        """Signal the engine to stop the recognize-act cycle."""
        self._halted = True

    def print(self, msg: str) -> None:
        """Append *msg* to the engine's trace log."""
        self._trace_log.append(msg)

    # ------------------------------------------------------------------
    # Internal — called by the engine after the action function returns
    # ------------------------------------------------------------------

    def _apply(self, engine: Any) -> None:
        """Replay all deferred mutations against *engine*'s working memory.

        Retractions are applied first (so that modify = retract + assert
        does not momentarily duplicate a fact).
        """
        for wme in self._pending_retracts:
            engine.retract(wme)
        for fact in self._pending_asserts:
            engine.assert_fact(fact)
