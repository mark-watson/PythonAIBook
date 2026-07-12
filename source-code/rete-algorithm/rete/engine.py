"""ReteEngine and WorkingMemory — the public API for embedded expert systems.

Usage::

    from rete import ReteEngine, Fact, Pat, Var, Gt

    engine = ReteEngine()

    @engine.rule(Pat(MyFact, x=Var("x"), y=Gt(10)), salience=5)
    def my_rule(ctx, x):
        ctx.assert_fact(Result(value=x))

    engine.assert_fact(MyFact(x=1, y=20))
    engine.run()
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from collections.abc import Callable, Iterator
from typing import Any, TypeVar, overload

from .beta import Instantiation
from .conflict import resolve_strategy
from .context import RuleContext
from .facts import Fact, WME
from .network import ReteNetwork
from .patterns import Cond

_FactT = TypeVar("_FactT", bound=Fact)

__all__ = ["WorkingMemory", "ReteEngine"]


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Stores all asserted facts and routes changes through the Rete network."""

    def __init__(self, network: ReteNetwork) -> None:
        self._facts: dict[int, WME] = {}  # wme.id -> WME
        self._timestamp: int = 0
        self._network = network

    def assert_fact(self, fact: Fact) -> WME:
        """Assert a new fact into working memory.

        Creates a WME with a monotonically increasing timestamp, stores
        it, and pushes it into the Rete network's alpha side.
        """
        self._timestamp += 1
        wme = WME(fact, wme_id=self._timestamp, timestamp=self._timestamp)
        self._facts[wme.id] = wme
        self._network.add_wme(wme)
        return wme

    def retract(self, wme: WME) -> None:
        """Retract a WME from working memory and propagate removal."""
        if wme.id in self._facts:
            del self._facts[wme.id]
            self._network.remove_wme(wme)

    def modify(self, wme: WME, **changes: Any) -> WME:
        """Retract *wme*, then assert a modified copy.

        Uses :func:`dataclasses.replace` on the underlying fact.
        """
        new_fact = dataclasses.replace(wme.fact, **changes)
        self.retract(wme)
        return self.assert_fact(new_fact)

    @overload
    def facts(self, fact_type: type[_FactT]) -> Iterator[_FactT]: ...
    @overload
    def facts(self, fact_type: None = None) -> Iterator[Fact]: ...
    def facts(self, fact_type: type | None = None) -> Iterator[Fact]:
        """Iterate over current facts, optionally filtered by type."""
        for wme in self._facts.values():
            if fact_type is None or isinstance(wme.fact, fact_type):
                yield wme.fact

    def clear(self) -> None:
        """Retract every WME."""
        for wme in list(self._facts.values()):
            self.retract(wme)


# ---------------------------------------------------------------------------
# Rete Engine
# ---------------------------------------------------------------------------


class ReteEngine:
    """Top-level Rete engine — register rules, assert facts, run the cycle.

    Parameters
    ----------
    strategy : str | Callable
        Conflict resolution strategy.  Built-in names: ``"lex"`` (default),
        ``"mea"``, ``"priority-only"``.  Or pass a custom callable
        ``(list[Instantiation]) -> Instantiation``.
    """

    def __init__(self, strategy: str | Callable[..., Any] = "lex") -> None:
        self._conflict_set: list[Instantiation] = []
        self._fired: set[tuple[str, frozenset[int]]] = set()  # refraction
        self._strategy = resolve_strategy(strategy)
        self._network = ReteNetwork()
        self.wm = WorkingMemory(self._network)
        self._tracing: bool = False
        self._trace_output: list[str] = []

    # ------------------------------------------------------------------
    # Rule registration
    # ------------------------------------------------------------------

    def rule(
        self,
        *patterns: Cond,
        salience: int = 0,
        name: str | None = None,
    ) -> Callable[..., Callable[..., Any]]:
        """Decorator to register a rule.

        Each positional argument must be a :class:`Cond` (built via
        ``Pat(...)`` or negated with ``~Pat(...)``).

        The decorated function becomes the RHS action.  Its first
        parameter receives a :class:`RuleContext`; subsequent parameters
        are filled from the matched variable bindings.

        Example::

            @engine.rule(Pat(Patient, name=Var("n")), salience=10)
            def my_rule(ctx, n):
                ...
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            rule_name = name or fn.__name__
            self.add_rule(rule_name, list(patterns), fn, salience)
            return fn

        return decorator

    def add_rule(
        self,
        name: str,
        patterns: list[Cond],
        action: Callable[..., Any],
        salience: int = 0,
    ) -> None:
        """Programmatic rule registration."""
        self._network.add_rule(
            name=name,
            conds=patterns,
            action=action,
            salience=salience,
            conflict_set=self._conflict_set,
        )

    # ------------------------------------------------------------------
    # Working memory (delegates)
    # ------------------------------------------------------------------

    def assert_fact(self, fact: Fact) -> WME:
        """Assert a fact into working memory."""
        return self.wm.assert_fact(fact)

    def retract(self, wme: WME) -> None:
        """Retract a WME from working memory."""
        self.wm.retract(wme)

    def modify(self, wme: WME, **changes: Any) -> WME:
        """Modify a WME (retract + assert with changed fields)."""
        return self.wm.modify(wme, **changes)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, max_cycles: int | None = None) -> int:
        """Run the recognize-act cycle.

        Returns the number of rules fired.  Stops when:
        - The conflict set is empty.
        - ``ctx.halt()`` was called.
        - *max_cycles* reached (if given).
        """
        fired_count = 0
        halted = False

        while not halted:
            if max_cycles is not None and fired_count >= max_cycles:
                break
            did_fire, halted = self._step_internal()
            if not did_fire:
                break
            fired_count += 1

        return fired_count

    def step(self) -> bool:
        """Execute one recognize-act cycle.  Returns True if a rule fired."""
        did_fire, _ = self._step_internal()
        return did_fire

    def _step_internal(self) -> tuple[bool, bool]:
        """One cycle.  Returns (did_fire, halted)."""
        # Find eligible instantiations (refraction filter)
        eligible = [
            inst
            for inst in self._conflict_set
            if (inst.rule_name, inst.token.wme_ids()) not in self._fired
        ]

        if not eligible:
            return False, False

        # Conflict resolution
        chosen = self._strategy(eligible)

        # Mark as fired (refraction)
        wme_ids = chosen.token.wme_ids()
        self._fired.add((chosen.rule_name, wme_ids))

        if self._tracing:
            self._trace_output.append(f"FIRE: {chosen.rule_name} {chosen.token}")

        # Build RuleContext
        ctx = RuleContext(engine=self, token_wmes=chosen.token.wmes())

        # Extract action parameter names (skip 'ctx')
        sig = inspect.signature(chosen.action)
        params = list(sig.parameters.keys())
        kwargs: dict[str, Any] = {}
        bd = chosen.token.binding_dict
        for p in params[1:]:  # skip first param ('ctx')
            if p in bd:
                kwargs[p] = bd[p]

        # Execute RHS
        chosen.action(ctx, **kwargs)

        # Flush trace log
        for msg in ctx._trace_log:
            if self._tracing:
                self._trace_output.append(msg)
            print(msg, file=sys.stderr)

        # Apply deferred mutations
        ctx._apply(self)

        return True, ctx._halted

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @overload
    def facts(self, fact_type: type[_FactT]) -> Iterator[_FactT]: ...
    @overload
    def facts(self, fact_type: None = None) -> Iterator[Fact]: ...
    def facts(self, fact_type: type | None = None) -> Iterator[Fact]:
        """Iterate over current facts in working memory."""
        return self.wm.facts(fact_type)

    def conflict_set(self) -> list[Instantiation]:
        """Return the current conflict set (live reference)."""
        return list(self._conflict_set)

    def trace(self, enabled: bool = True) -> None:
        """Enable or disable execution tracing."""
        self._tracing = enabled

    def get_trace(self) -> list[str]:
        """Return the accumulated trace log."""
        return list(self._trace_output)
