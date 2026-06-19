"""Beta network — join nodes, negative nodes, production nodes.

The beta network performs inter-element joins (variable consistency
checks) and terminates at production nodes that feed the conflict set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .alpha import AlphaMemory
from .facts import WME
from .patterns import Cond, Var
from .tokens import EMPTY_TOKEN, Token

__all__ = [
    "JoinTest",
    "BetaMemory",
    "JoinNode",
    "NegativeNode",
    "ProductionNode",
    "Instantiation",
]


def is_descendant(t: Token, token: Token) -> bool:
    """Return True if *t* is a descendant of *token* (or is *token* itself)."""
    curr: Token | None = t
    while curr is not None:
        if curr is token:
            return True
        curr = curr.parent
    return False


# ---------------------------------------------------------------------------
# Join test descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JoinTest:
    """Describes one consistency check between a token and a WME.

    The join succeeds when
    ``token.binding_dict[left_field] == getattr(wme.fact, right_field)``.
    """

    left_field: str   # variable name already bound in the token
    right_field: str  # field name on the new WME's fact


# ---------------------------------------------------------------------------
# Beta memory
# ---------------------------------------------------------------------------

class BetaMemory:
    """Stores tokens (partial matches) and propagates to children."""

    def __init__(self) -> None:
        self.tokens: list[Token] = []
        self.children: list[Any] = []  # JoinNode | NegativeNode

    def left_activate(self, token: Token) -> None:
        """Store *token* and propagate to child join nodes."""
        self.tokens.append(token)
        for child in self.children:
            child.left_activate(token)

    def left_remove(self, wme: WME) -> None:
        """Remove all tokens containing *wme* and propagate removals."""
        surviving: list[Token] = []
        removed: list[Token] = []
        for t in self.tokens:
            if wme in set(t.wmes()):
                removed.append(t)
            else:
                surviving.append(t)
        self.tokens = surviving
        for t in removed:
            for child in self.children:
                child.left_remove_token(t)

    def left_remove_token(self, token: Token) -> None:
        """Remove all tokens that are descendants of *token* and propagate."""
        surviving: list[Token] = []
        removed: list[Token] = []
        for t in self.tokens:
            if is_descendant(t, token):
                removed.append(t)
            else:
                surviving.append(t)
        self.tokens = surviving
        for child in self.children:
            child.left_remove_token(token)

    def __repr__(self) -> str:
        return f"BetaMemory({len(self.tokens)} tokens)"


# ---------------------------------------------------------------------------
# Join node
# ---------------------------------------------------------------------------

class JoinNode:
    """Performs an inter-element join between a beta input and an alpha memory.

    Left activation = new token from beta side.
    Right activation = new WME from alpha side.
    """

    def __init__(
        self,
        alpha_memory: AlphaMemory,
        tests: list[JoinTest],
        cond: Cond,
        parent: BetaMemory | None = None,
    ) -> None:
        self.alpha_memory = alpha_memory
        self.tests = tests
        self.cond = cond
        self.parent = parent
        self.children: list[Any] = []  # BetaMemory | ProductionNode | JoinNode

    def left_activate(self, token: Token) -> None:
        """A new token arrived from the beta (left) side."""
        for wme in self.alpha_memory.wmes:
            bindings = self._try_join(token, wme)
            if bindings is not None:
                new_token = token.extend(wme, bindings)
                for child in self.children:
                    if isinstance(child, BetaMemory):
                        child.left_activate(new_token)
                    elif isinstance(child, ProductionNode):
                        child.left_activate(new_token)
                    elif isinstance(child, JoinNode):
                        child.left_activate(new_token)
                    elif isinstance(child, NegativeNode):
                        child.left_activate(new_token)

    def right_activate(self, wme: WME) -> None:
        """A new WME arrived from the alpha (right) side."""
        parent_tokens = self.parent.tokens if self.parent else [EMPTY_TOKEN]
        for token in parent_tokens:
            bindings = self._try_join(token, wme)
            if bindings is not None:
                new_token = token.extend(wme, bindings)
                for child in self.children:
                    if isinstance(child, BetaMemory):
                        child.left_activate(new_token)
                    elif isinstance(child, ProductionNode):
                        child.left_activate(new_token)
                    elif isinstance(child, JoinNode):
                        child.left_activate(new_token)
                    elif isinstance(child, NegativeNode):
                        child.left_activate(new_token)

    def right_remove(self, wme: WME) -> None:
        """A WME was retracted from the alpha memory — remove dependent tokens."""
        for child in self.children:
            if isinstance(child, BetaMemory):
                child.left_remove(wme)
            elif isinstance(child, ProductionNode):
                child.left_remove(wme)
            elif isinstance(child, JoinNode):
                child._propagate_remove(wme)
            elif isinstance(child, NegativeNode):
                child._handle_wme_removal(wme)

    def _propagate_remove(self, wme: WME) -> None:
        """Propagate a WME removal downstream (called on intermediate JoinNodes)."""
        for child in self.children:
            if isinstance(child, BetaMemory):
                child.left_remove(wme)
            elif isinstance(child, ProductionNode):
                child.left_remove(wme)
            elif isinstance(child, JoinNode):
                child._propagate_remove(wme)
            elif isinstance(child, NegativeNode):
                child._handle_wme_removal(wme)

    def left_remove_token(self, token: Token) -> None:
        """Remove a specific token from downstream nodes."""
        for child in self.children:
            child.left_remove_token(token)

    def _try_join(self, token: Token, wme: WME) -> dict[str, Any] | None:
        """Try to join *token* with *wme*.

        Returns the new variable bindings to add, or ``None`` if the
        join fails.
        """
        bd = token.binding_dict

        # Check all explicit join tests (shared-variable consistency)
        for jt in self.tests:
            if jt.left_field in bd:
                wme_val = getattr(wme.fact, jt.right_field)
                if bd[jt.left_field] != wme_val:
                    return None

        # Check intra-element tests on the condition's field constraints
        for field_name, op in self.cond.field_constraints.items():
            if isinstance(op, Var):
                continue  # Var is just a binding, test always passes
            value = getattr(wme.fact, field_name)
            if not op.test(value):
                return None

        # Extract new variable bindings from Var operators
        new_bindings: dict[str, Any] = {}
        for field_name, op in self.cond.field_constraints.items():
            if isinstance(op, Var):
                val = getattr(wme.fact, field_name)
                var_name = op.bind_name
                # Consistency: if already bound, check match
                if var_name in bd and bd[var_name] != val:
                    return None
                new_bindings[var_name] = val

        return new_bindings

    def __repr__(self) -> str:
        return f"JoinNode({self.cond!r}, tests={self.tests})"


# ---------------------------------------------------------------------------
# Negative node
# ---------------------------------------------------------------------------

class NegativeNode:
    """Implements negated condition elements (NCEs).

    A token passes through only if NO WME in the alpha memory satisfies
    the join.  The node tracks which WMEs block each token; when a
    blocking WME is retracted, the token is re-evaluated and may be
    released downstream.
    """

    def __init__(
        self,
        alpha_memory: AlphaMemory,
        tests: list[JoinTest],
        cond: Cond,
        parent: BetaMemory | None = None,
    ) -> None:
        self.alpha_memory = alpha_memory
        self.tests = tests
        self.cond = cond
        self.parent = parent
        self.children: list[Any] = []
        # token -> set of blocking WMEs
        self._blocked: dict[int, tuple[Token, set[WME]]] = {}
        # tokens that are currently unblocked (passed through)
        self._passed: dict[int, Token] = {}

    def left_activate(self, token: Token) -> None:
        """A new token arrived.  Check if any WME blocks it."""
        blockers = set()
        bd = token.binding_dict
        for wme in self.alpha_memory.wmes:
            if self._matches(bd, wme):
                blockers.add(wme)

        tok_id = id(token)
        if blockers:
            self._blocked[tok_id] = (token, blockers)
        else:
            self._passed[tok_id] = token
            for child in self.children:
                if isinstance(child, BetaMemory):
                    child.left_activate(token)
                elif isinstance(child, ProductionNode):
                    child.left_activate(token)
                elif isinstance(child, JoinNode):
                    child.left_activate(token)

    def right_activate(self, wme: WME) -> None:
        """A new WME appeared — it may block currently-passing tokens."""
        # Check passed tokens: if the new WME blocks any, retract them
        newly_blocked: list[int] = []
        for tok_id, token in list(self._passed.items()):
            bd = token.binding_dict
            if self._matches(bd, wme):
                newly_blocked.append(tok_id)

        for tok_id in newly_blocked:
            token = self._passed.pop(tok_id)
            # Remove from downstream
            for child in self.children:
                child.left_remove_token(token)
            # Track as blocked
            self._blocked[tok_id] = (token, {wme})

        # Also add WME to existing blockers
        for tok_id, (token, blockers) in list(self._blocked.items()):
            if tok_id not in [nb for nb in newly_blocked]:
                bd = token.binding_dict
                if self._matches(bd, wme):
                    blockers.add(wme)

    def right_remove(self, wme: WME) -> None:
        """A WME was retracted — unblock tokens it was blocking."""
        self._handle_wme_removal(wme)

    def _handle_wme_removal(self, wme: WME) -> None:
        """Handle removal of a WME that may have been blocking tokens."""
        newly_unblocked: list[int] = []
        for tok_id, (token, blockers) in list(self._blocked.items()):
            blockers.discard(wme)
            if not blockers:
                newly_unblocked.append(tok_id)

        for tok_id in newly_unblocked:
            token, _ = self._blocked.pop(tok_id)
            self._passed[tok_id] = token
            for child in self.children:
                if isinstance(child, BetaMemory):
                    child.left_activate(token)
                elif isinstance(child, ProductionNode):
                    child.left_activate(token)
                elif isinstance(child, JoinNode):
                    child.left_activate(token)

    def left_remove_token(self, token: Token) -> None:
        """Remove a specific token from tracking and propagate downstream."""
        to_remove_blocked = [k for k, (t, _) in self._blocked.items() if is_descendant(t, token)]
        for k in to_remove_blocked:
            del self._blocked[k]
        to_remove_passed = [k for k, t in self._passed.items() if is_descendant(t, token)]
        for k in to_remove_passed:
            del self._passed[k]
        for child in self.children:
            child.left_remove_token(token)

    def _matches(self, bd: dict[str, Any], wme: WME) -> bool:
        """Check if *wme* satisfies the join tests and intra-element tests."""
        for jt in self.tests:
            if jt.left_field in bd:
                if bd[jt.left_field] != getattr(wme.fact, jt.right_field):
                    return False
        for field_name, op in self.cond.field_constraints.items():
            if isinstance(op, Var):
                var_name = op.bind_name
                if var_name in bd and bd[var_name] != getattr(wme.fact, field_name):
                    return False
                continue
            if not op.test(getattr(wme.fact, field_name)):
                return False
        return True


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

@dataclass
class Instantiation:
    """A complete rule match ready to fire.

    Attributes
    ----------
    rule_name : str
        Name of the matched rule.
    token : Token
        The complete token chain (all matching WMEs + bindings).
    salience : int
        Rule priority (higher = fires first).
    action : Callable
        The RHS function to execute.
    conds : list[Cond]
        The rule's condition elements (for specificity).
    timestamp : int
        Maximum WME timestamp in the token (for recency ordering).
    """

    rule_name: str
    token: Token
    salience: int
    action: Callable
    conds: list[Cond]
    timestamp: int


# ---------------------------------------------------------------------------
# Production node
# ---------------------------------------------------------------------------

class ProductionNode:
    """Terminal beta node — manages instantiations in the conflict set."""

    def __init__(
        self,
        rule_name: str,
        salience: int,
        action: Callable,
        conds: list[Cond],
        conflict_set: list[Instantiation],
    ) -> None:
        self.rule_name = rule_name
        self.salience = salience
        self.action = action
        self.conds = conds
        self.conflict_set = conflict_set
        # token identity -> Instantiation
        self.instantiations: dict[int, Instantiation] = {}

    def left_activate(self, token: Token) -> None:
        """A complete match arrived — add to conflict set."""
        wmes = token.wmes()
        ts = max((w.timestamp for w in wmes), default=0)
        inst = Instantiation(
            rule_name=self.rule_name,
            token=token,
            salience=self.salience,
            action=self.action,
            conds=self.conds,
            timestamp=ts,
        )
        tok_id = id(token)
        self.instantiations[tok_id] = inst
        self.conflict_set.append(inst)

    def left_remove(self, wme: WME) -> None:
        """Remove all instantiations whose tokens contain *wme*."""
        to_remove: list[int] = []
        for tok_id, inst in self.instantiations.items():
            if wme in set(inst.token.wmes()):
                to_remove.append(tok_id)
        for tok_id in to_remove:
            inst = self.instantiations.pop(tok_id)
            try:
                self.conflict_set.remove(inst)
            except ValueError:
                pass

    def left_remove_token(self, token: Token) -> None:
        """Remove all instantiations whose tokens are descendants of *token*."""
        to_remove = [
            tok_id for tok_id, inst in self.instantiations.items()
            if is_descendant(inst.token, token)
        ]
        for tok_id in to_remove:
            inst = self.instantiations.pop(tok_id)
            try:
                self.conflict_set.remove(inst)
            except ValueError:
                pass

    def __repr__(self) -> str:
        return f"ProductionNode({self.rule_name!r}, {len(self.instantiations)} insts)"
