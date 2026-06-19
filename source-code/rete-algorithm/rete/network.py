"""ReteNetwork — compiles rules into the alpha + beta network.

The builder creates or reuses alpha test chains and alpha memories for
each condition, then wires up the beta network (join nodes, negative
nodes) and attaches a production node at the end.
"""

from __future__ import annotations

from typing import Any, Callable

from .alpha import AlphaMemory, AlphaNetwork, AlphaTestNode
from .beta import (
    BetaMemory,
    Instantiation,
    JoinNode,
    JoinTest,
    NegativeNode,
    ProductionNode,
)
from .facts import WME
from .patterns import Cond, Var

__all__ = ["ReteNetwork"]


class ReteNetwork:
    """Builds and manages the combined alpha + beta Rete network.

    Call :meth:`add_rule` to compile a rule's conditions into network
    nodes (reusing shared prefixes), then :meth:`add_wme`/:meth:`remove_wme`
    to drive facts through the network.
    """

    def __init__(self) -> None:
        self.alpha_network = AlphaNetwork()
        # Cache: (fact_type, frozenset of (field, repr(op))) -> AlphaMemory
        self._alpha_memory_cache: dict[tuple, AlphaMemory] = {}

    # ------------------------------------------------------------------
    # Rule compilation
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        conds: list[Cond],
        action: Callable,
        salience: int,
        conflict_set: list[Instantiation],
    ) -> None:
        """Compile a rule into the network.

        Parameters
        ----------
        name : str
            Rule name (for tracing / conflict set display).
        conds : list[Cond]
            Ordered condition elements (positive or negated).
        action : Callable
            The RHS function to execute on firing.
        salience : int
            Priority for conflict resolution.
        conflict_set : list[Instantiation]
            The engine's shared conflict set (production nodes append to it).
        """
        if not conds:
            raise ValueError(f"Rule {name!r} has no conditions")

        # --- build alpha memories for each condition ---
        alpha_memories: list[AlphaMemory] = []
        for c in conds:
            am = self._get_or_build_alpha(c)
            alpha_memories.append(am)

        # --- build beta network chain ---
        # Track which variables have been bound in previous conditions
        seen_vars: dict[str, str] = {}  # var_name -> field_name (from the cond that first bound it)
        prev_beta: BetaMemory | None = None

        for i, c in enumerate(conds):
            am = alpha_memories[i]

            # Compute join tests: variables already seen that also appear in this cond
            join_tests = self._compute_join_tests(c, seen_vars)

            if c.negated:
                node = NegativeNode(
                    alpha_memory=am,
                    tests=join_tests,
                    cond=c,
                    parent=prev_beta,
                )
                am.successors.append(node)
                if prev_beta is not None:
                    prev_beta.children.append(node)
            else:
                node = JoinNode(
                    alpha_memory=am,
                    tests=join_tests,
                    cond=c,
                    parent=prev_beta,
                )
                am.successors.append(node)
                if prev_beta is not None:
                    prev_beta.children.append(node)

            # Record new Var bindings from this condition
            for field_name, op in c.field_constraints.items():
                if isinstance(op, Var) and op.bind_name not in seen_vars:
                    seen_vars[op.bind_name] = field_name

            # Is this the last condition?
            if i == len(conds) - 1:
                # Attach production node directly
                pnode = ProductionNode(
                    rule_name=name,
                    salience=salience,
                    action=action,
                    conds=conds,
                    conflict_set=conflict_set,
                )
                node.children.append(pnode)
            else:
                # Add a BetaMemory between conditions
                bm = BetaMemory()
                node.children.append(bm)
                prev_beta = bm

    # ------------------------------------------------------------------
    # WME routing
    # ------------------------------------------------------------------

    def add_wme(self, wme: WME) -> None:
        """Push a WME into the alpha network."""
        self.alpha_network.add_wme(wme)

    def remove_wme(self, wme: WME) -> None:
        """Remove a WME from the alpha network (triggers retraction)."""
        self.alpha_network.remove_wme(wme)

    # ------------------------------------------------------------------
    # Alpha network construction
    # ------------------------------------------------------------------

    def _get_or_build_alpha(self, c: Cond) -> AlphaMemory:
        """Get or create an AlphaMemory for the given condition.

        Conditions with the same fact type and the same set of non-Var
        tests share an alpha memory (node sharing).
        """
        # Build a cache key from the non-Var constraints
        test_key_parts: list[tuple[str, str]] = []
        for field, op in sorted(c.field_constraints.items()):
            if not isinstance(op, Var):
                test_key_parts.append((field, repr(op)))
        cache_key = (c.fact_type, frozenset(test_key_parts))

        if cache_key in self._alpha_memory_cache:
            return self._alpha_memory_cache[cache_key]

        # Build the alpha test chain
        non_var_constraints = [
            (field, op) for field, op in c.field_constraints.items()
            if not isinstance(op, Var)
        ]

        am = AlphaMemory()

        if not non_var_constraints:
            # No tests — every WME of this type passes directly
            self.alpha_network.register_entry(c.fact_type, am)
        else:
            # Build chain: first test -> second test -> ... -> AlphaMemory
            first_node: AlphaTestNode | None = None
            prev_node: AlphaTestNode | None = None
            for field, op in non_var_constraints:
                test_node = AlphaTestNode(field_name=field, test_fn=op.test)
                if first_node is None:
                    first_node = test_node
                if prev_node is not None:
                    prev_node.children.append(test_node)
                prev_node = test_node
            # Terminal node points to the alpha memory
            assert prev_node is not None
            prev_node.children.append(am)
            assert first_node is not None
            self.alpha_network.register_entry(c.fact_type, first_node)

        self._alpha_memory_cache[cache_key] = am
        return am

    @staticmethod
    def _compute_join_tests(
        c: Cond,
        seen_vars: dict[str, str],
    ) -> list[JoinTest]:
        """Compute join tests for condition *c* against previously bound vars.

        A JoinTest is emitted whenever a Var in *c* refers to a variable
        that was already bound by an earlier condition.
        """
        tests: list[JoinTest] = []
        for field_name, op in c.field_constraints.items():
            if isinstance(op, Var):
                var_name = op.bind_name
                if var_name in seen_vars:
                    tests.append(JoinTest(
                        left_field=var_name,
                        right_field=field_name,
                    ))
        return tests
