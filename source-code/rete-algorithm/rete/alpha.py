"""Alpha network — type dispatch, intra-element tests, and alpha memories.

The alpha network is the first half of the Rete network.  A new WME enters
at the ``AlphaNetwork`` root, is dispatched by fact type, passes through a
chain of ``AlphaTestNode`` gates (one per constant/predicate test on a
single field), and lands in an ``AlphaMemory`` if all tests pass.

Alpha memories feed into the beta network's join nodes.
"""

from __future__ import annotations

from typing import Any, Callable

from .facts import WME

__all__ = ["AlphaMemory", "AlphaTestNode", "AlphaNetwork"]


class AlphaMemory:
    """Stores WMEs that passed all alpha tests for one condition.

    Each alpha memory corresponds to one condition element's
    ``(fact_type, constant_tests)`` signature.  When a WME is added,
    all successor join/negative nodes are right-activated.
    """

    def __init__(self) -> None:
        self.wmes: set[WME] = set()
        self.successors: list[Any] = []  # JoinNode | NegativeNode

    def add(self, wme: WME) -> None:
        """Store *wme* and right-activate all successors."""
        self.wmes.add(wme)
        for node in self.successors:
            node.right_activate(wme)

    def remove(self, wme: WME) -> None:
        """Remove *wme* and notify successors to retract dependent tokens."""
        self.wmes.discard(wme)
        for node in self.successors:
            node.right_remove(wme)

    def __repr__(self) -> str:
        return f"AlphaMemory({len(self.wmes)} wmes, {len(self.successors)} successors)"


class AlphaTestNode:
    """Applies one intra-element test on a single field of a WME's fact.

    If the test passes, the WME is propagated to all children (which may
    be further ``AlphaTestNode``s or a terminal ``AlphaMemory``).
    """

    def __init__(
        self,
        field_name: str,
        test_fn: Callable[[Any], bool],
    ) -> None:
        self.field_name = field_name
        self.test_fn = test_fn
        self.children: list[AlphaTestNode | AlphaMemory] = []

    def activate(self, wme: WME) -> None:
        """Test *wme* and propagate to children on success."""
        value = getattr(wme.fact, self.field_name)
        if self.test_fn(value):
            for child in self.children:
                if isinstance(child, AlphaMemory):
                    child.add(wme)
                else:
                    child.activate(wme)

    def __repr__(self) -> str:
        return f"AlphaTestNode({self.field_name!r}, children={len(self.children)})"


class AlphaNetwork:
    """Root of the alpha network — dispatches WMEs by fact type.

    Maintains a reverse index (``wme_to_memories``) so that retraction
    can efficiently locate and remove a WME from all alpha memories it
    was stored in, without scanning the entire network.
    """

    def __init__(self) -> None:
        # type(fact) -> list of entry nodes (AlphaTestNode or AlphaMemory)
        self.type_dispatch: dict[type, list[AlphaTestNode | AlphaMemory]] = {}
        # wme.id -> list of AlphaMemory that contain this WME
        self.wme_to_memories: dict[int, list[AlphaMemory]] = {}

    def add_wme(self, wme: WME) -> None:
        """Push *wme* through the alpha network for its fact type."""
        fact_type = type(wme.fact)
        entry_nodes = self.type_dispatch.get(fact_type, [])
        # Track which memories this WME lands in
        memories_before = self._snapshot_memories(entry_nodes)
        for node in entry_nodes:
            if isinstance(node, AlphaMemory):
                node.add(wme)
            else:
                node.activate(wme)
        # Record reverse mapping for retraction
        memories_after = self._find_containing_memories(wme, entry_nodes)
        if memories_after:
            self.wme_to_memories[wme.id] = memories_after

    def remove_wme(self, wme: WME) -> None:
        """Remove *wme* from all alpha memories it resides in."""
        memories = self.wme_to_memories.pop(wme.id, [])
        for am in memories:
            am.remove(wme)

    def register_entry(self, fact_type: type, node: AlphaTestNode | AlphaMemory) -> None:
        """Register a top-level entry node for *fact_type*."""
        self.type_dispatch.setdefault(fact_type, []).append(node)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_memories(
        entry_nodes: list[AlphaTestNode | AlphaMemory],
    ) -> set[int]:
        """Placeholder — not actually needed, we scan after."""
        return set()

    @staticmethod
    def _find_containing_memories(
        wme: WME,
        entry_nodes: list[AlphaTestNode | AlphaMemory],
    ) -> list[AlphaMemory]:
        """Walk the entry nodes to find every AlphaMemory that now holds *wme*."""
        result: list[AlphaMemory] = []
        stack: list[AlphaTestNode | AlphaMemory] = list(entry_nodes)
        while stack:
            node = stack.pop()
            if isinstance(node, AlphaMemory):
                if wme in node.wmes:
                    result.append(node)
            else:
                stack.extend(node.children)
        return result
