"""Conflict resolution strategies for the Rete engine.

Each strategy receives a non-empty list of Instantiation objects and returns
the single instantiation that should fire next.

Strategies
----------
- **lex** — OPS5 LEX: salience → recency (max WME timestamp) → specificity.
- **mea** — OPS5 MEA: salience → recency of *first* condition's WME → specificity.
- **priority-only** — salience only (stable order for ties).

Users may supply a custom ``Callable[[list[Instantiation]], Instantiation]``
to :class:`ReteEngine` for full control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .beta import Instantiation

__all__ = [
    "resolve_lex",
    "resolve_mea",
    "resolve_priority_only",
    "STRATEGIES",
    "resolve_strategy",
]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def resolve_lex(instantiations: list[Instantiation]) -> Instantiation:
    """LEX strategy: salience (desc) → recency/timestamp (desc) → specificity (desc).

    *Recency* is the maximum WME timestamp across all conditions in the
    instantiation, matching OPS5's LEX ordering.
    """
    return max(
        instantiations,
        key=lambda inst: (
            inst.salience,
            inst.timestamp,
            len(inst.conds),
        ),
    )


def resolve_mea(instantiations: list[Instantiation]) -> Instantiation:
    """MEA strategy: salience (desc) → first-condition WME timestamp (desc) → specificity (desc).

    *Recency* is based on the timestamp of the WME that matched the **first**
    condition element, matching OPS5's MEA ordering.  If the token has no WMEs
    (shouldn't happen in practice), falls back to 0.
    """
    def _mea_key(inst: Instantiation):
        wmes = inst.token.wmes()
        first_ts = wmes[0].timestamp if wmes else 0
        return (inst.salience, first_ts, len(inst.conds))

    return max(instantiations, key=_mea_key)


def resolve_priority_only(instantiations: list[Instantiation]) -> Instantiation:
    """Priority-only strategy: sort solely by salience (desc).

    Ties are broken by insertion order (``max`` is stable on equal keys).
    """
    return max(instantiations, key=lambda inst: inst.salience)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, Callable[[list], object]] = {
    "lex": resolve_lex,
    "mea": resolve_mea,
    "priority-only": resolve_priority_only,
    "priority_only": resolve_priority_only,
}


def resolve_strategy(name_or_fn) -> Callable:
    """Return a conflict-resolution callable.

    Parameters
    ----------
    name_or_fn : str | Callable
        Either a strategy name (``"lex"``, ``"mea"``, ``"priority-only"``) or
        a user-supplied ``Callable[[list[Instantiation]], Instantiation]``.

    Returns
    -------
    Callable
        The resolution function.

    Raises
    ------
    ValueError
        If *name_or_fn* is a string that is not in :data:`STRATEGIES`.
    """
    if callable(name_or_fn) and not isinstance(name_or_fn, str):
        return name_or_fn
    if isinstance(name_or_fn, str):
        try:
            return STRATEGIES[name_or_fn]
        except KeyError:
            raise ValueError(
                f"Unknown conflict-resolution strategy {name_or_fn!r}. "
                f"Available: {', '.join(sorted(STRATEGIES))}"
            ) from None
    raise TypeError(f"Expected str or callable, got {type(name_or_fn).__name__}")
