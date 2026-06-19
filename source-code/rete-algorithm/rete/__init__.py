"""Rete Algorithm Engine for Python.

A lightweight, idiomatic Python implementation of the Rete algorithm
for embedded expert systems.

Quick start::

    from rete import ReteEngine, Fact, Pat, Var, Gt

    engine = ReteEngine()

    @engine.rule(Pat(MyFact, x=Var("x")))
    def my_rule(ctx, x):
        ...

    engine.assert_fact(MyFact(x=42))
    engine.run()
"""

from .context import RuleContext
from .engine import ReteEngine, WorkingMemory
from .facts import Fact, WME
from .patterns import (
    Cond,
    Contains,
    Eq,
    Ge,
    Gt,
    In,
    Le,
    Lt,
    Match,
    Pat,
    PatternOp,
    Test,
    Var,
    cond,
)

__all__ = [
    # Engine
    "ReteEngine",
    "WorkingMemory",
    "RuleContext",
    # Facts
    "Fact",
    "WME",
    # Patterns
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
