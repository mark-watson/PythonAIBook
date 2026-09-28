# Example 2 -- Boolean Logic Synthesis with Tree GP
#
# Genetic programming on the canonical GP problem class: evolve a PROGRAM
# (here, a combinational logic circuit) whose INPUT/OUTPUT behavior matches
# a specification given as a truth table. Fitness counts wrong rows, so the
# landscape is a staircase of partially-correct circuits -- exactly the kind
# of "behavior, not structure" scoring that distinguishes GP from ordinary
# parameter optimization.
#
# The specification is a machine-safety interlock:
#
#   ENABLE = (both operator keys pressed) AND (no fault)
#            AND (at least one of the two redundant sensors agrees)
#
# Inputs: K1, K2 = operator keys; F = fault flag (active high);
#         S1, S2 = redundant sensors. The interlock needs only ONE of the
#         redundant sensors to agree, so the pair is free to disagree on
#         any row and ENABLE still fires.
# Hand-writing the gate netlist for this is error-prone; GP rediscovers a
# correct netlist from 32 example rows in seconds.
#
# This is also a clean illustration of two GP subtleties:
#   * Ephemeral Boolean constants (TRUE/FALSE leaves) -- without them, GP
#     struggles badly on parity-like targets.
#   * Behavior-only scoring: the evolved netlist in our run is NOT the
#     textbook formula. It is a structurally different circuit that
#     computes the same function, and the demo verifies it row by row.
#     GP optimizes what a program DOES, never how it LOOKS.
#
# References:
#   Koza, J. (1992) "Genetic Programming: On the Programming of Computers
#                   by Means of Natural Selection", MIT Press (Boolean
#                   problems)
#   Parity difficulty in GP: Langdon & Poli (1998), "Why Building Blocks
#                   Don't Work on Parity Problems", CSRP-98-17:
#                   http://web4.cs.ucl.ac.uk/staff/W.Langdon/csrp-98-17/eq.html
#   Logic synthesis:           https://en.wikipedia.org/wiki/Logic_synthesis

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from gp_core import (
    TerminalSampler,
    Tree,
    evolve,
    mutate_hoist,
    mutate_point,
    mutate_subtree,
    random_tree,
)

SEED: Final = 12

INPUTS: Final[tuple[str, ...]] = ("K1", "K2", "F", "S1", "S2")


def specification(row: Sequence[bool]) -> bool:
    """The interlock we want GP to rediscover (row order = INPUTS)."""
    k1, k2, fault, s1, s2 = row
    return (k1 and k2) and not fault and (s1 or s2)


TRUTH_TABLE: Final[list[tuple[tuple[bool, ...], bool]]] = [
    (row, specification(row)) for row in itertools.product((False, True), repeat=len(INPUTS))
]

# The same cases keyed by input name, built once so fitness() does not
# rebuild a dict for every candidate. Evaluating a tree is the hot path.
CASES: Final[list[tuple[Mapping[str, bool], bool]]] = [
    (dict(zip(INPUTS, row, strict=True)), expected) for row, expected in TRUTH_TABLE
]

# Gate-level function set. NOT is unary; AND/OR binary. (NAND-only synthesis
# is possible too -- try it as an exercise!)
GATES: Final[dict[str, tuple[int, Callable[[Sequence[bool]], bool]]]] = {
    "AND": (2, lambda a: a[0] and a[1]),
    "OR": (2, lambda a: a[0] or a[1]),
    "NOT": (1, lambda a: not a[0]),
}

FUNCTION_ARITIES: Final[dict[str, int]] = {name: arity for name, (arity, _eval) in GATES.items()}


def terminals(rng: random.Random) -> TerminalSampler:
    """Inputs plus 20% chance of an Ephemeral Random Boolean constant."""
    pool = list(INPUTS)

    def sample() -> str:
        if rng.random() < 0.2:
            return rng.choice(("TRUE", "FALSE"))
        return rng.choice(pool)

    return sample


def evaluate(tree: Tree, row: Mapping[str, bool]) -> bool:
    """Interpret the tree as a gate netlist over one input row."""
    if tree.is_leaf:
        if tree.name == "TRUE":
            return True
        if tree.name == "FALSE":
            return False
        return row[tree.name]
    arity, op = GATES[tree.name]
    if len(tree.args) != arity:
        return False
    return op([evaluate(child, row) for child in tree.args])


def fitness(tree: Tree) -> float:
    """Fraction of truth-table rows the circuit gets WRONG (0.0 = perfect)."""
    wrong = 0
    for row, expected in CASES:
        if evaluate(tree, row) != expected:
            wrong += 1
    return float(wrong) / len(CASES)


def render(tree: Tree) -> str:
    """Human-readable gate expression, e.g. (NOT (K1 AND F))."""
    if tree.is_leaf:
        return tree.name
    arity, _op = GATES[tree.name]
    kids = [render(child) for child in tree.args]
    if arity == 2:
        return f"({kids[0]} {tree.name} {kids[1]})"
    return f"({tree.name} {kids[0]})"


def build_population(rng: random.Random, size: int, max_depth: int) -> list[Tree]:
    sampler = terminals(rng)
    return [
        random_tree(
            rng,
            FUNCTION_ARITIES,
            sampler,
            max_depth=max_depth,
            min_depth=2,
            method="grow",
        )
        for _ in range(size)
    ]


def main() -> None:
    rng = random.Random(SEED)
    population = build_population(rng, size=250, max_depth=8)
    sampler = terminals(rng)

    def mutate(tree: Tree) -> Tree:
        roll = rng.random()
        if roll < 0.45:
            return mutate_subtree(rng, tree, FUNCTION_ARITIES, sampler, max_depth=4)
        if roll < 0.9:
            return mutate_point(rng, tree, FUNCTION_ARITIES, sampler)
        return mutate_hoist(rng, tree)

    def report(generation: int, best_fitness: float, best: Tree) -> None:
        print(
            f"gen {generation:3d}  wrong rows {round(best_fitness * len(CASES)):2d}/32  "
            f"size {best.size:3d}  {render(best)[:58]}"
        )

    print("# Evolving a 5-input safety interlock from its 32-row truth table")
    print("# inputs: K1, K2 (keys), F (fault), S1, S2 (sensors)\n")

    result = evolve(
        rng,
        population,
        fitness,
        mutate,
        generations=60,
        crossover_rate=0.85,
        mutation_rate=0.12,
        tournament_size=5,
        elitism=1,
        max_depth=12,
        target=0.0,
        reporter=report,
    )

    print(
        f"\nPerfect circuit found: {result.solved} "
        f"(generation {result.generations}, {result.best.size} gates, depth {result.best.depth})"
    )
    print(f"  ENABLE = {render(result.best)}")

    # Double-check: re-evaluate EVERY row against the discovered netlist.
    failures = [row for row, expected in CASES if evaluate(result.best, row) != expected]
    print(f"  verification: {len(CASES) - len(failures)}/{len(CASES)} rows correct")

    print("\n  truth table (1 = ENABLE):")
    header = "  " + " ".join(f"{name:>2}" for name in INPUTS) + " | spec | evolved"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row, expected in CASES:
        got = evaluate(result.best, row)
        mark = "  " if got == expected else " <-- MISMATCH"
        bits = " ".join(f"{int(row[name]):>2}" for name in INPUTS)
        print(f"  {bits} |  {int(expected)}  |   {int(got)}{mark}")


if __name__ == "__main__":
    main()
