# Example 1 -- Symbolic Regression: Recovering a Damped Oscillator
#
# GP evolves computer programs (expression trees) that fit noisy numeric
# data. Unlike curve fitting with a FIXED model shape, GP searches over
# model STRUCTURES: the tree decides which operators appear, where, and
# with which constants. The result is a human-readable formula.
#
# This demo hides the true generating function -- a damped sine wave
#     y(x) = 2 * exp(-x/3) * sin(2x) + noise
# sampled at 40 points on [0, 5] -- and lets GP rediscover it. Note that
# the function set below deliberately has NO `exp` primitive, so the
# evolved formula cannot be an exact copy of the truth; GP must find an
# equally good structural approximation. That is typical: symbolic
# regression rewards "right shape", not "textbook answer".
#
# Two classic GP knobs are on display:
#   * Parsimony pressure -- fitness = error + lambda * tree size, which
#     discourages bloat (the runaway growth of useless subtrees).
#   * Protected operators -- division returns 1.0 when the denominator is
#     near zero, so every random tree evaluates to a finite number.
#
# References:
#   Symbolic regression: https://en.wikipedia.org/wiki/Symbolic_regression
#   Bloat in GP:         https://link.springer.com/chapter/10.1007/978-3-540-69181-0_14
#   GP field guide:      https://cs.gmu.edu/~kic/pubs/gp-field-guide-part1.pdf

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Final

from gp_core import (
    TerminalSampler,
    Tree,
    evolve,
    mutate_point,
    mutate_subtree,
    random_tree,
)

SEED: Final = 7  # chosen so this run converges quickly and repeatably

# ---------------------------------------------------------------------------
# The hidden ground truth and the data we are allowed to see.
# ---------------------------------------------------------------------------


def target(x: float) -> float:
    """The secret: a damped oscillation GP must approximate."""
    return 2.0 * math.exp(-x / 3.0) * math.sin(2.0 * x)


NOISE_SIGMA: Final = 0.05
XS: Final[list[float]] = [5.0 * i / 39 for i in range(40)]


def make_dataset(rng: random.Random) -> list[tuple[float, float]]:
    """Sample the target at XS with Gaussian measurement noise."""
    return [(x, target(x) + rng.gauss(0.0, NOISE_SIGMA)) for x in XS]


# ---------------------------------------------------------------------------
# Function set: name -> (arity, evaluation). All operators are "protected":
# they accept any float tuple and return a finite float.
# ---------------------------------------------------------------------------

ArityAndEval = tuple[int, Callable[[Sequence[float]], float]]

FUNCTIONS: Final[dict[str, ArityAndEval]] = {
    "add": (2, lambda a: a[0] + a[1]),
    "sub": (2, lambda a: a[0] - a[1]),
    "mul": (2, lambda a: a[0] * a[1]),
    "div": (  # protected: |denominator| < 1e-10 -> numerator (identity-ish)
        2,
        lambda a: a[0] / a[1] if abs(a[1]) > 1e-10 else 1.0,
    ),
    "sin": (1, lambda a: math.sin(a[0])),
    "cos": (1, lambda a: math.cos(a[0])),
    "neg": (1, lambda a: -a[0]),
    "abs": (1, lambda a: abs(a[0])),
}

FUNCTION_ARITIES: Final[dict[str, int]] = {
    name: arity for name, (arity, _eval) in FUNCTIONS.items()
}


def terminal_sampler(rng: random.Random) -> TerminalSampler:
    """Terminals = the input variable x plus Ephemeral Random Constants.

    An ERC is re-drawn every time it is needed, so no two trees ever share
    the same constant set: GP re-invents its own numeric literals every
    generation and selection tunes them statistically.
    """

    def sample() -> str:
        if rng.random() < 0.25:
            return "x"
        return f"{rng.uniform(-3.0, 3.0):.4f}"

    return sample


# ---------------------------------------------------------------------------
# Evaluating a tree on one input, and the parsimony-pressured fitness.
# ---------------------------------------------------------------------------


def evaluate(tree: Tree, x: float) -> float:
    """Recursive post-order evaluation. Leaves are 'x' or a float literal."""
    if tree.is_leaf:
        return x if tree.name == "x" else float(tree.name)
    arity, op = FUNCTIONS[tree.name]
    if len(tree.args) != arity:  # defensive: corrupted tree
        return 0.0
    values = [evaluate(child, x) for child in tree.args]
    result = op(values)
    return result if math.isfinite(result) else 0.0


PARSIMONY: Final = 5e-4  # fitness penalty per tree node


class RegressionFitness:
    """Mean squared error plus a size penalty (parsimony pressure)."""

    def __init__(self, data: list[tuple[float, float]]) -> None:
        self.data = data

    def __call__(self, tree: Tree) -> float:
        squared = sum((evaluate(tree, x) - y) ** 2 for x, y in self.data)
        return squared / len(self.data) + PARSIMONY * tree.size


# ---------------------------------------------------------------------------
# Pretty printing: prefix tree -> readable infix formula.
# ---------------------------------------------------------------------------


def render(tree: Tree) -> str:
    """Infix for binaries, call-style for unaries, plain for leaves."""
    if tree.is_leaf:
        return tree.name
    (arity, _op) = FUNCTIONS[tree.name]
    kids = [render(child) for child in tree.args]
    if arity == 2:
        symbol = {"add": " + ", "sub": " - ", "mul": " * ", "div": " / "}.get(tree.name)
        if symbol is not None:
            return f"({kids[0]}{symbol}{kids[1]})"
        return f"{tree.name}({kids[0]}, {kids[1]})"
    return f"{tree.name}({kids[0]})"


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def build_population(rng: random.Random, size: int, max_depth: int) -> list[Tree]:
    terminals = terminal_sampler(rng)
    return [
        random_tree(
            rng,
            FUNCTION_ARITIES,
            terminals,
            max_depth=max_depth,
            min_depth=2,
            method="half",
        )
        for _ in range(size)
    ]


def main() -> None:
    rng = random.Random(SEED)
    data = make_dataset(rng)
    scoring = RegressionFitness(data)

    population = build_population(rng, size=400, max_depth=8)
    terminals = terminal_sampler(rng)

    def mutate(tree: Tree) -> Tree:
        roll = rng.random()
        if roll < 0.55:
            return mutate_subtree(rng, tree, FUNCTION_ARITIES, terminals, max_depth=4)
        return mutate_point(rng, tree, FUNCTION_ARITIES, terminals)

    def report(generation: int, penalty_fitness: float, best: Tree) -> None:
        if generation % 5 and generation != 39:  # the run plateaus; print checkpoints
            return
        mse = sum((evaluate(best, x) - y) ** 2 for x, y in data) / len(data)
        print(f"gen {generation:3d}  best MSE {mse:8.5f}  size {best.size:3d}  {render(best)[:60]}")

    print(f"# Symbolic regression: fit {len(data)} noisy samples of a damped oscillator")
    print(f"# hidden truth: y = 2*exp(-x/3)*sin(2x), noise sd = {NOISE_SIGMA}\n")

    result = evolve(
        rng,
        population,
        scoring,
        mutate,
        generations=40,
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=7,
        elitism=2,
        max_depth=12,
        reporter=report,
    )

    mse = sum((evaluate(result.best, x) - y) ** 2 for x, y in data) / len(data)
    rms_target = sum(y * y for _, y in data) / len(data)
    print(f"\nBest after {result.generations} generations:")
    print(f"  formula   y = {render(result.best)}")
    print(f"  nodes     {result.best.size}, depth {result.best.depth}")
    print(
        f"  MSE {mse:.5f} vs signal power {rms_target:.5f}  "
        f"(explains {100 * (1 - mse / rms_target):.1f}% of variance)"
    )

    print("\n     x      y*      y_hat   error")
    for x in XS[::4]:
        y_hat = evaluate(result.best, x)
        print(f"  {x:5.2f}  {target(x):6.3f}  {y_hat:6.3f}  {y_hat - target(x):+6.3f}")


if __name__ == "__main__":
    main()
