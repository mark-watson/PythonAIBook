# Example 4 -- Evolving L-System Growth Rules to Match a Target Plant
#
# GP where the evolved object is a BIOLOGICAL DEVELOPMENT PROGRAM: an
# L-system rule whose repeated application grows a plant, and whose fitness
# is the plant's shape. This is the most "developmental" of the four demos --
# a tiny genotype (the rule tree) unfolds into a large phenotype (the
# rendered plant) through iterative rewriting, exactly as genes unfold
# into organisms.
#
# Task: we show GP a target plant -- grown by a known rule we keep secret --
# and evolve replacement rules F -> ... until the evolved plant's occupied
# cells overlap the target's cells. The fitness is the JACCARD DISTANCE of
# the two cell sets: 0.0 means "pixel-perfect" and higher is worse. Every
# candidate paints the shared start cell, so the distance stays below 1.0.
# Because every candidate is rendered to the same grid, the landscape is
# surprisingly smooth: rules that grow "taller, then bushier, then leaning
# left" climb the fitness staircase gradually.
#
# Turtle semantics (classic L-system drawing):
#   F  move forward, drawing a segment
#   f  move forward WITHOUT drawing (a stem you cannot see)
#   +  turn left 25 degrees        -  turn right 25 degrees
#   [ push position+heading         ]  pop them back (a branch!)
#
# References:
#   L-systems:       https://en.wikipedia.org/wiki/L-system
#   Prusinkiewicz & Lindenmayer, "The Algorithmic Beauty of Plants"
#   Jacob, C. (1994) "Genetic L-System Programming", PPSN III:
#                    https://doi.org/10.1007/3-540-58484-6_277

from __future__ import annotations

import math
import random
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

SEED: Final = 42
GRID_W: Final = 61
GRID_H: Final = 31
ITERATIONS: Final = 4
MAX_EXPANSION: Final = 12000  # rewrite length cap (anti-explosion guard)
TURN_DEG: Final = 25.0

# Rule-grammar function set: seq glues tokens, branch wraps in [ ].
FUNCTION_ARITIES: Final[dict[str, int]] = {"seq": 2, "branch": 1}

# Turtle alphabet as terminals. 'branch'/'seq' are the only nonterminals.
TERMINAL_POOL: Final[list[str]] = ["F", "F", "F", "f", "+", "-"]


def terminals(rng: random.Random) -> TerminalSampler:
    pool = list(TERMINAL_POOL)

    def sample() -> str:
        return rng.choice(pool)

    return sample


# ---------------------------------------------------------------------------
# Genotype -> phenotype: expand the rule tree, rewrite the axiom, draw.
# ---------------------------------------------------------------------------


def rule_text(tree: Tree) -> str:
    """Render a rule tree as L-system RHS text, e.g. F[+F][-F]F."""
    if tree.is_leaf:
        return tree.name
    kids = [rule_text(child) for child in tree.args]
    if tree.name == "seq":
        return kids[0] + kids[1]
    return f"[{kids[0]}]"


def expand(rule: str, iterations: int = ITERATIONS) -> str:
    """Apply `F -> rule` to every F, `iterations` times, from axiom F."""
    string = "F"
    for _ in range(iterations):
        string = string.replace("F", rule)
        if len(string) > MAX_EXPANSION:
            return string[:MAX_EXPANSION]
    return string


def render_grid(string: str) -> set[tuple[int, int]]:
    """Turtle-walk an L-system string; return the set of painted cells."""
    x = GRID_W // 2.0
    y = GRID_H - 1.0
    heading = 90.0  # pointing up
    stack: list[tuple[float, float, float]] = []
    cells: set[tuple[int, int]] = {(int(round(x)), int(round(y)))}

    for token in string:
        if token == "F":
            angle = math.radians(heading)
            x += math.cos(angle)
            y -= math.sin(angle)  # image rows grow downward
            cell = (int(round(x)), int(round(y)))
            if 0 <= cell[0] < GRID_W and 0 <= cell[1] < GRID_H:
                cells.add(cell)
        elif token == "f":
            angle = math.radians(heading)
            x += math.cos(angle)
            y -= math.sin(angle)
        elif token == "+":
            heading += TURN_DEG
        elif token == "-":
            heading -= TURN_DEG
        elif token == "[":
            stack.append((x, y, heading))
        elif token == "]" and stack:
            x, y, heading = stack.pop()
    return cells


# The hidden target plant: grown from a rule GP must rediscover.
TARGET_RULE: Final = "F[+F]F[-F]F"
TARGET_CELLS: Final[set[tuple[int, int]]] = render_grid(expand(TARGET_RULE))


def jaccard_distance(tree: Tree) -> float:
    """1 - |overlap| / |union| of painted cells against the target plant.

    The turtle always paints its start cell, so both sets are non-empty and
    the result is 0.0 for a pixel-perfect plant.
    """
    cells = render_grid(expand(rule_text(tree)))
    union = len(cells | TARGET_CELLS)
    return 1.0 - len(cells & TARGET_CELLS) / union


def print_plant(cells: set[tuple[int, int]], label: str) -> None:
    """Print one plant as an ASCII grid under a heading."""
    print(f"\n{label}")
    for row in range(GRID_H):
        print("  " + "".join("#" if (col, row) in cells else "." for col in range(GRID_W)))


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
    population = build_population(rng, size=300, max_depth=7)
    sampler = terminals(rng)

    def mutate(tree: Tree) -> Tree:
        roll = rng.random()
        if roll < 0.5:
            return mutate_subtree(rng, tree, FUNCTION_ARITIES, sampler, max_depth=3)
        if roll < 0.85:
            return mutate_point(rng, tree, FUNCTION_ARITIES, sampler)
        return mutate_hoist(rng, tree)

    def report(generation: int, best_fitness: float, best: Tree) -> None:
        print(f"gen {generation:3d}  jaccard {best_fitness:6.3f}  rule F -> {rule_text(best)[:30]}")

    print("# Evolving an L-system growth rule to reproduce a target plant")
    print(f"# grid {GRID_W}x{GRID_H}, {ITERATIONS} rewrite iterations, turn {TURN_DEG:.0f} deg\n")

    result = evolve(
        rng,
        population,
        jaccard_distance,
        mutate,
        generations=70,
        crossover_rate=0.85,
        mutation_rate=0.12,
        tournament_size=6,
        elitism=2,
        max_depth=10,
        target=0.02,
        reporter=report,
    )

    print(f"\nsecret target rule was:  F -> {TARGET_RULE}")
    print(f"evolved rule:            F -> {rule_text(result.best)}")
    print(
        f"shape distance {result.best_fitness:.3f} after {result.generations} generations "
        f"(pixel-perfect: {result.best_fitness == 0.0})"
    )

    print_plant(TARGET_CELLS, "TARGET PLANT (the hidden rule's phenotype):")
    print_plant(render_grid(expand(rule_text(result.best))), "EVOLVED PLANT (GP's rule phenotype):")


if __name__ == "__main__":
    main()
