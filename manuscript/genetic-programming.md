# Genetic Programming

Genetic programming (GP) is evolutionary search over *programs*. Ordinary machine learning fits numbers inside a model whose shape you choose. Genetic algorithms evolve fixed-length parameter vectors. GP goes one level up: it evolves the structure itself. The individual is a tree-shaped program, the training signal is a set of *fitness cases* (input/output examples of the behavior you want), and the operators of evolution build new programs out of parts of old ones.

John Koza established the field in the early 1990s with the observation that, given a function set, a terminal set, and a behavioral score, evolution discovers not only the coefficients of a formula but the formula. Three decades of later work refined the machinery (typed trees, linear representation, grammar-guided GP, geometric semantic crossover), but the core loop is small enough to implement in a few hundred lines of Python, which is exactly what you will do in this chapter.

We build everything from first principles, with no NumPy and no GP framework, and apply the same engine to four different problems:

| Example | Evolved object | Fitness cases |
|---|---|---|
| 1. Symbolic regression | math expression | 40 noisy `(x, y)` samples |
| 2. Boolean logic synthesis | gate netlist | 32-row truth table |
| 3. Regex synthesis | pattern program | 8 positive + 11 negative strings |
| 4. L-system plant | growth rule | cell grid of a target plant |

Run time for each demo is under three seconds on a laptop, and every run is seeded and reproducible.

## Why trees?

A GP individual is an ordered rooted tree. Internal nodes are *functions* (with a fixed arity), leaves are *terminals* (variables and constants). Two properties make trees the right representation:

1. **Closed under composition.** Every subtree is itself a valid program. You can cut a subtree from one tree and graft it into another at any node, and the offspring is guaranteed to be a syntactically valid program of the same species. A crossover of two floating-point vectors has no such guarantee; a crossover of two expression trees does. This is why GP inherits genetic algorithms' operators unchanged.
2. **Modularity.** Subtrees compute reusable chunks (`sin(x + x)` appears inside many better solutions). Selection can spread a good subtree through the population the way a good schema spreads in a genetic algorithm.

There is a price, discussed later: trees can *grow*. Useless code costs nothing to keep, so GP populations tend toward bloat unless you apply pressure against size.

## The GP loop

Generational GP looks like this:

1. Create a random population of `N` trees over the function set `F` and terminal set `T`.
2. Evaluate every tree on the fitness cases and compute a scalar fitness (we will minimize throughout).
3. Repeat for the generation budget:
   a. Select parents by *tournament*: draw `k` individuals uniformly at random, keep the best.
   b. With high probability apply *subtree crossover*: pick one random node in each parent, swap the subtrees rooted at those nodes. Otherwise clone a parent.
   c. With small probability apply a *mutation*: replace a random subtree with a fresh random tree (subtree mutation), swap a node for another node of the same type class (point mutation), or replace a node by one of its own descendants (hoist mutation).
   d. Optionally copy the best few individuals into the next generation unchanged (*elitism*).
   e. Evaluate the children.

Two design choices deserve names because you will meet them in every GP paper:

* **Initialization, `full` vs `grow`.** The *full* method creates perfect trees where every root-to-leaf path has length exactly `D`. The *grow* method makes its own function-versus-terminal decision at every step, yielding irregular trees. Koza's standard recipe, called *half-and-half*, flips a coin between the two per tree.
* **Fitness cases, not objectives.** GP does not optimize a model's parameters against a loss function you differentiated. It executes each candidate program on example inputs and scores the *behavior*. That is why GP applies to problems where no differentiable model exists at all: robot controllers, trading heuristics, regular expressions, growth rules.

## A tiny engine: `gp_core.py`

The engine below implements exactly that loop. The only design rule is separation of concerns: `gp_core` manipulates tree *structure* and never looks at what the nodes mean. Each example supplies three things the engine calls:

* a function-arity map `{name: arity}`,
* a *terminal sampler*, a zero-argument callable that returns one terminal symbol. Making terminals a callable instead of a list lets an example draw **Ephemeral Random Constants** (ERCs): a fresh random number every time a constant leaf is needed. Two trees never see the same constants, so constants are re-invented each generation and selection tunes them statistically.
* a fitness function `tree -> float` that the loop minimizes.

Study `all_nodes()`: it returns every node of a tree by reference together with its parent and position. One helper serves all the operators, because every operator reduces to "pick a random node, then splice or overwrite its subtree."

```python
# Genetic Programming Core: A Tiny, Dependency-Free Tree-GP Engine
#
# Provides the machinery every tree-based GP example in this chapter needs:
# random tree creation ("full" and "grow" methods), subtree crossover,
# subtree and point mutation, tournament selection, and a generational
# evolve() loop with elitism. The engine knows nothing about what the trees
# *mean* -- each example supplies its own function set, terminal sampler, and
# fitness function, so the same core drives symbolic regression, Boolean
# logic synthesis, program evolution, and L-system rule evolution.
#
# References:
#   GP overview:    https://en.wikipedia.org/wiki/Genetic_programming
#   Koza, J. (1992) "Genetic Programming: On the Programming of Computers
#                   by Means of Natural Selection", MIT Press (full/grow
#                   initialization, tournament selection)
#   Bloat:          https://en.wikipedia.org/wiki/Genetic_programming#Bloat


from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

# A terminal sampler is a zero-argument callable that returns one terminal
# symbol (variable name, literal constant, character-class tag, ...).
# Examples use it to draw Ephemeral Random Constants (ERCs) on demand.
TerminalSampler = Callable[[], str]


@dataclass
class Tree:
    """A GP expression tree: a name plus an ordered list of child subtrees.

    Leaves have no children; their `name` carries the terminal symbol
    (for example "x", "0.42", "digit"). Internal nodes name a function
    whose arity must agree with len(args).
    """

    name: str
    args: list[Tree] = field(default_factory=list)

    def copy(self) -> Tree:
        """Return a deep copy of this subtree."""
        return Tree(self.name, [child.copy() for child in self.args])

    @property
    def size(self) -> int:
        """Total number of nodes (functions + terminals)."""
        return 1 + sum(child.size for child in self.args)

    @property
    def depth(self) -> int:
        """Longest root-to-leaf path, counted in nodes."""
        return 1 + max((child.depth for child in self.args), default=0)

    @property
    def is_leaf(self) -> bool:
        return not self.args

    def __str__(self) -> str:
        """Prefix notation, e.g. (add x (mul 0.5 y))."""
        if self.is_leaf:
            return self.name
        inner = " ".join(str(arg) for arg in self.args)
        return f"({self.name} {inner})"


# A mutator takes a tree and returns a mutated copy of it.
Mutator = Callable[[Tree], Tree]

# A fitness function maps a tree to a real number; evolve() MINIMIZES it.
FitnessFunction = Callable[[Tree], float]


def sequence_sampler(rng: random.Random, symbols: Sequence[str]) -> TerminalSampler:
    """Wrap a fixed terminal list into a TerminalSampler (uniform choice)."""
    pool = list(symbols)

    def sample() -> str:
        return rng.choice(pool)

    return sample


def all_nodes(tree: Tree) -> list[tuple[Tree, Tree | None, int]]:
    """Listing of (node, parent, index-in-parent) triples, root first.

    Nodes are returned *by reference* so callers can splice subtrees in
    place. The root's parent is None and its index is -1. Every operator
    needs only each node together with its parent and index, so the exact
    visit order (depth-first, siblings right-to-left) is a nicety, not a
    contract.
    """
    out: list[tuple[Tree, Tree | None, int]] = [(tree, None, -1)]
    stack: list[Tree] = [tree]
    while stack:
        node = stack.pop()
        # Appending while iterating backwards lists each parent's children
        # right-to-left, and pushes the leftmost child last so the walk
        # itself continues left-to-right.
        for index in range(len(node.args) - 1, -1, -1):
            child = node.args[index]
            out.append((child, node, index))
            stack.append(child)
    return out


def random_tree(
    rng: random.Random,
    functions: Mapping[str, int],
    terminals: TerminalSampler,
    max_depth: int,
    min_depth: int = 1,
    method: str = "half",
) -> Tree:
    """Create one random valid tree.

    Parameters:
        rng        -- seeded random.Random for reproducibility
        functions  -- map of function name -> arity (nonterminals)
        terminals  -- sampler producing terminal symbols
        max_depth  -- hard cap on root-to-leaf path length (in nodes)
        min_depth  -- trees must reach at least this depth
        method     -- "full" (perfect trees), "grow" (mixed shapes) or
                      "half" (coin flip between the two, Koza's default)

    The "grow" method stops a branch with probability 0.4 once the minimum
    depth is reached, so branches end at different depths and the tree comes
    out irregular -- the shape GP is known for.
    """
    if method not in ("full", "grow", "half"):
        raise ValueError(f"unknown initialization method {method!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be at least 1, got {max_depth}")
    if min_depth > max_depth:
        raise ValueError(f"min_depth {min_depth} exceeds max_depth {max_depth}")
    if method == "half":
        method = "full" if rng.random() < 0.5 else "grow"

    names = list(functions)

    def build(depth_left: int, depth: int) -> Tree:
        can_branch = depth_left > 1 and bool(names)
        if not can_branch:
            return Tree(terminals())
        must_branch = depth < min_depth
        if not must_branch and method == "grow" and rng.random() < 0.4:
            return Tree(terminals())
        name = rng.choice(names)
        arity = functions[name]
        return Tree(name, [build(depth_left - 1, depth + 1) for _ in range(arity)])

    return build(max_depth, 1)


def crossover(
    rng: random.Random,
    parent_a: Tree,
    parent_b: Tree,
    max_depth: int | None = None,
) -> tuple[Tree, Tree]:
    """Subtree crossover: swap one random subtree between two clones.

    Both offspring are returned; the parents are never modified. If
    `max_depth` is given, a swap that would make either offspring deeper
    than the cap is skipped (the clones come back unchanged), so bloat
    stays bounded without disabling crossover.
    """
    child_a, child_b = parent_a.copy(), parent_b.copy()

    spots_a = all_nodes(child_a)
    spots_b = all_nodes(child_b)
    node_a, parent_na, index_a = rng.choice(spots_a)
    node_b, parent_nb, index_b = rng.choice(spots_b)

    if max_depth is not None:
        # Refuse the swap if either side would exceed the depth cap.
        depth_a_rest = parent_a.depth - node_a.depth  # context above node_a
        depth_b_rest = parent_b.depth - node_b.depth
        new_depth_a = depth_a_rest + node_b.depth
        new_depth_b = depth_b_rest + node_a.depth
        if max(new_depth_a, new_depth_b) > max_depth:
            return child_a, child_b

    # Splicing at the root means the whole other subtree becomes the child.
    if parent_na is None:
        child_a = node_b.copy()
    else:
        parent_na.args[index_a] = node_b.copy()
    if parent_nb is None:
        child_b = node_a.copy()
    else:
        parent_nb.args[index_b] = node_a.copy()
    return child_a, child_b


def mutate_subtree(
    rng: random.Random,
    tree: Tree,
    functions: Mapping[str, int],
    terminals: TerminalSampler,
    max_depth: int = 4,
) -> Tree:
    """Subtree mutation: replace one random node's subtree with fresh randoms.

    The replacement subtree respects `max_depth`, but the whole tree can
    still end up deeper than the original where the graft happened.
    """
    child = tree.copy()
    node, _parent, _index = rng.choice(all_nodes(child))
    new = random_tree(rng, functions, terminals, max_depth=max_depth, method="grow")
    # Overwrite in place: reusing the node object keeps the tree connected.
    node.name, node.args = new.name, new.args
    return child


def mutate_point(
    rng: random.Random,
    tree: Tree,
    functions: Mapping[str, int],
    terminals: TerminalSampler,
) -> Tree:
    """Point mutation: swap a function for another of the SAME arity, or a
    terminal for another terminal. Structure (shape) is preserved."""
    child = tree.copy()
    node, _parent, _index = rng.choice(all_nodes(child))
    if node.is_leaf:
        node.name = terminals()
    else:
        same_arity = [n for n, a in functions.items() if a == len(node.args)]
        if same_arity:
            node.name = rng.choice(same_arity)
    return child


def mutate_hoist(rng: random.Random, tree: Tree) -> Tree:
    """Hoist mutation: replace a node by one of its own descendants, pruning
    the surrounding context. A cheap anti-bloat operator."""
    child = tree.copy()
    node, _parent, _index = rng.choice(all_nodes(child))
    pick, _p, _i = rng.choice(all_nodes(node))
    node.name = pick.name
    node.args = [arg.copy() for arg in pick.args]
    return child


def tournament(
    rng: random.Random,
    fitnesses: Sequence[float],
    population: Sequence[Tree],
    k: int = 3,
) -> Tree:
    """Tournament selection (minimization). Draw k DISTINCT individuals at
    random, return the one with the LOWEST fitness. Indices refer into the
    parallel `fitnesses` and `population` sequences."""
    n = len(population)
    if n == 0:
        raise ValueError("cannot select from an empty population")
    if k < 1:
        raise ValueError(f"tournament size must be at least 1, got {k}")
    contenders = rng.sample(range(n), min(k, n))
    return population[min(contenders, key=fitnesses.__getitem__)]


@dataclass
class EvolutionResult:
    """Outcome of one evolve() run.

    `history[i]` is the best-so-far fitness reported at the start of
    generation `i`, and the final entry repeats `best_fitness`.
    """

    best: Tree
    best_fitness: float
    generations: int
    solved: bool
    history: list[float] = field(default_factory=list)


def evolve(
    rng: random.Random,
    population: list[Tree],
    fitness: FitnessFunction,
    mutate: Mutator,
    *,
    generations: int,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
    tournament_size: int = 7,
    elitism: int = 1,
    max_depth: int = 17,
    target: float | None = None,
    reporter: Callable[[int, float, Tree], None] | None = None,
) -> EvolutionResult:
    """Generational GP loop (minimizes `fitness`).

    Each generation: rank the population, optionally stop at `target`,
    carry the best `elitism` trees unchanged, then fill the rest of the
    next generation by tournament-selected crossover, mutating each child
    with probability `mutation_rate`.

    Parameters:
        rng             -- seeded random source
        population      -- initial list of Trees (mutated in no way)
        fitness         -- objective to minimize
        mutate          -- the mutator chosen by the example
        generations     -- budget of generations
        crossover_rate  -- probability of crossover per child (else clone)
        mutation_rate   -- probability of mutating each crossed/cloned child
        tournament_size -- k for selection tournaments
        elitism         -- number of best trees copied unchanged each gen;
                           an elitism at or above the population size leaves
                           no room for children, so nothing evolves
        max_depth       -- depth cap enforced on crossover; mutators apply
                           their own (relative) limits, so a mutated child
                           can still exceed this value
        target          -- stop early once best fitness <= target
        reporter        -- called as reporter(generation, best_fitness, best_tree)

    Returns:
        EvolutionResult with the best tree ever seen. `generations` counts
        the reproduction rounds actually completed, so an early stop at
        generation `g` reports `g`, not `g + 1`.
    """
    if not population:
        raise ValueError("population must not be empty")
    if generations < 0:
        raise ValueError("generations must not be negative")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1")
    if elitism < 0:
        raise ValueError("elitism must not be negative")
    n = len(population)
    fitnesses = [fitness(tree) for tree in population]
    best_index = min(range(n), key=fitnesses.__getitem__)
    best_overall = (fitnesses[best_index], population[best_index].copy())
    result = EvolutionResult(
        best=best_overall[1],
        best_fitness=best_overall[0],
        generations=0,
        solved=False,
    )

    completed = 0
    for generation in range(generations):
        result.history.append(best_overall[0])
        if reporter is not None:
            reporter(generation, best_overall[0], best_overall[1])
        if target is not None and best_overall[0] <= target:
            result.solved = True
            break

        order = sorted(range(n), key=fitnesses.__getitem__)
        children: list[Tree] = [population[i].copy() for i in order[:elitism]]
        while len(children) < n:
            parent_a = tournament(rng, fitnesses, population, tournament_size)
            parent_b = tournament(rng, fitnesses, population, tournament_size)
            if rng.random() < crossover_rate:
                child, _sibling = crossover(rng, parent_a, parent_b, max_depth=max_depth)
            else:
                child = parent_a.copy()
            if rng.random() < mutation_rate:
                child = mutate(child)
            children.append(child)

        population = children
        fitnesses = [fitness(tree) for tree in population]
        gen_best_index = min(range(n), key=fitnesses.__getitem__)
        if fitnesses[gen_best_index] < best_overall[0]:
            best_overall = (
                fitnesses[gen_best_index],
                population[gen_best_index].copy(),
            )
        completed = generation + 1

    result.generations = completed
    result.best = best_overall[1]
    result.best_fitness = best_overall[0]
    result.history.append(result.best_fitness)
    return result
```

A few notes on the listing.

`random_tree()` enforces both a depth cap and a depth floor. The floor matters: a population of depth-1 trees (`just "x"`) evaluates instantly and selects instantly, then evolution has nothing to select *between* once the cap-only version floods with junk.

`crossover()` returns two children but `evolve()` keeps only the first. That is the common GP convention (it halves the evaluation cost and behaves nearly identically). The depth check refuses swaps that would exceed the depth cap rather than retrying forever, and the root case is handled explicitly, because a root node has no parent to splice into.

`tournament()` draws `k` **distinct** competitors with `rng.sample()`. Drawing with replacement would let a tournament of size `k = N` miss the best individual entirely, which would break the invariant test in `tests/test_gp_core.py`.

`evolve()` keeps the best tree ever seen (`best_overall`), not merely the final generation's best. GP runs are noisy; the best individual typically appears well before the last generation.

## Example 1: Symbolic regression

Symbolic regression fits data with programs. You give GP `(x, y)` examples and an operator alphabet, and it evolves the formula. Unlike interpolation or least-squares fitting, nothing fixes the model class: GP decides which operators appear and how they nest.

Our hidden target is a damped oscillator:

```$
y(x) = 2\,e^{-x/3}\,\sin(2x)
```

sampled at 40 points on `[0, 5]` with Gaussian noise, standard deviation 0.05. The data the search actually sees looks like this (first four rows, as `make_dataset` produces them):

```text
  x=0.000  y=-0.012794
  x=0.128  y=+0.511572
  x=0.256  y=+0.889584
  x=0.385  y=+1.208017
```

Two deliberate handicaps teach the classic GP lessons:

* The function set contains **no `exp`**. GP cannot copy the textbook answer; it must find an equally cheap structure that fits the noisy samples.
* **Parsimony pressure.** Fitness is `MSE + \lambda \cdot |T|`$ with `|T|`$ the node count and `0.0005`$ per node. Adding a subtree must buy at least half a thousandth of squared error to be worth keeping. This is the cheapest known cure for bloat.

Division is a *protected* operator: when the denominator is closer to zero than `1e-10`$ it returns `1.0`$ instead of raising. Every random tree must evaluate on every fitness case, or the search stalls on exceptions. (The same protection applies to `sqrt` and `log` in fielded systems.)

```python
# Example 1 -- Symbolic Regression: Recovering a Damped Oscillator
#
# GP evolves computer programs (expression trees) that fit noisy numeric
# data. Unlike curve fitting with a FIXED model shape, GP searches over
# model STRUCTURES: the tree decides which operators appear, where, and
# with which constants. The result is a human-readable formula.
#
# This demo hides the true generating function -- a damped sine wave
#     y(x) = 2 * exp(-x/3) * sin(2x) + noise
# sampled at 40 points on [0, 5] -- and asks GP to approximate it. The
# function set below deliberately has NO `exp` primitive, so the evolved
# formula cannot be an exact copy of the truth; GP must find an equally
# good structural approximation. That is typical: symbolic regression
# rewards "right shape", not "textbook answer".
#
# Two classic GP knobs are on display:
#   * Parsimony pressure -- fitness = error + lambda * tree size, which
#     discourages bloat (the runaway growth of useless subtrees).
#   * Protected operators -- division returns 1.0 when the denominator is
#     near zero, so every random tree evaluates to a finite number.
#
# References:
#   Symbolic regression: https://en.wikipedia.org/wiki/Symbolic_regression
#   Bloat control:       Luke & Panait (2006), "A Comparison of Bloat
#                        Control Methods for Genetic Programming",
#                        https://doi.org/10.1162/evco.2006.14.3.309
#   GP field guide:      https://www.gp-field-guide.org.uk/

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
GENERATIONS: Final = 40

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
# Function set: name -> (arity, evaluation). The raw operators are partial:
# sin overflows the domain at huge inputs, mul can overflow, div can divide
# by zero. Division is protected here and evaluate() maps every non-finite
# result to 0.0, so every random tree still yields a finite float.
# ---------------------------------------------------------------------------

ArityAndEval = tuple[int, Callable[[Sequence[float]], float]]

FUNCTIONS: Final[dict[str, ArityAndEval]] = {
    "add": (2, lambda a: a[0] + a[1]),
    "sub": (2, lambda a: a[0] - a[1]),
    "mul": (2, lambda a: a[0] * a[1]),
    "div": (  # protected: |denominator| < 1e-10 -> 1.0
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
        # The run plateaus early, so print checkpoints only.
        if generation % 5 and generation != GENERATIONS - 1:
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
        generations=GENERATIONS,
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=7,
        elitism=2,
        max_depth=12,
        reporter=report,
    )

    mse = sum((evaluate(result.best, x) - y) ** 2 for x, y in data) / len(data)
    signal_power = sum(y * y for _, y in data) / len(data)
    print(f"\nBest after {result.generations} generations:")
    print(f"  formula   y = {render(result.best)}")
    print(f"  nodes     {result.best.size}, depth {result.best.depth}")
    print(
        f"  MSE {mse:.5f} vs signal power {signal_power:.5f}  "
        f"(explains {100 * (1 - mse / signal_power):.1f}% of variance)"
    )

    print("\n     x      y*      y_hat   error")
    for x in XS[::4]:
        y_hat = evaluate(result.best, x)
        print(f"  {x:5.2f}  {target(x):6.3f}  {y_hat:6.3f}  {y_hat - target(x):+6.3f}")


if __name__ == "__main__":
    main()
```

### Running it

```bash
uv sync
make regression    # or: uv run python example_1_symbolic_regression.py
```

Output (seed 7):

```text
# Symbolic regression: fit 40 noisy samples of a damped oscillator
# hidden truth: y = 2*exp(-x/3)*sin(2x), noise sd = 0.05

gen   0  best MSE  0.14600  size   7  neg((sin((x + x)) * -0.6569))
gen   5  best MSE  0.08975  size   4  sin((x + x))
gen  10  best MSE  0.08975  size   4  sin((x + x))
gen  15  best MSE  0.08975  size   4  sin((x + x))
gen  20  best MSE  0.08975  size   4  sin((x + x))
gen  25  best MSE  0.08975  size   4  sin((x + x))
gen  30  best MSE  0.08975  size   4  sin((x + x))
gen  35  best MSE  0.08975  size   4  sin((x + x))
gen  39  best MSE  0.08975  size   4  sin((x + x))

Best after 40 generations:
  formula   y = sin((x + x))
  nodes     4, depth 3
  MSE 0.08975 vs signal power 0.56206  (explains 84.0% of variance)

     x      y*      y_hat   error
   0.00   0.000   0.000  +0.000
   0.51   1.441   0.855  -0.586
   1.03   1.260   0.887  -0.373
   1.54   0.077   0.065  -0.013
   2.05  -0.827  -0.820  +0.008
   2.56  -0.778  -0.915  -0.136
   3.08  -0.092  -0.129  -0.036
   3.59   0.472   0.781  +0.309
   4.10   0.478   0.939  +0.461
   4.62   0.083   0.193  +0.110
```

### Reading the results

The best MSE is the mean of squared deviations over the 40 noisy samples,

```$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}(x_i) - y_i\right)^2
```

and the last header line compares it to the signal power `P = \frac{1}{n}\sum y_i^2`$: "explains 84.0% of variance" means `1 - \mathrm{MSE}/P`$ is 0.84.

Three things happened, and each is typical GP behavior:

1. GP found the *frequency* of the oscillator. `sin(x + x)` is `\sin(2x)`$, and notice where the 2 lives: in an addition `x + x`$, not a constant multiply. Both spellings cost the same four nodes, but `x + x` needs no lucky constant, so the search reaches it long before a constant that happens to land near `2.0`$ does. It never recovered the *damping*, because without `exp` the damping factor costs a large tree, and every large tree that might approximate `e^{-x/3}` lost the parsimony auction to the small one.
2. Improvement stopped at generation 2 and the population coasted for 37 more generations. Early convergence on a plateau is the normal case, not a bug: tournament selection with `k=7`$ plus elitism consumes diversity quickly. Raise the mutation rate, lower the tournament size, or use the `target` argument of `evolve()` (like examples 2, 3, and 4 do) to stop the waste.
3. There is no bloat in the final answer (4 nodes!). The parsimony term is small enough not to distort the error comparison (0.0005 per node versus an MSE near 0.09) but large enough that bloated mutants never reproduce.

If you add `"exp": (1, lambda a: math.exp(a[0]))` to `FUNCTIONS`, watch the run discover the damping factor. (That is practice problem 1.)

## Example 2: Boolean logic synthesis

The canonical GP problem class: evolve a program whose input/output behavior matches a specification. Here the specification is a machine safety interlock. Enable the motor only when *both* operator keys are turned on, there is *no* fault signal, and *at least one* of the two redundant sensors agrees:

```$
E = K_1 \wedge K_2 \wedge \neg F \wedge (S_1 \vee S_2)
```

The search sees the 32-row truth table and nothing else. Terminals are the five input names plus ephemeral Boolean constants `TRUE`/`FALSE` (20% of terminal draws); functions are the three gates `AND`, `OR`, `NOT`. Fitness is the fraction of truth-table rows computed incorrectly, a plain count of wrong behaviors.

Two notes before the listing. First, the constant leaves matter: parity-like and multiplexer-style targets are famously hard for GP *without* ephemeral constants, because the population spends generations discovering trivial constants. Second, note how little this problem is about syntax: the interlock above has many equivalent netlists, and GP finds one of them, not "the" one.

```python
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
```

### Running it

```bash
make logic    # or: uv run python example_2_boolean_logic.py
```

Output (seed 12):

```text
# Evolving a 5-input safety interlock from its 32-row truth table
# inputs: K1, K2 (keys), F (fault), S1, S2 (sensors)

gen   0  wrong rows  3/32  size   4  (NOT (NOT (NOT TRUE)))
gen   1  wrong rows  2/32  size  28  ((NOT ((F AND S1) OR ((F AND ((NOT K1) OR (S1 OR S2))) OR 
gen   2  wrong rows  2/32  size  28  ((NOT ((F AND S1) OR ((F AND ((NOT K1) OR (S1 OR S2))) OR 
gen   3  wrong rows  1/32  size  13  (K1 AND ((NOT (F OR (NOT (K2 AND (K1 AND K1))))) AND K1))
gen   4  wrong rows  1/32  size  13  (K1 AND ((NOT (F OR (NOT (K2 AND (K1 AND K1))))) AND K1))
gen   5  wrong rows  1/32  size  13  (K1 AND ((NOT (F OR (NOT (K2 AND (K1 AND K1))))) AND K1))
gen   6  wrong rows  1/32  size  13  (K1 AND ((NOT (F OR (NOT (K2 AND (K1 AND K1))))) AND K1))
gen   7  wrong rows  1/32  size  13  (K1 AND ((NOT (F OR (NOT (K2 AND (K1 AND K1))))) AND K1))
gen   8  wrong rows  0/32  size  18  (((NOT K1) OR (S1 OR S2)) AND ((NOT (F OR (NOT (K2 AND (K2

Perfect circuit found: True (generation 8, 18 gates, depth 8)
  ENABLE = (((NOT K1) OR (S1 OR S2)) AND ((NOT (F OR (NOT (K2 AND (K2 AND K1))))) AND K2))
  verification: 32/32 rows correct

  truth table (1 = ENABLE):
  K1 K2  F S1 S2 | spec | evolved
  -------------------------------
   0  0  0  0  0 |  0  |   0  
   0  0  0  0  1 |  0  |   0  
   0  0  0  1  0 |  0  |   0  
   0  0  0  1  1 |  0  |   0  
   0  0  1  0  0 |  0  |   0  
   0  0  1  0  1 |  0  |   0  
   0  0  1  1  0 |  0  |   0  
   0  0  1  1  1 |  0  |   0  
   0  1  0  0  0 |  0  |   0  
   0  1  0  0  1 |  0  |   0  
   0  1  0  1  0 |  0  |   0  
   0  1  0  1  1 |  0  |   0  
   0  1  1  0  0 |  0  |   0  
   0  1  1  0  1 |  0  |   0  
   0  1  1  1  0 |  0  |   0  
   0  1  1  1  1 |  0  |   0  
   1  0  0  0  0 |  0  |   0  
   1  0  0  0  1 |  0  |   0  
   1  0  0  1  0 |  0  |   0  
   1  0  0  1  1 |  0  |   0  
   1  0  1  0  0 |  0  |   0  
   1  0  1  0  1 |  0  |   0  
   1  0  1  1  0 |  0  |   0  
   1  0  1  1  1 |  0  |   0  
   1  1  0  0  0 |  0  |   0  
   1  1  0  0  1 |  1  |   1  
   1  1  0  1  0 |  1  |   1  
   1  1  0  1  1 |  1  |   1  
   1  1  1  0  0 |  0  |   0  
   1  1  1  0  1 |  0  |   0  
   1  1  1  1  0 |  0  |   0  
   1  1  1  1  1 |  0  |   0  
```

### Reading the results

Progress is honest: 3 wrong rows at generation 0 (an always-false constant circuit: `NOT(NOT(NOT TRUE))`$ mislabels the 3 rows where the interlock should ENABLE), 2 at generation 1, 1 by generation 3, then a perfect 18-gate circuit at generation 8. The verification line and the printed truth table confirm 32/32 rows.

Compare the evolved netlist with the hand-written formula. They compute the same function, but GP's answer contains `K2 AND (K2 AND K1)` where you would write `K1 AND K2`, and a `NOT (F OR (NOT ...))` wrapper that De Morgan's law collapses into the `NOT F` you would have written yourself. These are *intrinsically redundant* subexpressions: code that can never change an output is invisible to selection, so evolution keeps it. If you ship this circuit, you pass it through a Boolean simplifier first. The printed truth table is the point of the whole demo: spec column and evolved column agree on all 32 rows, and GP earned that by behavior, not by deriving anything.

## Example 3: Evolving regular expressions

GP is not restricted to numeric programs. The individual here is a *pattern program*: a tree whose nodes are regex operators (`cat`, `alt`, `opt`, `star`, `plus`) and whose leaves are pre-generalized character fragments (`D` compiles to `\d`$, `h1` to `[0-1]`$, `h3` to `[0-3]`$, and so on). Compile the tree to a regex string, run Python's `re` engine on example strings, score accept/reject.

The task: learn a validator for 24-hour clock times from 8 positive strings and 11 negative strings:

```
positive: 09:15  23:59  00:01  17:42  08:05  12:00  01:30  20:47
negative: 24:00  29:00  09:5  9:15  0915  ab:cd  09:60  1:2:3  ''  09:1a  123:45
```

Nobody tells GP about clock arithmetic. It must discover from misfit pain that hour tens digits live in `[0-1]`$, that a leading `2`$ restricts its follower to `[0-3]`$, and that minute tens digits live in `[0-5]`$.

The fitness function has three parts, and all three earned their place the hard way (see `NOTES.md`):

* **Partial credit on positives.** A rejected positive costs `1 - c`$, where `c`$ is the largest fraction of the string the pattern can fully match as a prefix. Patterns that get "23:" or "09:1" right are closer to the truth than patterns that cannot start. This ramp is what lets the search escape the "rejects everything" plateau.
* **Full price on false accepts.** Accepting any negative costs a flat 1.0, so a pattern that accepts everything stays expensive.
* **A never-matching pattern `NEVER_MATCHES`$ as a bloat guard.** Trees with nested quantifiers or more than `MAX_QUANTIFIERS`$ quantifier nodes compile to it. This is not cosmetic: Python's `re` engine backtracks exponentially on patterns like `(?:(?:(?:\d*)*)+)*`, and one such individual can stall the population evaluation for minutes. We measured this: an unguarded version of this demo hung the machine.

```python
# Example 3 -- Evolving Regular Expressions from String Examples
#
# GP where the "program" is a MATCHER instead of a computer: the evolved
# tree is a pattern program that reads a string and returns accept/reject.
# This flips the usual GP setup in an instructive way -- the phenotype is a
# piece of text-processing logic, the fitness cases are example strings,
# and the primitives are regex operators instead of arithmetic.
#
# Task: synthesize a validator for 24-hour clock times ("09:15", "23:59")
# from POSITIVE and NEGATIVE examples only. Nobody tells GP the structure
# of clock arithmetic; it has to discover that the hour's first digit is
# limited to [0-2], that the minute tens digit is limited to [0-5], and so
# on -- purely from the pain of accepting "24:00" or rejecting "09:15".
#
# Fitness is example-weighted error with a smoothing trick: an unmatched
# positive is charged (1 - best_prefix_fraction) instead of a flat 1.0 --
# how much of the string the pattern could match from the left. Accepting
# a negative still costs a full 1.0, so the search cannot cheat by
# accepting every string, while partial credit stops it from getting
# trapped in plateaus like `(?:(?:[0-2]|:))+` (rejects everything, matches
# almost nothing). A small size term breaks ties toward shorter, cleaner
# patterns.
#
# The terminal alphabet is *pre-generalized*: it contains character-class
# fragments like [0-3] and [4-9] alongside single literals, mimicking how
# real GP-regex tools build patterns from library components.
#
# References:
#   Bartoli, De Lorenzo, Medvet & Tarlao, "Playing Regex Golf with Genetic
#   Programming" (GECCO 2014):
#                   https://doi.org/10.1145/2576768.2598333
#   Bartoli, De Lorenzo, Medvet & Tarlao, "On the Automatic Construction
#   of Regular Expressions from Examples (GP vs. Humans 1-0)" (GECCO
#   2016):          https://doi.org/10.1145/2908961.2930946
#   Python re module:           https://docs.python.org/3/library/re.html

from __future__ import annotations

import functools
import random
import re
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

SEED: Final = 1  # chosen so this run converges quickly and repeatably

POSITIVE: Final[list[str]] = [
    "09:15",
    "23:59",
    "00:01",
    "17:42",
    "08:05",
    "12:00",
    "01:30",
    "20:47",
]
NEGATIVE: Final[list[str]] = [
    "24:00",  # hour out of range
    "29:00",  # hour out of range (also punishes [0-5]-for-hours)
    "09:5",  # single-digit minutes
    "9:15",  # single-digit hour
    "0915",  # missing separator
    "ab:cd",  # not digits
    "09:60",  # minute out of range
    "1:2:3",  # extra separator
    "",  # empty input
    "09:1a",  # non-digit tail
    "123:45",  # too many hour digits
]

# (negative string, cost of accepting it). All equal here; the tuple form
# leaves room to weight near-miss cases differently.
NEGATIVE_WEIGHTS: Final[list[tuple[str, float]]] = [(s, 1.0) for s in NEGATIVE]

# ---------------------------------------------------------------------------
# Pattern-program function set. Every node compiles to a regex fragment;
# unary ops wrap children in (?:...) so precedence is always safe.
# ---------------------------------------------------------------------------

PATTERN_OPS: Final[dict[str, int]] = {
    "cat": 2,  # concatenation:  a b
    "alt": 2,  # alternation:    a|b
    "opt": 1,  # optional:       a?
    "star": 1,  # Kleene star:    a*
    "plus": 1,  # one-or-more:    a+
}

# The quantifier operators, used by the anti-backtracking guard.
QUANTIFIER_OPS: Final = frozenset({"opt", "star", "plus"})

# Terminal alphabet -> regex fragment. Pre-generalized classes do most of
# the "cleverness"; GP discovers WHERE they must appear. Deliberately NO
# "." wildcard: with prefix credit below, `.` would grant every positive
# full credit and flatten the whole fitness landscape into plateaus.
TERMINAL_FRAGMENTS: Final[dict[str, str]] = {
    "D": r"\d",  # any digit
    "h1": r"[0-1]",  # hour tens digit 0 or 1
    "h2": r"[0-2]",  # ... alternative tens constraint
    "h3": r"[0-3]",  # hour units digit when the tens digit is 2
    "h9": r"[3-9]",  # hour units digit paired with a 0/1 tens digit
    "m1": r"[0-5]",  # minute tens constraint
    "m9": r"[6-9]",  # minute tens digit that makes the time invalid
    "c": r":",  # the separator, as a literal
    "d0": r"0",
    "d1": r"1",
    "d2": r"2",
    "d5": r"5",
    "d9": r"9",
}

TERMINAL_POOL: Final[list[str]] = list(TERMINAL_FRAGMENTS)

FUNCTION_ARITIES: Final[dict[str, int]] = dict(PATTERN_OPS)

SIZE_PRESSURE: Final = 0.005  # fitness = example error + 0.5% per tree node
MAX_QUANTIFIERS: Final = 4  # anti-backtracking guard (see compile_tree)
MAX_SOURCE_LEN: Final = 300

NEVER_MATCHES: Final = r"(?!a)a"  # impossible pattern: fails instantly


def terminals(rng: random.Random) -> TerminalSampler:
    """Bias leaf sampling toward the meaningful classes, not bare literals."""
    weights = {"D": 4, "h1": 4, "h2": 3, "h3": 4, "h9": 2, "m1": 4, "m9": 2, "c": 4}
    pool: list[str] = []
    for name in TERMINAL_POOL:
        pool.extend([name] * weights.get(name, 1))

    def sample() -> str:
        return rng.choice(pool)

    return sample


# ---------------------------------------------------------------------------
# Compile a pattern tree into an anchored regex string (and evaluate it).
# ---------------------------------------------------------------------------


def quantifier_stats(tree: Tree) -> tuple[int, bool]:
    """Return (number of quantifier nodes, whether any quantifier nests).

    `nested` is True when a quantifier node has another quantifier anywhere
    below it, the shape that makes the backtracking `re` engine exponential
    even on 6-character strings: `(?:(?:(?:[0-5]*)*)+)*` can hang a CPU
    forever. One post-order pass answers both questions.
    """
    total = 1 if tree.name in QUANTIFIER_OPS else 0
    nested = False
    for child in tree.args:
        child_total, child_nested = quantifier_stats(child)
        total += child_total
        if child_nested or (tree.name in QUANTIFIER_OPS and child_total):
            nested = True
    return total, nested


def compile_tree(tree: Tree) -> str:
    r"""Tree -> regex source string.

    Bloat guard: trees with more than `MAX_QUANTIFIERS` quantifier nodes,
    NESTED quantifiers, or an absurdly long source compile to a
    never-matching pattern. Without this, degenerate candidates like the
    greedy `(?:\d|:)+` or `(?:(?:[0-5]*)*)+` make fitness evaluation take
    exponential time on a six-character string -- a dramatic demonstration
    that fitness evaluation must stay computable. The exact solution here
    uses no quantifiers at all, so the guard only removes junk.

    The guard is a heuristic, not a proof: a single quantifier over an
    ambiguous alternation (for example one digit class followed by another)
    can still backtrack. It is enough because these fitness cases are at
    most six characters.
    """
    count, nested = quantifier_stats(tree)
    if count > MAX_QUANTIFIERS or nested:
        return NEVER_MATCHES
    source = _compile_node(tree)
    return source if len(source) <= MAX_SOURCE_LEN else NEVER_MATCHES


def _compile_node(tree: Tree) -> str:
    if tree.is_leaf:
        return TERMINAL_FRAGMENTS.get(tree.name, re.escape(tree.name))
    arity = PATTERN_OPS.get(tree.name)
    if arity is None:  # defensive: unknown function
        return NEVER_MATCHES
    kids = [_compile_node(child) for child in tree.args]
    if len(kids) != arity:  # defensive: malformed tree
        return NEVER_MATCHES
    match tree.name:
        case "cat":
            return kids[0] + kids[1]
        case "alt":
            return f"(?:{kids[0]}|{kids[1]})"
        case "opt":
            return f"(?:{kids[0]})?"
        case "star":
            return f"(?:{kids[0]})*"
        case "plus":
            return f"(?:{kids[0]})+"
        case _:
            raise AssertionError(f"unknown pattern op {tree.name!r}")


@functools.lru_cache(maxsize=8192)
def compile_cached(source: str) -> re.Pattern[str] | None:
    """Compile anchored regex text once per distinct string; None if invalid."""
    try:
        return re.compile(f"^(?:{source})$")
    except re.error:
        return None


def prefix_credit(tree: Tree, text: str) -> float:
    """Fraction of `text` the pattern can consume from the left (1.0 = match).

    Turns the staircase fitness into a ramp: a pattern that gets half of
    "23:59" right is closer to the truth than one that cannot even start.
    Note we must ask "can the WHOLE prefix k be matched?" (fullmatch on
    text[:k]) -- a plain .match() succeeds as long as the pattern consumes
    even one character, which would silently grant full credit.
    """
    if not text:
        return 0.0
    anchored = compile_cached(compile_tree(tree))
    if anchored is None:
        return 0.0
    for k in range(len(text), 0, -1):
        if anchored.fullmatch(text[:k]):
            return k / len(text)
    return 0.0


def matches(tree: Tree, text: str) -> bool:
    pattern = compile_cached(compile_tree(tree))
    return pattern is not None and pattern.fullmatch(text) is not None


def fitness(tree: Tree) -> float:
    r"""Weighted example error + tiny pressure toward short patterns.

    Rejecting a positive costs (1 - prefix_credit); accepting a negative
    costs its full weight -- see module docstring for why this mix keeps
    both degenerate strategies (the greedy `(?:\d|:)+` and the
    never-matching pattern) bad while still giving the search somewhere to
    climb.
    """
    error = 0.0
    for text in POSITIVE:
        error += 1.0 - prefix_credit(tree, text)
    for text, weight in NEGATIVE_WEIGHTS:
        if matches(tree, text):
            error += weight
    return error + SIZE_PRESSURE * tree.size


def render(tree: Tree) -> str:
    """Pattern source, kept short for progress lines."""
    text = compile_tree(tree)
    return text if len(text) <= 58 else text[:55] + "..."


def build_population(rng: random.Random, size: int, max_depth: int) -> list[Tree]:
    sampler = terminals(rng)
    return [
        random_tree(
            rng,
            FUNCTION_ARITIES,
            sampler,
            max_depth=max_depth,
            min_depth=3,
            method="grow",
        )
        for _ in range(size)
    ]


def main() -> None:
    rng = random.Random(SEED)
    population = build_population(rng, size=200, max_depth=9)
    sampler = terminals(rng)

    def mutate(tree: Tree) -> Tree:
        roll = rng.random()
        if roll < 0.5:
            return mutate_subtree(rng, tree, FUNCTION_ARITIES, sampler, max_depth=4)
        if roll < 0.85:
            return mutate_point(rng, tree, FUNCTION_ARITIES, sampler)
        return mutate_hoist(rng, tree)

    def report(generation: int, best_fitness: float, best: Tree) -> None:
        accepted = sum(1 for s in POSITIVE if matches(best, s))
        rejected = sum(1 for s, _w in NEGATIVE_WEIGHTS if not matches(best, s))
        print(
            f"gen {generation:3d}  fit {best_fitness:6.3f}  "
            f"pos {accepted}/{len(POSITIVE)}  neg {rejected}/{len(NEGATIVE)}  {render(best)[:40]}"
        )

    print("# Evolving a regular expression for 24-hour clock times")
    print(f"# {len(POSITIVE)} positive, {len(NEGATIVE)} negative example strings\n")

    result = evolve(
        rng,
        population,
        fitness,
        mutate,
        generations=100,
        crossover_rate=0.85,
        mutation_rate=0.12,
        tournament_size=5,
        elitism=2,
        max_depth=12,
        target=SIZE_PRESSURE * 20,  # zero example error with <= 20 nodes
        reporter=report,
    )

    best_source = compile_tree(result.best)
    print(f"\nEvolved pattern ({result.best.size} nodes, depth {result.best.depth}):")
    print(f"  ^{best_source}$")
    print(f"  all examples satisfied: {result.solved}")

    print("\n  input       expected  evolved")
    for text in POSITIVE:
        print(f"  {text:<10}  MATCH     {'MATCH' if matches(result.best, text) else 'no match'}")
    for text in NEGATIVE:
        shown = text if text else "<empty>"
        verdict = "REJECTED" if not matches(result.best, text) else "accepted (!)"
        print(f"  {shown:<10}  reject    {verdict}")

    # Show generalization: strings the GP system never saw.
    print("\n  unseen probes:")
    for text in ["05:59", "23:61", "22:00", "07:7", "11:11", "2:03"]:
        verdict = "MATCH" if matches(result.best, text) else "reject"
        truth = "valid" if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", text) else "invalid"
        print(f"  {text:<8} -> {verdict:<7} ({truth})")


if __name__ == "__main__":
    main()
```

### Running it

```bash
make regex    # or: uv run python example_3_regex_synthesis.py
```

Output (seed 1):

```text
# Evolving a regular expression for 24-hour clock times
# 8 positive, 11 negative example strings

gen   0  fit  5.430  pos 0/8  neg 11/11  (?:9[3-9]|(?:[0-3])+)
gen   1  fit  5.410  pos 0/8  neg 11/11  (?:[0-3])+
gen   2  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   3  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   4  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   5  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   6  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   7  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   8  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen   9  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  10  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  11  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  12  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  13  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  14  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  15  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  16  fit  4.830  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:[0-3])+)
gen  17  fit  4.070  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:(?:5:)+|[0-1]\d:))
gen  18  fit  3.680  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:(?:[0-3][3-9]|(?:[0-3])
gen  19  fit  3.270  pos 0/8  neg 11/11  (?:[0-3][3-9]|(?:(?:5:)+|[0-5]\d:))
gen  20  fit  3.260  pos 0/8  neg 11/11  (?:(?:[0-3][3-9]|(?:[0-3])+)(?::)+|(?:[0
gen  21  fit  2.730  pos 0/8  neg 11/11  (?:[0-3](?:[0-3](?::)+|:)[3-9]|(?:(?:5:)
gen  22  fit  2.730  pos 0/8  neg 11/11  (?:[0-3](?:[0-3](?::)+|:)[3-9]|(?:(?:5:)
gen  23  fit  2.720  pos 0/8  neg 11/11  (?:[0-3](?:[0-3](?::)+|:)[3-9]|(?:(?:[0-
gen  24  fit  2.350  pos 0/8  neg 11/11  (?:(?:[0-3](?:[0-3](?::)+|[0-3])|(?:[0-3
gen  25  fit  1.755  pos 3/8  neg 11/11  (?:(?:[0-3](?:[0-3](?::)+|[0-3])|(?:[0-3
gen  26  fit  1.755  pos 3/8  neg 11/11  (?:(?:[0-3](?:[0-3](?::)+|[0-3])|(?:[0-3
gen  27  fit  1.755  pos 3/8  neg 11/11  (?:(?:[0-3](?:[0-3](?::)+|[0-3])|(?:[0-3
gen  28  fit  1.190  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:(?:[0-3](?:[
gen  29  fit  1.170  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:(?:[0-3](?:[
gen  30  fit  1.170  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:(?:[0-3](?:[
gen  31  fit  1.170  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:(?:[0-3](?:[
gen  32  fit  1.165  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:(?:[0-3](?:[
gen  33  fit  1.135  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:[0-3](?:[0-3
gen  34  fit  1.135  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:[0-3](?:[0-3
gen  35  fit  1.125  pos 8/8  neg 10/11  (?:[0-1]\d:(?:[0-5])+|(?:(?:[0-3](?:[0-3
gen  36  fit  0.140  pos 8/8  neg 11/11  (?:[0-1]\d:(?:[0-5])+(?:[0-5])+|(?:(?:[0
gen  37  fit  0.140  pos 8/8  neg 11/11  (?:[0-1]\d:(?:[0-5])+(?:[0-5])+|(?:(?:[0
gen  38  fit  0.130  pos 8/8  neg 11/11  (?:[0-1]\d:(?:[0-5])+(?:[0-5])+|(?:[0-3]
gen  39  fit  0.130  pos 8/8  neg 11/11  (?:[0-1]\d:(?:[0-5])+(?:[0-5])+|(?:[0-3]
gen  40  fit  0.125  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5](?:[0-5])+|(?:[0-3](?:[0
gen  41  fit  0.120  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5](?:[0-5])+|(?:(?:[0-3]\d
gen  42  fit  0.120  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5](?:[0-5])+|(?:(?:[0-3]\d
gen  43  fit  0.120  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5](?:[0-5])+|(?:(?:[0-3]\d
gen  44  fit  0.105  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5][0-5]|(?:[0-3]\d:|[0-5])
gen  45  fit  0.105  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5][0-5]|(?:[0-3]\d:|[0-5])
gen  46  fit  0.105  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5][0-5]|(?:[0-3]\d:|[0-5])
gen  47  fit  0.105  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5][0-5]|(?:[0-3]\d:|[0-5])
gen  48  fit  0.095  pos 8/8  neg 11/11  (?:[0-1]\d:[0-5][0-5]|[0-3]\d:\d[3-9])

Evolved pattern (19 nodes, depth 6):
  ^(?:[0-1]\d:[0-5][0-5]|[0-3]\d:\d[3-9])$
  all examples satisfied: True

  input       expected  evolved
  09:15       MATCH     MATCH
  23:59       MATCH     MATCH
  00:01       MATCH     MATCH
  17:42       MATCH     MATCH
  08:05       MATCH     MATCH
  12:00       MATCH     MATCH
  01:30       MATCH     MATCH
  20:47       MATCH     MATCH
  24:00       reject    REJECTED
  29:00       reject    REJECTED
  09:5        reject    REJECTED
  9:15        reject    REJECTED
  0915        reject    REJECTED
  ab:cd       reject    REJECTED
  09:60       reject    REJECTED
  1:2:3       reject    REJECTED
  <empty>     reject    REJECTED
  09:1a       reject    REJECTED
  123:45      reject    REJECTED

  unseen probes:
  05:59    -> MATCH   (valid)
  23:61    -> reject  (invalid)
  22:00    -> reject  (valid)
  07:7     -> reject  (invalid)
  11:11    -> MATCH   (valid)
  2:03     -> reject  (invalid)
```

### Reading the results

Read the progress log as a story of discovery. For twenty generations the population only builds hour-fraction fragments (`[0-3][3-9]`$, `[0-1]\d:`$) because partial credit pays for prefixes. At generation 25 the best pattern already accepts 3 of 8 positives. At generation 36 it accepts all 8 positives *and* rejects all 11 negatives, and the rest of the run is pure size pressure deleting wrapper nodes (`(?:[0-5])+(?:[0-5])+`$ simplifies toward `[0-5][0-5]`$).

The final answer deserves a close look:

```
^(?:[0-1]\d:[0-5][0-5]|[0-3]\d:\d[3-9])$
```

It is *not* the textbook regex `([01]\d|2[0-3]):[0-5]\d`$. The first branch is textbook-like (`[0-1]\d`$ hours, `[0-5][0-5]`$ minutes). The second branch handles the `2`-o'clock hours in a way a human would not write: `[0-3]\d:`$ for the hours, followed by `\d[3-9]`$ for the minutes. Why is that correct on all examples? It never has to be "the minutes rule". It only has to reject `24:00`$ and `29:00`$ while accepting `23:59`$. Pairing "hours in `[0-3]`\d" with "minute units in `[3-9]`$" does exactly that on this example set, and the two branches together cover all eight positives. It satisfies the specification-as-examples. It is also *wrong* about clock times in general, and the unseen-probes section prints the confession: `22:00`$ is a valid time that this pattern rejects.

That is the overfitting lesson of this chapter, delivered automatically: GP searched until the fitness cases were exhausted, and the fitness cases were all it ever saw. More negative examples (`22:00`$ as a positive would have done it) or a held-out validation set are the standard cures.

## Example 4: Evolving plant growth rules

The last example evolves the most "program-like" individuals yet: L-system growth rules, judged by the organisms they grow. An L-system rewrites a string iteratively; here the rule is `F -> (something)`, applied four times starting from the axiom `F`, and the final string is read by a turtle:

```
F  move forward, drawing a segment
f  move forward WITHOUT drawing
+  turn left 25 degrees     -  turn right 25 degrees
[  push position and heading    ]  pop them back  (a branch)
```

A GP tree, say `seq(F, branch(seq(+, F)))`, renders to the rule text `F[+F]`. Apply it, draw it, and compare the painted cells against a target plant grown from a hidden rule. Fitness is the Jaccard distance between the two cell sets:

```$
d(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}
```

with 0.0 meaning pixel-perfect and larger values meaning less overlap. Every candidate paints the shared start cell, so the distance never quite reaches 1.0.

This demo shows the genotype-phenotype distinction more sharply than the other three. The genotype is a small rule tree (the winner below uses 25 nodes); the phenotype is a plant built from a string of thousands of characters grown by iterating one rule four times. Selection never touches the string; it only sees the cells.

One tuning lesson, recorded in `NOTES.md`: with five rewrite iterations every candidate plant saturates the grid into a dense blob, the Jaccard score stops discriminating, and runs plateau around 0.34 forever. Four iterations keep the target sparse and the landscape informative. When a GP run "cannot improve," check whether the fitness function still distinguishes good from better.

```python
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
```

### Running it

```bash
make plants    # or: uv run python example_4_lsystem_plants.py
```

Output (seed 42), including both ASCII-art plants:

```text
# Evolving an L-system growth rule to reproduce a target plant
# grid 61x31, 4 rewrite iterations, turn 25 deg

gen   0  jaccard  0.608  rule F -> [FF[f]]fF[F]FFF
gen   1  jaccard  0.538  rule F -> [[[F]]]F[[f]][[F]]FF[f][-FFF]
gen   2  jaccard  0.538  rule F -> [[[F]]]F[[f]][[F]]FF[f][-FFF]
gen   3  jaccard  0.537  rule F -> [[[[F]F]]]F[FfF]FF[-F[F]]
gen   4  jaccard  0.537  rule F -> [[[[F]F]]]F[FfF]FF[-F[F]]
gen   5  jaccard  0.467  rule F -> [[[F]]]F[[f]][F]FF[f][F[[f]][[
gen   6  jaccard  0.467  rule F -> [[[F]]]F[[f]][F]FF[f][F[[f]][[
gen   7  jaccard  0.458  rule F -> [F+F]F[FfF]FF[-F[F]]
gen   8  jaccard  0.458  rule F -> [F+F]F[FfF]FF[-F[F]]
gen   9  jaccard  0.432  rule F -> [F+F]F[Ff[+]]FF[-[F]]
gen  10  jaccard  0.395  rule F -> [F+F]F[Ff[+]]FF[[[F]]]
gen  11  jaccard  0.160  rule F -> [F+F]F[F-F]FF[f[F]]
gen  12  jaccard  0.160  rule F -> [F+F]F[F-F]FF[f[F]]
gen  13  jaccard  0.000  rule F -> [F+F]F[F-F]FF[f[+]]

secret target rule was:  F -> F[+F]F[-F]F
evolved rule:            F -> [F+F]F[F-F]FF[f[+]]
shape distance 0.000 after 13 generations (pixel-perfect: True)

TARGET PLANT (the hidden rule's phenotype):
  ............................###..............................
  .............................##..............................
  ..............................#..............................
  ..............................##.............................
  ..............................###.#..........................
  ..............................#..###.........................
  .............................#####...........................
  .............................#####...........................
  ..............................###............................
  ..............................##.............................
  ..............................###............................
  ..............................#..............................
  ..............................##.............................
  ..........................###.###............................
  ..........................##..#..............................
  ...........................####..............................
  ..........................#####..............................
  ...........................##.#..............................
  ............................###..............................
  .............................##..............................
  ..............................#..............................
  ..............................##.............................
  ..............................###............................
  ..............................#..............................
  .............................##..............................
  .............................##..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................

EVOLVED PLANT (GP's rule phenotype):
  ............................###..............................
  .............................##..............................
  ..............................#..............................
  ..............................##.............................
  ..............................###.#..........................
  ..............................#..###.........................
  .............................#####...........................
  .............................#####...........................
  ..............................###............................
  ..............................##.............................
  ..............................###............................
  ..............................#..............................
  ..............................##.............................
  ..........................###.###............................
  ..........................##..#..............................
  ...........................####..............................
  ..........................#####..............................
  ...........................##.#..............................
  ............................###..............................
  .............................##..............................
  ..............................#..............................
  ..............................##.............................
  ..............................###............................
  ..............................#..............................
  .............................##..............................
  .............................##..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................
  ..............................#..............................
```

### Reading the results

The run goes 0.608 to 0.537 to 0.467 to 0.395, then 0.160 at generation 11, then 0.000 at generation 13: thirteen generations, a few thousand evaluated plants, one perfect plant. Look at the two rules:

```
secret target rule: F -> F[+F]F[-F]F
evolved rule:       F -> [F+F]F[F-F]FF[f[+]]
```

They are *not the same rule*. They grow the same set of cells on this grid. The turtle's world is 61 by 31 integer cells and a plain `F`$ step and an `F`$ inside a pushed branch can land on identical cells when the turn angles are small; the evolved rule's trailing `[f[+]]`$ pushes an invisible stub that paints nothing. Both plants therefore have the same phenotype. The final lines print both grids so you can verify the claim yourself.

This is why GP papers insist on *behavioral* fitness evaluation and on validation examples: scoring the genotype (the rule text) would be a string-edit metric with no relationship to the plants, and scoring behavior on one example plant (as we do) is perfect only up to equivalence classes GP never sees through.

## Wrap up

All four demos share one engine (`gp_core.py`, 381 lines) and differ only in what a tree means:

* **Representation is everything.** Trees make crossover "always a valid program." Examples 1 and 2 exploit it for functions and circuits; example 3 exploits it for pattern programs; example 4 for growth rules. The same `crossover()` call splices all of them.
* **You must pay for size control.** Bloat is the default outcome of any GP setup that does not punish size: useless subtrees are free. We used parsimony terms (`\lambda \cdot |T|`$) in examples 1 and 3, a hard depth cap everywhere, hoist mutation in examples 2, 3, and 4, and a length-capped rewrite in example 4.
* **Fitness evaluation must stay computable and discriminative.** Protected operators and finite returns (example 1), never-matching guards on exponentially slow regexes (example 3), and a grid resolution coarse enough to keep plants sparse (example 4) are all examples of the same engineering duty.
* **GP optimizes fitness cases, and stops there.** The evolved regex rejects a valid time; the evolved plant rule differs from the hidden one. More fitness cases, held-out validation, and behavior-level tests are the difference between "fits the data" and "knows the rule".

The chapter-level takeaway is that GP is the right tool precisely when you want the *structure* and cannot write it down beforehand: formulas, controllers, parsers, schedules. When you only need coefficients inside a known model class, use ordinary fitting. GP is slower and its output needs curation (simplify, validate, guard), but its answers are readable programs.

For larger work, use DEAP or gplearn. Their GP modules implement this same loop (typed trees, guarded operators, tournament selection, parsimony or multi-objective size control), and you can port every experiment here by swapping `gp_core` calls for theirs.

## Practice problems

1. **Give GP `exp`.** Add `exp` (protected: `math.exp` with the argument clamped to `[-50, 50]`) to example 1's function set. Rerun and measure: does the evolved formula now express damping? Does its tree grow? What does `PARSIMONY` need to become to keep it small?
2. **NAND-only synthesis.** Rewrite example 2's function set to `{NAND: 2}`. The truth table is unchanged. NAND is functionally complete, so the optimum is reachable, but the search space is deeper. How many generations does the 32-row interlock now need? Then try parity-of-4 (`E = K_1 \oplus K_2 \oplus K_3 \oplus K_4`$) as the specification and observe why parity is the classic GP difficulty.
3. **A second regex, blindfolded.** Point example 3 at a new task (US five-digit ZIP codes plus four-digit extensions, or `dd.mm.yyyy` dates): swap `POSITIVE`/`NEGATIVE` and re-generalize `TERMINAL_FRAGMENTS`. Keep `prefix_credit` and the quantifier guard; watch where the partial credit does and does not help, and design an example set that punishes the overfit you find.
4. **Different angle, different plant.** Set example 4's `TURN_DEG` to 60, 45, 90 and rerun. At 90 degrees plants degenerate into grid-aligned snakes; at 60 the target rule becomes easier. Report which angle gives the fastest convergence and explain the result in terms of cell collisions (the mechanism behind the "same phenotype, different genotype" observation).
5. **Selection pressure dial.** In example 4 replace tournament size 6 with 2, 10, and 30 (with `k=30` you get near-greedy selection). Plot generations-to-0 against tournament size. Explain the shape of that curve with one sentence about diversity.
6. **Initialization methods.** Example 1 seeds its population with `method="half"`; examples 2-4 use `"grow"`. Rerun example 1 with `"full"`$ and with `"grow"` across five seeds each. Compare the generation-0 best MSE and the final size of the winner. Write the result in three sentences; the honest answer is "usually a little, and not reliably."
7. **Validation split.** Modify example 3 to train on a random half of the positives and report fitness on the other half. The textbook regex `([01]\d|2[0-3]):[0-5]\d`$ should now beat the two-branch overfit. That single change is the difference between a demo and a system.
