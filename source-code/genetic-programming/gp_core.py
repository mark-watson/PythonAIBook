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
#   Koza, J. (1992) "Genetic Programming on Protein" (full/grow init,
#                   tournament selection)
#   Parsimony:      https://en.wikipedia.org/wiki/Parsimony_pressure


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
    """Preorder listing of (node, parent, index-in-parent) triples.

    Nodes are returned *by reference* so callers can splice subtrees in
    place. The root's parent is None and its index is -1.
    """
    out: list[tuple[Tree, Tree | None, int]] = [(tree, None, -1)]
    stack: list[Tree] = [tree]
    while stack:
        node = stack.pop()
        # Push children in reverse so the preorder listing stays left-first.
        # Operators only need every node reachable with its parent, so the
        # exact visit order is a nicety, not a contract.
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

    The "grow" method picks a random node type at each step, weighting
    functions and terminals equally at depths where both are allowed,
    which produces the irregular trees GP is known for.
    """
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
    """Subtree mutation: replace one random node's subtree with fresh randoms."""
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


def mutate_hoist(tree: Tree, rng: random.Random) -> Tree:
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
    contenders = rng.sample(range(n), min(k, n))
    return population[min(contenders, key=fitnesses.__getitem__)]


@dataclass
class EvolutionResult:
    """Outcome of one evolve() run."""

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
    next generation by tournament-selected subtree crossover with
    post-hoc subtree mutation (B-like reproduction, one child survives).

    Parameters:
        rng             -- seeded random source
        population      -- initial list of Trees (mutated in no way)
        fitness         -- objective to minimize
        mutate          -- the mutator chosen by the example
        generations     -- budget of generations
        crossover_rate  -- probability of crossover per child (else clone)
        mutation_rate   -- probability of mutating each crossed/cloned child
        tournament_size -- k for selection tournaments
        elitism         -- number of best trees copied unchanged each gen
        max_depth       -- depth cap enforced on crossover and on all children
        target          -- stop early once best fitness <= target
        reporter        -- called as reporter(generation, best_fitness, best_tree)

    Returns:
        EvolutionResult with the best tree ever seen.
    """
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

    result.generations = generation + 1 if generations else 0
    result.best = best_overall[1]
    result.best_fitness = best_overall[0]
    result.history.append(result.best_fitness)
    return result
