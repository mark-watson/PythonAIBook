"""Unit tests for gp_core: every GP operator must preserve well-formedness."""

import random

import pytest

from gp_core import (
    Tree,
    all_nodes,
    crossover,
    evolve,
    mutate_hoist,
    mutate_point,
    mutate_subtree,
    random_tree,
    sequence_sampler,
    tournament,
)

FUNCS = {"add": 2, "sub": 2, "sin": 1, "cos": 1}
TERMS = ["x", "y", "0.5"]


def check_valid(tree: Tree, functions: dict[str, int]) -> None:
    """Every leaf is a terminal, every internal node has the right arity."""
    if tree.is_leaf:
        assert tree.name in TERMS or tree.name == "1.25"
    else:
        assert tree.name in functions
        assert len(tree.args) == functions[tree.name]
        for child in tree.args:
            check_valid(child, functions)


def test_tree_size_depth_and_copy() -> None:
    tree = Tree("add", [Tree("x"), Tree("sin", [Tree("y")])])
    assert tree.size == 4
    assert tree.depth == 3
    clone = tree.copy()
    clone.args[0].name = "0.5"
    assert tree.args[0].name == "x"  # copy was deep
    assert str(tree) == "(add x (sin y))"


@pytest.mark.parametrize("method", ["full", "grow", "half"])
def test_random_tree_respects_bounds(method: str) -> None:
    rng = random.Random(0)
    sampler = sequence_sampler(rng, TERMS)
    for _ in range(50):
        tree = random_tree(rng, FUNCS, sampler, max_depth=5, min_depth=2, method=method)
        check_valid(tree, FUNCS)
        assert tree.depth <= 5
        assert tree.depth >= 2


def test_all_nodes_covers_every_node_once() -> None:
    rng = random.Random(42)
    sampler = sequence_sampler(rng, TERMS)
    tree = random_tree(rng, FUNCS, sampler, max_depth=6, method="grow")
    nodes = all_nodes(tree)
    assert len(nodes) == tree.size
    assert nodes[0][0] is tree
    assert nodes[0][1] is None


def test_crossover_preserves_validity_and_parents() -> None:
    rng = random.Random(1)
    sampler = sequence_sampler(rng, TERMS)
    for _ in range(30):
        a = random_tree(rng, FUNCS, sampler, max_depth=6)
        b = random_tree(rng, FUNCS, sampler, max_depth=6)
        before_a, before_b = str(a), str(b)
        child_a, child_b = crossover(rng, a, b, max_depth=17)
        assert str(a) == before_a and str(b) == before_b  # parents untouched
        check_valid(child_a, FUNCS)
        check_valid(child_b, FUNCS)


def test_mutations_preserve_validity() -> None:
    rng = random.Random(2)
    sampler = sequence_sampler(rng, TERMS + ["1.25"])
    for _ in range(30):
        tree = random_tree(rng, FUNCS, sampler, max_depth=6)
        for mutant in (
            mutate_subtree(rng, tree, FUNCS, sampler, max_depth=3),
            mutate_point(rng, tree, FUNCS, sampler),
            mutate_hoist(tree, rng),
        ):
            check_valid(mutant, FUNCS)


def test_tournament_picks_the_best_when_k_is_full() -> None:
    rng = random.Random(3)
    population = [Tree(f"t{i}") for i in range(10)]
    fitnesses = [float((i - 4) ** 2) for i in range(10)]  # min at index 4
    for _ in range(20):
        assert tournament(rng, fitnesses, population, k=10).name == "t4"


def test_evolve_finds_exact_match_on_tiny_search_space() -> None:
    """With one binary function, two terminals and shallow trees, the
    optimum (add x x) is dense enough that GP must rediscover it."""
    rng = random.Random(5)
    sampler = sequence_sampler(rng, TERMS)
    population = [random_tree(rng, FUNCS, sampler, max_depth=4) for _ in range(60)]

    def fitness(tree: Tree) -> float:
        tokens = str(tree).replace("(", " ").replace(")", " ").split()
        want = ["add", "x", "x"]
        if len(tokens) != len(want):
            return float(abs(len(tokens) - len(want))) + 3.0
        return float(sum(1 for a, b in zip(tokens, want, strict=False) if a != b))

    def mutate(tree: Tree) -> Tree:
        return mutate_point(rng, tree, FUNCS, sampler)

    result = evolve(
        rng,
        population,
        fitness,
        mutate,
        generations=30,
        crossover_rate=0.7,
        mutation_rate=0.3,
        tournament_size=3,
        elitism=1,
        max_depth=6,
        target=0.0,
    )
    assert result.solved
    assert result.best_fitness == 0.0
    assert str(result.best) == "(add x x)"
    # History must never increase across generations.
    previous, current = result.history[:-1], result.history[1:]
    assert all(later <= earlier for earlier, later in zip(previous, current, strict=True))
