"""Per-example tests: check the fitness/eval semantics on hand-built trees.

We deliberately do NOT run full GP searches here -- seeded demo output is
stable but slow, and asserting on evolved text would be brittle. Instead we
plant correct solutions by hand and confirm each example's scoring function
recognizes them, plus an AST smoke parse of every script.
"""

import ast
import random
from pathlib import Path

import pytest

import example_1_symbolic_regression as ex1
import example_2_boolean_logic as ex2
import example_3_regex_synthesis as ex3
import example_4_lsystem_plants as ex4
from gp_core import Tree

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "gp_core.py",
    "example_1_symbolic_regression.py",
    "example_2_boolean_logic.py",
    "example_3_regex_synthesis.py",
    "example_4_lsystem_plants.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)


# --- Example 1: symbolic regression ---------------------------------------


def test_ex1_evaluation_and_protection() -> None:
    tree = Tree("mul", [Tree("x"), Tree("2.5")])
    assert ex1.evaluate(tree, 4.0) == pytest.approx(10.0)
    # Protected division must never raise on a zero denominator.
    zero_div = Tree("div", [Tree("x"), Tree("0.0")])
    assert ex1.evaluate(zero_div, 3.0) == pytest.approx(1.0)


def test_ex1_fitness_rewards_truth() -> None:
    data = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    scoring = ex1.RegressionFitness(data)
    identity = Tree("x")  # perfect fit: fitness is pure parsimony penalty
    zero = Tree("0.0")  # MSE 2/3
    assert scoring(identity) == pytest.approx(ex1.PARSIMONY * 1)
    assert scoring(identity) < scoring(zero)
    # Protected ops keep evaluation finite on hostile inputs.
    assert ex1.evaluate(Tree("div", [Tree("x"), Tree("0.0")]), 1e12) == pytest.approx(1.0)
    _ = ex1.make_dataset(random.Random(0))


# --- Example 2: Boolean logic synthesis ------------------------------------


def test_ex2_perfect_netlist_scores_zero() -> None:
    # AND(AND(K1,K2), AND(NOT(F), OR(S1,S2))) -- the interlock itself.
    perfect = Tree(
        "AND",
        [
            Tree("AND", [Tree("K1"), Tree("K2")]),
            Tree("AND", [Tree("NOT", [Tree("F")]), Tree("OR", [Tree("S1"), Tree("S2")])]),
        ],
    )
    assert ex2.fitness(perfect) == 0.0
    assert ex2.evaluate(perfect, {"K1": True, "K2": True, "F": False, "S1": False, "S2": True})


def test_ex2_wrong_netlist_penalized() -> None:
    always_true = Tree("OR", [Tree("TRUE"), Tree("FALSE")])
    # The interlock is True on exactly 3/32 rows, so a constant-True
    # circuit is wrong on 29 of them.
    assert ex2.fitness(always_true) == pytest.approx(29 / 32)


# --- Example 3: regex synthesis --------------------------------------------


def _t(name: str, *kids: Tree) -> Tree:
    return Tree(name, list(kids))


def test_ex3_handbuilt_correct_pattern_has_zero_example_error() -> None:
    # ([01]\d|2[0-3]):[0-5]\d
    hours = _t("alt", _t("cat", _t("h1"), _t("D")), _t("cat", _t("d2"), _t("h3")))
    minutes = _t("cat", _t("m1"), _t("D"))
    pattern = _t("cat", _t("cat", hours, _t("c")), minutes)
    assert ex3.compile_tree(pattern) == r"(?:[0-1]\d|2[0-3]):[0-5]\d"
    # Example error is exactly 0; only the tiny size term remains.
    assert ex3.fitness(pattern) == pytest.approx(ex3.SIZE_PRESSURE * pattern.size)
    assert ex3.matches(pattern, "23:59")
    assert not ex3.matches(pattern, "24:00")


def test_ex3_greedy_pattern_is_punished() -> None:
    # alt(D, ":") repeated accepts ANY digit/colon soup: every positive
    # scores full credit and 8 of the 11 negatives sneak through -- the
    # textbook degenerate solution the example set punishes.
    greedy = _t("plus", _t("alt", _t("D"), _t("c")))
    error = ex3.fitness(greedy)
    assert error == pytest.approx(8.0 + ex3.SIZE_PRESSURE * greedy.size)


# --- Example 4: L-system rule evolution -------------------------------------


def test_ex4_rule_text_rendering() -> None:
    # seq(seq(F, branch(+F)), seq(F, branch(-F))) -> F[+F]F[-F]
    tree = _t(
        "seq",
        _t("seq", _t("F"), _t("branch", _t("seq", _t("+"), _t("F")))),
        _t("seq", _t("F"), _t("branch", _t("seq", _t("-"), _t("F")))),
    )
    assert ex4.rule_text(tree).startswith("F[+F]F[-F]")


def test_ex4_target_rule_scores_perfect() -> None:
    # Rebuild the secret target rule "F[+F]F[-F]F" as a tree.
    def leaf(text: str) -> Tree:
        node = Tree(text)
        return node

    def branch(inner: Tree) -> Tree:
        return Tree("branch", [inner])

    def seq(a: Tree, b: Tree) -> Tree:
        return Tree("seq", [a, b])

    target_tree = seq(
        seq(leaf("F"), branch(seq(leaf("+"), leaf("F")))),
        seq(seq(leaf("F"), branch(seq(leaf("-"), leaf("F")))), leaf("F")),
    )
    assert ex4.rule_text(target_tree) == ex4.TARGET_RULE
    assert ex4.jaccard_distance(target_tree) == pytest.approx(0.0)
    # A bare "F" stick plant overlaps a little but is far from the target.
    assert ex4.jaccard_distance(leaf("F")) > 0.5
