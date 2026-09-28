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
# emitting `.*`, while partial credit stops it from getting trapped in
# plateaus like `(?:(?:[0-2]|:))+` (rejects everything, matches almost
# nothing). A small size term breaks ties toward shorter, cleaner patterns.
#
# The terminal alphabet is *pre-generalized*: it contains character-class
# fragments like [0-3] and [4-9] alongside single literals, mimicking how
# real GP-regex tools build patterns from library components.
#
# References:
#   Forster & Sigurðsson, "Semantic Regex Search" (genprog-style regex
#   synthesis): https://dl.acm.org/doi/10.1145/2908812.2908862
#   Foster et al., "gentype" generalizations:
#                   https://link.springer.com/chapter/10.1007/978-3-642-32954-4_6
#   Python re module:           https://docs.python.org/3/library/re.html

from __future__ import annotations

import functools
import random
import re
from typing import Final

from gp_core import (
    TerminalSampler,
    Tree,
    all_nodes,
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
    "29:00",  # hour out of range (also punishes the [0-5] dot-plateau)
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

# Terminal alphabet -> regex fragment. Pre-generalized classes do most of
# the "cleverness"; GP discovers WHERE they must appear. Deliberately NO
# "." wildcard: with prefix credit below, `.` would grant every positive
# full credit and flatten the whole fitness landscape into plateaus.
TERMINAL_FRAGMENTS: Final[dict[str, str]] = {
    "D": r"\d",  # any digit
    "h1": r"[0-1]",  # hour tens digit 0 or 1
    "h2": r"[0-2]",  # ... alternative tens constraint
    "h3": r"[0-3]",  # hour units digit when the tens digit is 2
    "h9": r"[3-9]",  # hour tens digit when units is free
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


def count_quantifiers(tree: Tree) -> int:
    """Number of star/plus/opt nodes in the tree."""
    return sum(1 for node, _p, _i in all_nodes(tree) if node.name in ("opt", "star", "plus"))


def has_nested_quantifiers(tree: Tree) -> bool:
    """True if any quantifier wraps another quantifier.

    Nesting is what makes the backtracking `re` engine exponential even on
    6-character strings: `(?:(?:(?:[0-5]*)*)+)*` can hang a CPU forever.
    """
    return any(
        node.name in ("opt", "star", "plus") and count_quantifiers(node) > 1
        for node, _p, _i in all_nodes(tree)
    )


def compile_tree(tree: Tree) -> str:
    """Tree -> regex source string.

    Bloat guard: trees with more than `MAX_QUANTIFIERS` quantifier nodes,
    NESTED quantifiers, or an absurdly long source compile to a
    never-matching pattern. Without this, degenerate candidates like
    `.*.*.*.*.*.*.*` or `(?:(?:X+)*)+` make fitness evaluation take
    exponential time on a six-character string -- a dramatic demonstration
    that fitness evaluation must stay computable. The exact solution here
    uses no quantifiers at all, so the guard only removes junk.
    """
    if count_quantifiers(tree) > MAX_QUANTIFIERS or has_nested_quantifiers(tree):
        return NEVER_MATCHES
    source = _compile_node(tree)
    return source if len(source) <= MAX_SOURCE_LEN else NEVER_MATCHES


def _compile_node(tree: Tree) -> str:
    if tree.is_leaf:
        return TERMINAL_FRAGMENTS.get(tree.name, re.escape(tree.name))
    kids = [_compile_node(child) for child in tree.args]
    arity = PATTERN_OPS[tree.name]
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
    """Weighted example error + tiny pressure toward short patterns.

    Rejecting a positive costs (1 - prefix_credit); accepting a negative
    costs its full weight -- see module docstring for why this mix keeps
    both degenerate strategies (`.*` and the never-matching pattern) bad
    while still giving the search somewhere to climb.
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
        return mutate_hoist(tree, rng)

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
