# NOTES -- problems found (and fixed) in this example code

Development log of real bugs and design failures hit while building
`gp_core.py` and the four examples. Each item is either a bug that is now
fixed (kept here because the fix is instructive) or a known limitation of
the current code.

## Bugs fixed during development

1. **gp_core: type aliases referenced `Tree` before it was defined.**
   `Mutator = Callable[[Tree], Tree]` is a runtime statement, not an
   annotation, so `from __future__ import annotations` does not save you.
   NameError at import. Fix: define the aliases after the class.

2. **gp_core: subtree crossover crashed when a selected node was the root.**
   Root nodes have `parent = None`, so `parent.args[index] = other` blew up.
   Fix: when a crossover point is the root, the whole other subtree becomes
   the child (see "Splicing at the root" in `crossover()`).

3. **gp_core: crossover shared subtrees between both offspring.**
   Swapping `node_a`/`node_b` by reference made children share mutable
   nodes. Nothing writes in place later, but one stray mutation-in-place
   would corrupt both children. Fix: splice copies.

4. **gp_core: "grow" initialization actually produced only full trees.**
   The original terminal-vs-function coin flip was gated on a depth
   condition that was false during early growth, so `grow` behaved like
   `full`. Fix: branch only when `depth < min_depth`, otherwise take a
   terminal with probability 0.4 (`random_tree.build()`).

5. **gp_core: tournament selection drew with replacement.**
   `rng.randrange` repeated `k` times can pick the same individual twice,
   so a `k = len(population)` tournament could still miss the best. Fix:
   `rng.sample(range(n), k)` (distinct draws).

6. **gp_core: `min(zip(fitnesses, population))` crashed on fitness ties.**
   When two fitnesses are equal, Python compares the second tuple element,
   and `Tree` has no order. Fix: `min(range(n), key=fitnesses.__getitem__)`.

7. **example_2: fitness passed raw truth-table tuples into `evaluate()`,**
   which indexes the input row by *name* (`row["K1"]`). TypeError. Fix:
   build `dict(zip(INPUTS, row, strict=True))` per row.

8. **example_3: the demo hung the whole machine for minutes.**
   Evolved trees like `(?:(?:(?:[0-5]*)*)+)*` and `.*.*.*.*.*.*.*` are
   valid regexes, and Python's backtracking `re` engine takes exponential
   time matching them against even a 6-character string. One bloated
   candidate froze the population evaluation. Fix: `compile_tree()` refuses
   nested quantifiers, quantifier count > `MAX_QUANTIFIERS`, or source
   longer than `MAX_SOURCE_LEN` by compiling to `NEVER_MATCHES`. The true
   solution uses no quantifiers, so the guard prunes only junk.

9. **example_3: prefix credit silently granted full credit.**
   `open_pat.match(text[:k])` succeeds whenever the pattern consumes even
   ONE character from the left, so nearly every candidate scored error 0 on
   positives and the run "solved" at generation 0 with garbage. Fix: test
   `fullmatch(text[:k])` -- the whole prefix must be consumed.

10. **example_3: the `.` wildcard terminal flattened the fitness landscape.**
    With prefix credit, any pattern containing `.` can consume any positive,
    so `.` earns full positive credit. Runs plateaued hard at patterns like
    `[0-5]\d:[0-5]\d` (error exactly 1.0: it accepts "24:00") for all 20
    seeds tested. Fix: removed the `.` fragment entirely and added the
    negative "29:00" so digit-class `[0-5]`-for-hours is also punished.
    Lesson: partial-credit fitness and wildcard primitives interact badly.

11. **example_3: crash at the end of `main()`** -- `for text, _weight in
    NEGATIVE` unpacked strings (the tuples live in `NEGATIVE_WEIGHTS`).
    ValueError after all interesting output was already printed.

12. **example_4: pixel-perfection was unreachable at 5 iterations.**
    With `ITERATIONS = 5` the target plant saturates the 61x31 grid (3125
    steps), every candidate is a dense blob, and Jaccard stalls around
    0.34-0.35 for every seed. Fix: 4 iterations (625 steps): the plant
    stays sparse, the landscape stays discriminative, and GP now hits
    Jaccard 0.0 in about 14 generations.
    Diagnosis method: render the target grid for candidate settings BEFORE
    wiring GP to them.

13. **Toolchain:** ruff 0.16 enables rules beyond E/F by default for some
    configs; this project pins `select = ["E", "F", "UP", "B", "SIM", "I"]`
    in `pyproject.toml` to match the other chapter projects. pyrefly's
    `non-exhaustive-match` check forced a `case _` branch in
    `example_3._compile_node()`.

## Known limitations of the current code (not bugs, but real)

* **Example 1 plateaus at the undamped oscillator.** The function set has
  no `exp`, and approximating `exp(-x/3)` from `sin/cos/div/mul` needs a
  big tree, which parsimony pressure fights. Result: `sin(x + x)` =
  `sin(2x)`, 84.0% of variance explained, found by generation 2, then
  nothing. This is honest GP behavior (early convergence + parsimony
  tradeoff) but readers should know the demo does NOT recover the damping.
  Add `"exp"` to `FUNCTIONS` to watch it snap to something close to the
  truth formula.
* **Example 3's evolved regex overfits the examples.** It rejects the
  valid time "22:00" (see the unseen-probes section). It satisfies every
  fitness case; nothing in GP guarantees correctness beyond them.
* **Example 4's "pixel-perfect" rule differs from the hidden one.**
  `[F+F]F[F-F]FF[f[+]]` draws the same cells as `F[+F]F[-F]F` on a discrete
  grid (the `f`/`[+]` stubs paint nothing visible). Fitness sees cells, not
  rules: genotypically different, phenotypically identical.
* **`evolve()` early-stops on the all-time best, not the current
  generation's best**, so `result.generations` can lag one generation
  behind the improvement. Harmless for these demos.
* **Bloat guards are problem-specific.** `MAX_EXPANSION` (ex 4),
  `MAX_QUANTIFIERS` (ex 3), `PARSIMONY` (ex 1) and `max_depth` (all) are
  hand-tuned per problem. A different target may need different values.
* **Tournament selection with `rng.sample` is O(k) per draw plus a sort of
  the whole population each generation**; fine at N <= 400, but this core
  is a teaching engine, not a production one. For real work use DEAP or
  gplearn.
