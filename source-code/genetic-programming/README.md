# Genetic Programming – Source Code

Example code for the **Genetic Programming** chapter. Four complete, self-contained GP
runs built on `gp_core.py`, a 381-line dependency-free tree-GP engine (random full/grow
initialization, subtree crossover, subtree/point/hoist mutation, tournament selection,
generational evolution with elitism). Everything is pure Python stdlib -- no NumPy, no
GP framework -- so you can read every line of the search.

Each demo evolves a different *kind* of program, covering the classic GP problem classes:

| File | Evolved object | Fitness |
|---|---|---|
| `gp_core.py` | (shared engine) -- expression trees | -- |
| `example_1_symbolic_regression.py` | a math formula | noisy MSE + parsimony pressure |
| `example_2_boolean_logic.py` | a logic-gate netlist | wrong rows of a truth table |
| `example_3_regex_synthesis.py` | a regular expression | weighted +/- string examples |
| `example_4_lsystem_plants.py` | an L-system growth rule | cell overlap (Jaccard) with a target plant |

## Run the demos

```bash
uv sync
make regression   # rediscover a damped oscillator from 40 noisy samples
make logic        # synthesize a 5-input safety interlock from its truth table
make regex        # evolve a 24-hour clock-time validator from example strings
make plants       # grow L-system rules toward a hidden target plant (ASCII art)
make run          # all four
```

Every demo is seeded (`SEED` constant) and reproducible; each finishes in a few seconds.
Try changing `SEED`, the population size, `PARSIMONY`/`SIZE_PRESSURE`, or the function
sets -- the examples are deliberately small enough to experiment with.

## What each example demonstrates

- **Example 1 -- Symbolic regression.** Fit noisy data with programs, not fixed model
  shapes. Highlights: Ephemeral Random Constants (ERCs), *protected* division plus a
  finite-value check in `evaluate()` so every tree produces a number, and parsimony
  pressure (fitness = error + λ·size) fighting tree bloat. The function set hides `exp`
  on purpose, so GP finds an equivalent *structure* rather than the textbook formula.
  It recovers the frequency (`sin(2x)`) but not the damping; `NOTES.md` explains why.
- **Example 2 -- Boolean logic synthesis.** The canonical Koza-style problem: evolve a
  gate netlist whose I/O behavior matches a 32-row truth table (a machine interlock:
  both keys on, no fault, at least one of the two redundant sensors agreeing). Fitness
  scores *behavior*, not syntax, so many structurally different circuits earn the same
  perfect score.
- **Example 3 -- Regular-expression evolution.** GP where the program is a *matcher*.
  Trees are pattern programs (concat, alt, opt, star, plus over character-class
  fragments) compiled to Python `re` strings; balanced positive/negative example sets
  stop the search from winning by accepting every string. Shows how GP learns general
  rules from labeled examples, and how easy it is to overfit them.
- **Example 4 -- L-system plants.** GP on developmental programs: a tiny rule tree is
  rewritten over four iterations into a turtle-drawn plant, scored by cell overlap with
  a hidden target plant. Genotype ≠ phenotype, and the fitness landscape is
  pleasingly smooth -- plants get taller, bushier, and better each generation.

## Tests

```bash
just check    # ruff format + lint + pyrefly (strict) + pytest
```

The 28 tests plant hand-built perfect solutions (the true interlock netlist, the correct
clock regex `([01]\d|2[0-3]):[0-5]\d`, the target L-system rule) and assert each example's
fitness function recognizes them, plus invariant and argument-validation tests for the
engine -- property checks instead of brittle seed snapshots.

## Further reading

- Koza, *Genetic Programming: On the Programming of Computers by Means of Natural
  Selection* -- https://en.wikipedia.org/wiki/Genetic_programming
- Poli, Langdon & McPhee, *A Field Guide to Genetic Programming* --
  https://www.gp-field-guide.org.uk/
- For production-scale GP, see the DEAP (`deap.readthedocs.io`) and gplearn
  (`gplearn.readthedocs.io`) libraries; this chapter's `gp_core.py` implements
  the same algorithms from first principles.

See `NOTES.md` for the bugs found while building this code and the known limitations of
the current version.
