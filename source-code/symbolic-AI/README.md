# Symbolic AI – Source Code

This directory contains example code for the **Symbolic AI** chapter.

## Pure Python

- **frame.py** — Lisp-like frame data structures with nested subframes and a BookShelf search container.

## Python + Swi-Prolog

Requires [Swi-Prolog](https://www.swi-prolog.org/download/stable) and the Python bridge:

```bash
brew install swi-prolog   # macOS
uv pip install swiplserver
```

- **n_queens.py** / **n_queens.pl** — Solve the 8-queens problem using Prolog's constraint library (clpfd).
- **family.py** / **family.pl** — Assert family facts and query grandparent relationships via Prolog rules.
- **hackernews.py** — Fetch Hacker News stories, extract entities with spaCy, and assert them as Prolog facts.

## Python + MiniZinc (Constraint Programming)

Requires [MiniZinc](https://www.minizinc.org/):

```bash
brew install minizinc   # macOS
uv pip install minizinc
```

- **test_mzn.py** / **test_mzn.mzn** — Simple constraint satisfaction: find x, y such that x+y=n and x*y=m.
- **us_states.py** / **us_states.mzn** — Four-color map coloring of US states.

## Soar Cognitive Architecture

Install the Python bindings:

```bash
uv pip install soar-sml
```

- **bw.py** — Blocks world example using the Soar cognitive architecture from Python.
