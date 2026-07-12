# Anomaly Detection – Source Code

Example code for the **Anomaly Detection** chapter of *Practical Artificial Intelligence Programming With Python*. Two complementary approaches are applied to the Wisconsin Diagnostic Breast Cancer dataset:

1. **Gaussian Statistical Detector** — A from-scratch implementation that models per-feature Gaussian distributions (mean + variance) and flags observations whose aggregate probability falls below a tunable epsilon threshold. Direct port of the Java version from the companion *Java AI Book*.
2. **Isolation Forest Detector** — A scikit-learn wrapper around the industry-standard Isolation Forest algorithm, which isolates anomalies by random partitioning without needing labelled data.

The project uses a modern uv-based workflow with strict static typing (Pyrefly), automatic formatting and linting (Ruff), runtime type enforcement (Beartype + Typeguard), and property-based testing (Hypothesis) — all wired together with `just` and Claude Code hooks that run automatically on every file edit.

## TL;DR — copy-paste bootstrap

If you have Python 3.13+ and Homebrew, one block gets you from zero to a working checkout:

```bash
# 1. Install the toolchain (skip anything you already have)
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
brew install just                                 # just (Rust task runner)

# 2. Set up the project
uv sync                                           # creates .venv, installs deps
just check                                        # fmt-check + lint + typecheck + test
just run                                          # runs the Wisconsin example
```

If any step fails, jump to [Troubleshooting](#troubleshooting).

## Project layout

```
.
├── src/anomaly_detection/
│   ├── __init__.py       # beartype_this_package() — runtime type enforcement
│   ├── detectors.py      # GaussianAnomalyDetector, IsolationForestDetector, evaluate
│   └── wisconsin.py      # data loading, histograms, main() entry point
├── tests/
│   ├── conftest.py       # typeguard install_import_hook
│   └── test_detectors.py # unit tests + Hypothesis property-based tests
├── cleaned_wisconsin_cancer_data.csv  # 648-sample dataset (9 features + class)
├── FIG_anomaly_detection.jpg          # architecture diagram
├── pyproject.toml        # project metadata and dev dependencies
├── pyrefly.toml          # strict type-checking config
├── justfile              # task runner (fmt, lint, typecheck, test, run)
├── Makefile              # thin fallback (clean, test, run)
├── CLAUDE.md             # workflow rules for Claude Code
└── .claude/
    ├── settings.json     # hook wiring
    └── hooks/
        ├── py-check.sh   # per-edit gate: format → lint → typecheck
        └── py-stop.sh    # end-of-turn gate: full just check
```

## Requirements

- **Python 3.13+** — uv will download this automatically if you don't have it
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (replaces pip + venv + pyenv)
- **[just](https://just.systems/)** — Rust-based command runner (**not** the Python `just` package on PyPI)

## Installation

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify: `uv --version` should print `uv 0.x.x`.

### 2. Install just (the Rust task runner)

```bash
# macOS
brew install just

# Any other platform (needs Rust)
cargo install just
# or the standalone installer:
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

Verify: `just --version` should print `just 1.x.x`.

> **Note:** There is a Python package also named `just` on PyPI. If you have run `uv tool install just` or `pip install just`, that package will shadow the Rust `just` and the `justfile` commands will fail. Install the Rust version via brew or the script above and make sure it appears first on your `PATH`.

### 3. Install project dependencies

From the project root:

```bash
uv sync
```

This creates a project-local `.venv` and installs all runtime deps (numpy, pandas, scikit-learn, matplotlib) plus dev deps (pyrefly, ruff, pytest, hypothesis, beartype, typeguard, etc.). Re-run `uv sync` any time `pyproject.toml` or `uv.lock` changes.

### 4. Verify the setup

```bash
just check
```

You should see four green stages complete: `fmt-check`, `lint`, `typecheck`, `test`. If that passes, you're ready to go.

## Updating an existing checkout

If you cloned this repo earlier when it used the flat file layout (`anomaly_detection.py` + `wisconsin_anomaly.py` at the root), do this to catch up:

```bash
git pull                          # pull the src/-layout refactor
rm -rf .venv                      # discard the old virtualenv (deps have changed)
uv sync                           # rebuild the venv from the current lockfile
just check                        # verify everything works end-to-end
```

If your editor still shows import errors after this, restart the language server or reload the window so it picks up the new `.venv`.

## Running the example

```bash
just run
# equivalent to: uv run python -m anomaly_detection.wisconsin
```

This produces:

- Console output with precision, recall, and F1 for both detectors
- `histograms.png` — a 3×3 grid of per-feature histograms colour-coded by class
- A "quick demo predictions" block showing each detector's verdict on a synthetic malignant and benign sample

## Interactive use

```bash
uv run python
```

```python
from anomaly_detection import GaussianAnomalyDetector, IsolationForestDetector, evaluate
from anomaly_detection.wisconsin import load_wisconsin_data

X_train, X_cv, y_cv, X_test, y_test, _, _ = load_wisconsin_data()

gauss = GaussianAnomalyDetector()
gauss.fit(X_train, y_cv, X_cv)
evaluate("Gaussian", y_test, gauss.predict(X_test))

iforest = IsolationForestDetector(contamination=0.35)
iforest.fit(X_train)
evaluate("Isolation Forest", y_test, iforest.predict(X_test))
```

## Development workflow

| Command | What it does |
|---------|-------------|
| `just fmt` | Format all Python files with ruff |
| `just fmt-check` | Check formatting without modifying files |
| `just lint` | Lint and apply safe autofixes |
| `just typecheck` | Run pyrefly on the whole project |
| `just test` | Fast test run — testmon, only affected tests |
| `just test-all` | Full parallel test run (pytest -n auto) |
| `just run` | Run the Wisconsin example |
| `just check` | fmt-check + lint + typecheck + test (the full gate) |

`just check` is the same gate that runs automatically when Claude Code ends its turn.

## Editor / IDE setup

Point your editor at the project's `.venv` so autocomplete, go-to-definition, and inline type errors all work.

### VS Code / Cursor

1. Install the **Python** extension (Microsoft), the **Ruff** extension (Astral), and the **Pyrefly** extension.
2. Open the folder. When VS Code asks to select an interpreter, pick `./.venv/bin/python` (or use `Cmd/Ctrl+Shift+P → Python: Select Interpreter`).
3. Add to your workspace `settings.json`:
   ```json
   {
     "python.defaultInterpreterPath": ".venv/bin/python",
     "python.testing.pytestEnabled": true,
     "[python]": {
       "editor.defaultFormatter": "charliermarsh.ruff",
       "editor.formatOnSave": true,
       "editor.codeActionsOnSave": { "source.fixAll.ruff": "explicit" }
     }
   }
   ```

### PyCharm / IntelliJ

- **Preferences → Project → Python Interpreter → Add → Existing environment** and select `.venv/bin/python`.
- Install the **Ruff** and **Pyrefly** plugins from the marketplace and enable format-on-save.

### Neovim / Vim (LSP)

- Configure `pyright` or `pyrefly` to use `.venv/bin/python` as the Python interpreter (via `pyproject.toml` or an `.envrc`).
- Add `ruff-lsp` as a language server for formatting and linting.

## Type safety layers

This project uses three complementary type-checking mechanisms:

1. **Pyrefly** (`pyrefly.toml`, `preset = "strict"`) — static analysis at edit time and in CI
2. **Beartype** (`src/anomaly_detection/__init__.py`) — zero-overhead runtime enforcement in all production code paths via `beartype_this_package()`
3. **Typeguard** (`tests/conftest.py`) — runtime enforcement during tests via the import hook, catching gaps that slip past static analysis

## Claude Code integration

When used with [Claude Code](https://claude.com/claude-code), two hooks run automatically:

**Per-edit hook** (`.claude/hooks/py-check.sh`) — fires after every `Write` or `Edit` on a `.py` file:

1. `ruff format` — auto-formats the file
2. `ruff check --fix` — applies safe lint autofixes
3. `pyrefly check` — typechecks the single file; exits with code 2 to feed errors back to Claude as actionable feedback

**End-of-turn hook** (`.claude/hooks/py-stop.sh`) — fires when Claude finishes its turn:

- Runs the full gate (`ruff format --check`, `ruff check`, `pyrefly check`, `pytest`) using the project's `.venv` binaries directly
- If it fails, Claude must resolve the errors before the session ends
- Guarded against infinite stop→fix→stop loops via `stop_hook_active`

See `CLAUDE.md` for the workflow rules Claude follows in this repo.

## Pyrefly config notes

`pyrefly.toml` uses hyphenated keys (`python-version`, not `python_version` — the underscore form is silently ignored). Unknown keys in the `[errors]` table cause the entire config to fail silently and fall back to defaults, so add error-kind entries one at a time.

## Troubleshooting

**`just: command not found`** — `just` isn't installed, or the Python PyPI `just` package is shadowing the Rust one. Run `which just`; if it points into a venv or site-packages, uninstall it (`pip uninstall just`, `uv tool uninstall just`) and reinstall with `brew install just`.

**`ModuleNotFoundError: No module named 'anomaly_detection'`** — the venv isn't in sync with the source layout. Run `uv sync`. If it still fails, delete `.venv` and re-run `uv sync`.

**`FileNotFoundError: cleaned_wisconsin_cancer_data.csv`** — the CSV path is resolved relative to `src/anomaly_detection/wisconsin.py`, so it works from any CWD as long as the CSV lives at the project root. Confirm the file is present with `ls cleaned_wisconsin_cancer_data.csv`.

**Ruff reports a formatting diff but you didn't touch the file** — run `just fmt` to apply the change. `just fmt-check` (used by `just check`) is read-only; `just fmt` is the fix.

**Pyrefly complains about missing type stubs for numpy / sklearn** — these libraries ship their own type information, but if you see stub-missing errors, run `uv sync` again to make sure the versions in `.venv` match the lockfile.

**Editor still shows red squigglies after `uv sync`** — the language server is caching the old interpreter path. Reload the window (VS Code: `Developer: Reload Window`) or restart the LSP.

**Slow `uv sync` on first run** — uv downloads Python 3.13 and all the wheels on the first invocation; subsequent syncs use the local cache and finish in a second or two.

## Architecture

![Anomaly detection pipeline architecture](FIG_anomaly_detection.jpg)

## Copyright and License

Copyright 2024-2026 Mark Watson. All rights reserved.

This example is released using the Apache 2 license.
