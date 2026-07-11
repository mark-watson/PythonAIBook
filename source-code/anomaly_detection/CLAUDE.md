# anomaly-detection

A uv-based Python project with strict typing, automated formatting, and two-tier quality gates. Two anomaly detectors (Gaussian statistical + scikit-learn Isolation Forest) applied to the Wisconsin Diagnostic Breast Cancer dataset.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
just run     # run the Wisconsin example
```

## Workflow rules

After any Python edit, `.claude/hooks/py-check.sh` runs automatically — it formats the file with ruff, applies safe autofixes, then typechecks it with pyrefly. Fix any reported errors before moving on.

When Claude finishes a turn, `.claude/hooks/py-stop.sh` runs `just check` (full project). If it fails, Claude must fix the errors before the session ends.

Run `just check` manually at any time to verify the whole project.

## Tools

| Command | What it does |
|---------|-------------|
| `just fmt` | Format all Python files |
| `just lint` | Lint and autofix all Python files |
| `just typecheck` | Run pyrefly on the whole project |
| `just test` | Fast test run (testmon — only affected tests) |
| `just test-all` | Full parallel test run |
| `just run` | Run the Wisconsin anomaly-detection example |

## Typing discipline

- `pyrefly.toml` is set to `preset = "strict"` with `python-version = "3.13"`
- Config keys are **hyphenated** (`python-version`, not `python_version`)
- Unknown error-kind keys in `[errors]` will silently break the config — add them one at a time
- `beartype_this_package()` in `src/anomaly_detection/__init__.py` enforces types at runtime in all code paths
- `typeguard` install hook in `tests/conftest.py` enforces types during tests

## Package layout

- `src/anomaly_detection/detectors.py` — `GaussianAnomalyDetector`, `IsolationForestDetector`, `evaluate()`
- `src/anomaly_detection/wisconsin.py` — data loading, histograms, and `main()` entry point
- `cleaned_wisconsin_cancer_data.csv` at the project root is loaded via a path resolved from `wisconsin.py`'s location, so it works regardless of the CWD.
