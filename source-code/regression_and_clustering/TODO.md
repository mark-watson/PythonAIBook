# TODO — regression_and_clustering

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (2 tests).

## Pyrefly (9 errors)

Seven errors are the same class of issue — sklearn's `fetch_california_housing` / `load_iris` returns a `Bunch`, but its default overload (called with no `return_X_y=True`) is typed as returning a bare tuple in the stubs. This surfaces as "object of class tuple has no attribute data / feature_names / target / target_names":

- `regression.py:19–21`  (fetch_california_housing)
- `clustering.py`  (load_iris — 4 sites)

Fix: pass the overload disambiguator or cast:

```python
from sklearn.utils import Bunch
from typing import cast

housing = cast(Bunch, fetch_california_housing())
```

(Or use `fetch_california_housing(as_frame=True)` and consume the DataFrame directly — that path is typed more precisely.)

The remaining errors:

### `regression.py:25` and `:56` — untyped `df` parameter
```python
import pandas as pd

def run_linear_regression(df: pd.DataFrame) -> None:
    ...

def run_polynomial_regression(df: pd.DataFrame) -> None:
    ...
```

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `2 passed` from pytest.
