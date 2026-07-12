# TODO — reinforcement_learning

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (2 tests).

## Runtime dependency fix already applied

The original `pyproject.toml` pinned `pymdptoolbox>=4.0.0`, but PyPI only publishes up to `4.0b3`, so `uv sync` was unsatisfiable. The pin was loosened to `pymdptoolbox>=4.0b3` during setup — no other change to runtime deps.

## Pyrefly (19 errors)

The errors cluster in two files. Categories:

### `frozen_lake_qlearning.py` — untyped `q_learning` / `evaluate` / `print_policy` parameters (~12 errors)

Sample fix:

```python
import gymnasium as gym
import numpy as np

def q_learning(
    env: gym.Env,
    episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
) -> np.ndarray:
    ...

def evaluate(env: gym.Env, Q: np.ndarray, episodes: int = 100) -> float: ...
def print_policy(env_name: str, Q: np.ndarray) -> None: ...
```

### `frozen_lake_qlearning.py` — `env.action_space.n` / `env.observation_space.n`
`gymnasium.spaces.Space` does not have `.n` — only `Discrete` does. Narrow at the call site:

```python
from gymnasium.spaces import Discrete
assert isinstance(env.action_space, Discrete)
n_actions: int = env.action_space.n
```

### `mdp_demo.py` — `np.round(vi.V, 2)` where `vi.V` is `Unknown | None`
`pymdptoolbox`'s `.V` attribute isn't well-typed in the stubs. Two options:

```python
# Option A — assert non-None
V = vi.V
assert V is not None
print(f"Value function: {np.round(V, 2)}")

# Option B — cast to ndarray at the call site
from typing import cast
import numpy as np
print(f"Value function: {np.round(cast(np.ndarray, vi.V), 2)}")
```

### `mdp_demo.py` — "Cannot index into dict"
Comes from indexing an untyped policy/return-dict. Give the local variable a `dict[str, np.ndarray]` (or similar) annotation at the assignment site.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `2 passed` from pytest.
