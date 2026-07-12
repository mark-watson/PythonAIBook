# TODO — deep_learning_basics

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step because of these. Tests pass and lint is clean.

## Pyrefly (6 errors, all in `cancer_model.py`)

### 1. `CancerNet.forward` — missing `@override` decorator
`cancer_model.py:74` — overrides `nn.Module.forward` without decoration.

```python
from typing_extensions import override   # or `from typing import override` on 3.12+

class CancerNet(nn.Module):
    ...
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### 2. `CancerNet.forward` parameter `x` — untyped
Same line. Fixed by the annotation above.

### 3–6. `train_model` — all four parameters untyped
`cancer_model.py:81` — `model`, `train_loader`, `epochs`, `lr`.

```python
from torch.utils.data import DataLoader

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 60,
    lr: float = 0.01,
) -> None:
    ...
```

## Nice-to-have follow-ups

- Annotate `load_data()`'s return tuple (`tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`) — pyrefly currently infers it, but an explicit annotation makes the module easier to consume from tests.
- Consider extracting the `if __name__ == "__main__":` body into a `main()` function so a smoke test can call it with a temp directory or mock DataFrame — right now `tests/test_model.py` only exercises `CancerNet`, not the training loop.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `2 passed` from pytest.
