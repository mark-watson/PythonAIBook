# TODO — openknowledge_format

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (1 test).

## Pyrefly (9 errors, all in `okf_explorer.py`)

Eight of them are the same class of issue — bare `dict` annotations without type arguments (each surfaces twice, once per missing type variable):

| Line | Bad annotation | Suggested fix |
|------|----------------|---------------|
| ~94  | `result: dict = {}` | `result: dict[str, Any] = {}` |
| ~125 | `frontmatter: dict = {}` | `frontmatter: dict[str, str] = {}` (or `dict[str, Any]` if values are mixed) |

Add `from typing import Any` at the top of the file.

The remaining error:

### `okf_explorer.py:265` — `Assistant.ask` may return `None`
```python
def ask(self, question: str) -> str:
    ...
    return response.message.content or ""
```

(or narrow with `assert response.message.content is not None`.)

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `1 passed` from pytest.
