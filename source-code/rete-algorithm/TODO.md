# TODO — rete-algorithm

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format` reflowed 14 files, `ruff check` needed one manual fix (removed an unused `memories_before` snapshot from `rete/alpha.py`). `pytest` passes: **14 tests, ~0.34s**.

## Pyrefly (209 errors)

This is a bigger backlog than the other projects because `rete-algorithm` is a proper package (~1000 lines across `rete/`), plus 6 example scripts and 14 test cases. Almost none of it was annotated when written.

Category breakdown:

| Count | Category | Fix pattern |
|-------|----------|-------------|
| 135 | missing parameter annotations | add types to `def foo(x): ...` — start at leaves, work inward |
| 51 | `has no attribute` | usually a subtype-narrowing issue — e.g. `WME` doesn't have `.id`, but a subclass does; use `isinstance` or `cast` |
| 12 | bare generics (`list`, `dict`, `set`) | add type args (`list[str]`, `dict[str, Any]`) |
| 11 | missing `@override` decorators | `from typing import override; @override` on subclass methods |

### Recommended annotation order

1. **`rete/patterns.py`** — smallest file, defines the core primitives (`Fact`, `Pat`, `Var`, `Cond`) that everything else depends on. Getting types right here unlocks better inference elsewhere.
2. **`rete/facts.py`, `rete/tokens.py`** — pure data flow.
3. **`rete/alpha.py`, `rete/beta.py`, `rete/network.py`** — the join network. Bigger surface but constrained by the primitives you already typed.
4. **`rete/engine.py`, `rete/conflict.py`, `rete/context.py`** — driver + resolution.
5. **`example_*.py`** at repo root — user-facing demos.
6. **`rete/tests/test_engine.py`** — test callbacks like `def cheap(ctx, n):` need `(ctx: Context, n: int)`.

### Specific tripwires

- `rete/tokens.py:81, 85` — `WME` (parent) doesn't expose `.id`, but the runtime tokens always come from a subclass that does. Add `.id: int` to the `WME` dataclass, or narrow with `isinstance` at the call site.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `14 passed` from pytest.

Progress can be tracked incrementally — annotating one file at a time will drop the error count in chunks of 5–20.
