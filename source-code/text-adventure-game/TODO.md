# TODO — text-adventure-game

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (1 test).

## Pyrefly (4 errors)

All 4 in `game.py`. Two are the same untyped-`dict` issue; one is the recurring OpenAI-SDK optional-return; one is a downstream `bad-argument-type`.

### `game.py:41` — `list[dict]` needs type arguments (produces 2 errors on the same line)
```python
def get_ai_response(client: OpenAI, messages: list[dict[str, str]]) -> str:
```

### `game.py:45` — `messages=` list is not assignable to `Iterable[ChatCompletionMessageParam]`
Fixed by the correct `dict[str, str]` annotation above plus a cast at the call site:

```python
from typing import cast
from openai.types.chat import ChatCompletionMessageParam

response = client.chat.completions.create(
    model=MODEL,
    messages=cast(list[ChatCompletionMessageParam], messages),
)
```

### `game.py:47` — `response.choices[0].message.content` is `str | None`
```python
return response.choices[0].message.content or ""
```

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `1 passed` from pytest.
