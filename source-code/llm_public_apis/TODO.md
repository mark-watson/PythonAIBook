# TODO — llm_public_apis

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (13 script-parse tests).

All 9 pyrefly errors are SDK-stub friction of two kinds:

1. **Missing parameter annotations** on top-level `chat` helper functions.
2. **Optional-return narrowing** — `openai` / `google-genai` SDK return types are `Optional[...]` for `.content`, `.text`, etc., and the demo scripts assume the happy path.

## Pyrefly (9 errors)

### 1. `fireworks_conversation.py:21` and `gemini_conversation.py:20` — untyped `chat(user_message)`
```python
def chat(user_message: str) -> str:
    ...
```

### 2. `fireworks_conversation.py:26` — `messages=` list is `list[dict[str, str]]`, expected `Iterable[ChatCompletionMessageParam]`
Cast the list to satisfy the OpenAI stubs:

```python
from typing import cast
from openai.types.chat import ChatCompletionMessageParam

response = client.chat.completions.create(
    model=MODEL,
    messages=cast(list[ChatCompletionMessageParam], messages),
)
```

### 3. `gemini_conversation.py:26` — same shape, `contents=conversation`
`google-genai`'s `generate_content` accepts many input shapes but the stubs don't include our `list[Content]` variant. Same fix pattern — cast:

```python
from typing import cast, Any

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=cast(Any, conversation)
)
```

### 4. `gemini_image.py:25` — `contents=[prompt, image]` with an unloaded `Image`
Same category as #3 — cast to `Any` at the call site.

### 5. `fireworks_structured.py:33` — `response.choices[0].message.content` is `str | None`
```python
raw = (response.choices[0].message.content or "").strip()
```

### 6. `gemini_structured.py:31` — `response.text` is `str | None`
```python
raw = (response.text or "").strip().removeprefix("```json").removesuffix("```").strip()
```

### 7. `openai_search.py:28` and `openai_text.py:22` — iterating `item.content` on a huge union type
`response.output` is a union of ~24 output-item shapes; only a few have `.content`. The safe move is to filter by type:

```python
from openai.types.responses import ResponseOutputMessage

for item in response.output:
    if not isinstance(item, ResponseOutputMessage):
        continue
    for content in item.content:
        ...
```

Or, if you know the demo only ever returns messages, a targeted `# type: ignore[missing-attribute]` is defensible.

## Nice-to-have follow-ups

- None of the 13 scripts guard their work under `if __name__ == "__main__":`, so `test_smoke.py` `ast.parse`s each file rather than importing it. If you refactor any of them into a `main()` guarded by `__name__`, promote its entry in `test_smoke.py::SCRIPTS` to a real import test.
- Every `fireworks_*` script duplicates the `client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=...)` boilerplate. If you add a fourth Fireworks script, factor it into a shared `_fireworks.py` helper.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `13 passed` from pytest.
