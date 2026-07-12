# TODO — semantic_web_LLM

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (4 tests).

## Pyrefly (23 errors)

Categories:

- **18 bare-`dict` annotations** — same pattern as elsewhere. Add `[str, str]` / `[str, Any]` type parameters.
- **1 SPARQL result narrowing** — `SPARQLWrapper.query().convert()` returns a union of `Document | Graph | bytes | str | dict`. Cast at the call site:
  ```python
  from typing import cast, Any
  results = cast(dict[str, Any], sparql.query().convert())
  ```
- **1 OpenAI Fireworks call** — `messages=list[dict[str, str]]` isn't assignable to `Iterable[ChatCompletionMessageParam]`. Same fix pattern as other projects:
  ```python
  from typing import cast
  from openai.types.chat import ChatCompletionMessageParam
  response = client.chat.completions.create(
      model=MODEL_ID,
      messages=cast(list[ChatCompletionMessageParam], messages),
      max_tokens=3500,
  )
  ```
- **1 unsupported `in`** — `library.py:185` uses `if key in lower:` where `lower` is inferred as `Sized`. Add an explicit `str` annotation upstream so pyrefly sees `lower: str`.
- **1 untyped `answer_fn` parameter** — `library.py:211`:
  ```python
  from collections.abc import Callable
  def run_cli(answer_fn: Callable[[str], str], script_name: str) -> None: ...
  ```
- **1 default-arg mismatch** — flagged by pyrefly's `bad-default` rule; fix by matching the annotation to the actual default value.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `4 passed` from pytest.
