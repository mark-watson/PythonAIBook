# TODO — llm_local_models

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (7 script-parse tests).

All 7 pyrefly errors cluster in two files and stem from `ollama.chat(...).message.content` being typed as `str | None`.

## Pyrefly (7 errors)

### `ollama_memory.py:32` — `LocalAssistant.chat` may return `None`
`ollama.chat(...).message.content` is `str | None`; `chat` is declared `-> str`. Two easy fixes:

**Option A** — assert non-null:
```python
reply = response.message.content
assert reply is not None
self.messages.append({"role": "assistant", "content": reply})
return reply
```

**Option B** — coerce with a fallback:
```python
reply = response.message.content or ""
```

### `ollama_reasoning.py:17` — return type `dict` needs parameters
```python
def reason_about(question: str, model: str = "deepseek-r1:7b") -> dict[str, str]:
```

### `ollama_reasoning.py:27–29` — `content` is `str | None`
Four cascading errors on the same root cause. Same fix pattern as `ollama_memory.py`:

```python
content = response.message.content or ""
if "<think>" in content and "</think>" in content:
    reasoning = content.split("<think>")[1].split("</think>")[0].strip()
    answer = content.split("</think>")[1].strip()
```

## Nice-to-have follow-ups

- None of the seven scripts wrap their work in `if __name__ == "__main__":` — so `test_smoke.py` uses `ast.parse` on each file rather than a real import. If you refactor any of them into a `main()` function with a guard, promote its entry in `test_smoke.py::SCRIPTS` to an import test for stronger coverage.
- `image_to_text_description.py` references `ticket.png` at module top-level via `image_path = 'ticket.png'` — currently only used inside the top-level `ollama.chat(...)` call. If the file goes missing the smoke test still passes (no import happens), but a live run would fail with a confusing error before the model responds. Consider making the path a CLI arg with a default.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `7 passed` from pytest.
