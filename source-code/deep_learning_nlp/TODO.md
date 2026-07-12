# TODO — deep_learning_nlp

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (3 import-only smoke tests, ~17s including torch startup).

Only 2 pyrefly errors, both in the same file: `AutoTokenizer.from_pretrained()` is stubbed to return `Optional[PreTrainedTokenizerBase]` in the `transformers` package, so pyrefly can't call it or invoke `.decode()`.

## Pyrefly (2 errors)

### `summarization.py:43, 59` — `tokenizer` treated as possibly `None`
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert tokenizer is not None
```

right after the `from_pretrained` call. `assert` is enough for pyrefly to narrow the type, and it also gives a clear runtime error if the load ever fails silently.

Alternatively, cast to the concrete tokenizer class:

```python
from typing import cast
from transformers import BartTokenizerFast   # or whatever bart-large-cnn uses

tokenizer = cast(BartTokenizerFast, AutoTokenizer.from_pretrained(model_name))
```

## Nice-to-have follow-ups

- `sentence_similarity.py`, `summarization.py`, and `zero_shot_classification.py` all wrap their work in `main()` guarded by `__name__`, so the smoke tests are meaningful — they exercise the imports (including the heavy `sentence_transformers` / `transformers` startup). Test run currently takes ~17s cold. If that becomes annoying, consider skipping imports of the model-heavy scripts under a `pytest.mark.slow` gate.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `3 passed` from pytest.
