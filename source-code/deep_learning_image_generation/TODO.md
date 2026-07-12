# TODO — deep_learning_image_generation

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (2 tests).

All 6 pyrefly errors are SDK-stub-driven: `diffusers.DiffusionPipeline.from_pretrained` and `google-genai`'s response types are declared as `Optional[...]` in their stubs, and this code assumes the happy path. Two low-friction ways to resolve them:

## Option A — `assert` narrowing (recommended)

Insert `assert x is not None` right after each call whose return type is optional. This gives pyrefly enough info to narrow and produces a real error at runtime if the SDK ever returns `None`.

## Option B — precise annotations

Cast the pipe / response into the concrete concrete subtype from the SDK (e.g. `cast(StableDiffusionPipeline, DiffusionPipeline.from_pretrained(...))`), which sidesteps the optional-return type.

## Pyrefly (6 errors)

### `image_generation.py:27, 31, 41` — `pipe` treated as `Optional[DiffusionPipeline]`
Because `DiffusionPipeline.from_pretrained(...)` is stubbed to return `Optional[...]`.

Fix per `Option A`:
```python
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
assert pipe is not None
pipe = pipe.to("cuda")
```

And similarly for the MPS and CPU branches. Add one more `assert pipe is not None` right before `image = pipe(...)`.

Line 41 also produces `not-callable — Expected a callable, got DiffusionPipeline`. `DiffusionPipeline` doesn't advertise `__call__` at the base-class level in the stubs — only concrete pipelines like `StableDiffusionPipeline` do. Fix by narrowing on `hasattr(pipe, "__call__")` or by casting:

```python
from typing import cast
from diffusers import StableDiffusionPipeline
pipe = cast(StableDiffusionPipeline, pipe)
image = pipe(prompt, num_inference_steps=25).images[0]
```

### `gemini_image_generation.py:42–43` — `response.generated_images` is `Optional[list[...]]`
And each item's `.image` is also `Optional`.

Fix per `Option A`:
```python
assert response.generated_images is not None
for generated_image in response.generated_images:
    assert generated_image.image is not None
    image = Image.open(io.BytesIO(generated_image.image.image_bytes))
    ...
```

## Nice-to-have follow-ups

- The Stable Diffusion script's device-selection block repeats `DiffusionPipeline.from_pretrained` three times. Extract a helper that picks the device + dtype and calls `from_pretrained` once.
- Neither `main()` returns anything useful. Consider returning the output path so the smoke tests could assert against it after mocking the pipeline / API.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `2 passed` from pytest.
