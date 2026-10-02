# deep_learning_image_generation

Two text-to-image generation demos: one local (Stable Diffusion via HuggingFace `diffusers`) and one cloud (Google Imagen 4, reached through [`litellm`](https://github.com/BerriAI/litellm)'s `image_generation()` entry point rather than an SDK). Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
deep_learning_image_generation/
├── image_generation.py           # Stable Diffusion (segmind/tiny-sd) — runs locally
├── gemini_image_generation.py    # Google Imagen 4 via litellm.image_generation
├── tests/
│   ├── conftest.py               # sys.path shim
│   └── test_smoke.py             # import-only smoke tests (no network / no downloads)
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

After any Python edit, `.claude/hooks/py-check.sh` runs automatically — it formats the file with ruff, applies safe autofixes, then typechecks it with pyrefly. Fix any reported errors before moving on.

When Claude finishes a turn, `.claude/hooks/py-stop.sh` runs the full gate (`ruff format --check`, `ruff check`, `pyrefly check`, `pytest`). If it fails, Claude must fix the errors before the session ends.

Run `just check` manually at any time to verify the whole project.

## Tools

| Command | What it does |
|---------|-------------|
| `just fmt` | Format all Python files |
| `just lint` | Lint and autofix all Python files |
| `just typecheck` | Run pyrefly on the whole project |
| `just test` | Fast test run (testmon — only affected tests) |
| `just test-all` | Full parallel test run |
| `make local` | Run the Stable Diffusion demo (downloads ~1.1 GB on first run) |
| `make gemini` | Run the Imagen demo (needs `GOOGLE_API_KEY`) |

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.14"`.
- The local script is mostly `diffusers` glue; annotation errors there typically come from `diffusers` type stubs. The cloud script is a few lines around `litellm.image_generation`, whose results are `ImageObject` values with optional `b64_json` / `url` fields.
- Config keys are **hyphenated** (`python-version`, not `python_version`).
- Unknown error-kind keys in `[errors]` silently break the config — add them one at a time.

## Testing notes

- Tests import both scripts via the `tests/conftest.py` `sys.path` shim.
- `test_smoke.py` is import-only — it does **not** download model weights or call the Gemini API. Both scripts guard their real work behind `if __name__ == "__main__":`.
- If you want to smoke-test the pipeline itself, mock `DiffusionPipeline.from_pretrained` and `litellm.image_generation` — actually calling either is prohibitively slow / expensive for CI.
