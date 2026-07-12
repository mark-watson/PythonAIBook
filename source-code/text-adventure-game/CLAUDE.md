# text-adventure-game

A tiny text-adventure driver that streams responses from GPT to keep the story going. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Needs `OPENAI_API_KEY` for the actual game.

## Layout

```
text-adventure-game/
├── game.py            # main game loop (guarded)
├── story.txt          # opening scene / narrator prompt
├── tests/
│   ├── conftest.py
│   └── test_smoke.py  # import-only
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

`.claude/hooks/py-check.sh` runs after every edit. `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. `just check` runs the same gate manually.

## Testing notes

- `game.py` is `if __name__ == "__main__":`-guarded, so importing it is safe.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.14"`.
