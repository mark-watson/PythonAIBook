# Text Adventure Game

A terminal-based text adventure game powered by Fireworks.ai's `deepseek-v4-flash` model. The LLM acts as the Game Master, dynamically generating the story, describing scenes, and responding to your choices.

## Setup

1. Set your Fireworks API key:

   ```bash
   export FIREWORKS_API_KEY="your-api-key"
   ```

2. Install dependencies and run:

   ```bash
   uv sync
   uv run python game.py
   ```

## How to Play

The game presents scenes and choices. Type your action or choice at the prompt and the AI Game Master advances the story.

### Commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/restart` | Start a new adventure |
| `/quit` | Exit the game |

## Customizing the Story

Edit `story.txt` to change the setting, rules, or tone of the game. The contents of this file are sent as the system prompt, defining the Game Master's behavior and the world.

## Development workflow

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner. Install both, then:

```bash
uv sync
just check       # fmt-check + lint + typecheck + test
just fmt         # ruff format
just lint        # ruff --fix
just typecheck   # pyrefly (strict)
just test        # pytest with testmon (fast)
just test-all    # full parallel pytest run
```

Under Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the workflow contract.
