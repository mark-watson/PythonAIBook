# Coding Harness Agent

An interactive coding agent that reads your code, runs shell commands, and proposes file edits. You approve each edit before it is written. It works with any OpenAI-compatible LLM provider, including [Fireworks](https://fireworks.ai) hosted models and local models served by [MLX](https://github.com/ml-explore/mlx).

This is example code for a book chapter. It is a teaching implementation, not a production tool.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.10+

## Setup

```
uv sync
```

This creates `.venv/` and installs the one dependency (`requests`).

Set your API key (for Fireworks or any OpenAI-compatible provider):

```
export FIREWORKS_API_KEY=your-key
```

## Run

Interactive REPL:

```
uv run agent.py
```

One-shot prompt:

```
uv run agent.py -p "add type hints to search.py"
```

Pipe a prompt through stdin (with or without `--stdin`):

```
git diff | uv run agent.py -p "review this diff"
git diff | uv run agent.py --stdin -p "review this diff"
```

Or install it as a global tool, which gives you a `coding-agent` command:

```
make install    # uv tool install .
coding-agent
```

## Configuration

The agent reads a hierarchical JSON config:

1. Global: `~/.coding_harness.json`
2. Local: `.local_coding_harness.json` in the working directory. It overrides the global file.

Each provider profile sets the type, endpoint, model, generation parameters, and the name of the environment variable that holds the API key:

```json
{
  "default_provider": "fireworks",
  "providers": {
    "fireworks": {
      "type": "openai",
      "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
      "api_key_env": "FIREWORKS_API_KEY",
      "model": "accounts/fireworks/models/deepseek-v4p1-flash",
      "generation": { "temperature": 0.6, "max_tokens": 32768 },
      "pricing": { "input": 0.14, "cached_input": 0.028, "output": 0.28 }
    },
    "mlx": {
      "type": "mlx",
      "endpoint": "http://localhost:11434/v1/chat/completions",
      "model": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
    }
  }
}
```

Provider type `openai` works with Fireworks and any OpenAI-compatible endpoint. Type `mlx` targets a local server such as `mlx_lm.server` (port 11434), oMLX (port 8000), or sushi (port 12345). The types `ollama`, `omlx`, and `sushi` also work and map to `mlx`. The optional `pricing` block holds per-1M-token USD rates for the `/tokens` command.

Precedence: CLI flags > environment variables > config file > built-in defaults.

## CLI flags

| Flag | Meaning |
| --- | --- |
| `-p, --prompt TEXT` | Prompt text. Repeat or combine with positional words. |
| `--stdin` | Read prompt text from stdin (implied when stdin is a pipe). |
| `positional PROMPT` | Words after the flags form the prompt. |
| `-y, --yes` | Auto-approve edits. |
| `--dry-run` | Show diffs but do not write files. |
| `--provider NAME` | Select a provider profile. |
| `--model ID` | Override the model. |
| `--cwd DIR` | Set the working directory. |
| `--debug` | Log requests and tool calls to stderr. |
| `-q, --quiet` | Less output. |
| `--plain`, `--no-color` | No ANSI colors. |
| `-v, --version` | Print version. |

## Tools the agent can call

- `read_file` — read a file.
- `list_dir` — list a directory.
- `grep` — recursive regex search that skips hidden files and directories.
- `run_shell` — run a whitelisted command (`make`, `ls`, `pwd`, `cat`, `uv`).
- `propose_edit` — whole-file edit: show a unified diff, ask for approval, write, then run `make check`.
- `replace_in_file` — surgical edit: replace one unique snippet, same approval flow and `make check` gate.

The tools refuse hidden and internal paths (any path component that is a dotfile, ends in `~`, or is wrapped in `#...#`) and anything outside the working directory. `grep` and `list_dir` skip those entries rather than reporting them.

Two limits are worth stating plainly. `run_shell`'s whitelist decides *which programs* may run, not *what they do*: `make` and `uv run` execute project code by design. And the set of arguments the model can reach is bounded only by the path checks, so review diffs before approving them.

## Tests

The tool layer, the agentic loop, and the CLI are covered by offline smoke tests: the loop takes a `post_fn` callback, so a fake model is just a list of canned responses.

```
make test        # uv run python -m unittest discover -s tests -v
uv run --extra test pytest -q
```

## Make targets

```
make run          # start the REPL with uv
make sync         # create/update .venv
make lock         # regenerate uv.lock
make check        # byte-compile all sources
make test         # run the smoke tests
make install      # install coding-agent as a uv tool
make build        # build sdist + wheel into dist/
make completions  # generate bash/zsh/fish completions
make clean        # remove dev artifacts
make distclean    # clean plus remove .venv and uv.lock
```

## Files

| File | Role |
| --- | --- |
| `agent.py` | CLI parsing, REPL loop, prompt building. |
| `harness_config.py` | Hierarchical config loading and provider profiles. |
| `chat_loop.py` | Chat completion loop with tool-call dispatch. |
| `fireworks_ai.py` | OpenAI-compatible client for hosted providers. |
| `mlx_serve.py` | Client for a local MLX server. |
| `tools.py` | Tool definitions and execution. |
| `approval.py` | Diffs and interactive approve/reject. |
| `line_input.py` | Readline-backed line editing with fallback. |
| `search.py` | Brave Search and Exa web search helpers. |
| `tests/` | Offline smoke tests (`make test`). |

The chapter in `manuscript/coding-harness-agent.md` prints these modules in full. After editing a module, run `python tools/sync_listings.py --check` (or `--write`) from the repository root so the printed listings keep matching the source.

## License

Copyright (C) 2026 Mark Watson <markw@markwatson.com>
Released under the GNU Affero General Public License v3.0 (AGPL-3.0).
