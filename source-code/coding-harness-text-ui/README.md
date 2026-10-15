# Coding Harness Text UI

A [Textual](https://textual.textualize.io/) terminal UI for the coding agent in
[`../coding-harness-agent`](../coding-harness-agent). This directory holds only
the UI. Providers, the tool-calling loop, the tools, slash commands, skills, and
context compaction all come from the agent library, which is installed as an
editable path dependency.

```
┌ Coding Harness ──────────────────────────────────────────────────────┐
│ ╭ Conversation ─────────────────────────────╮ ╭ Session ───────────╮ │
│ │ > add type hints to search.py             │ │ state     ready    │ │
│ │ [intent: coding → coding tools]           │ │ provider  fireworks│ │
│ │ * read_file {"path": "search.py"}         │ │ model ...          │ │
│ │ --- a/search.py                           │ │ context   ~2100 tok│ │
│ │ +++ b/search.py                           │ │ cost      $0.0012  │ │
│ ╰───────────────────────────────────────────╯ ╰────────────────────╯ │
│ > prompt or /command                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Setup and run

```
uv sync
export FIREWORKS_API_KEY=your-key     # or configure a local MLX provider
uv run harness_tui.py                  # or: make run
uv run harness_tui.py --cwd ~/myproject --provider mlx
```

Configuration is the agent's: `~/.coding_harness.json` plus an optional
`.local_coding_harness.json` in the working directory (see the agent README).

| Flag | Meaning |
| --- | --- |
| `--provider NAME` | Select a provider profile. |
| `--model ID` | Override the model. |
| `--cwd DIR` | Directory the agent works in (config is loaded from there). |
| `-y, --yes` | Auto-approve edits (diffs are still shown). |
| `--dry-run` | Show diffs but never write files. |
| `--debug` | Log raw requests and responses. |

## Using it

- Type a request and press Enter. The turn runs in a background thread; tool
  calls, diffs, and the reply appear in the conversation log.
- When the agent proposes an edit, a dialog shows the diff: `y` applies it, `n`
  rejects it, `s` skips it and asks for a reason that is sent back to the model.
- All agent slash commands work (`/help`, `/context`, `/compact`, `/provider`,
  `/model`, `/tokens`, `/search`, `/skills`, `/<skill>`, `/reset`). The UI adds
  `/clear` (clear the log) and `/quit`.
- Up/Down recalls input history; Right arrow accepts a suggested `/command`.
- `Ctrl+L` clears the log, `Ctrl+Q` quits.

## Files

| File | Role |
| --- | --- |
| `harness_tui.py` | The Textual app: conversation log, status panel, prompt, approval dialog. |
| `agent_bridge.py` | Adapts the agent library's `print()` output and `input()` approval prompts to a UI, without modifying the library. |
| `panes_lib.py` | Unrelated throwaway Textual demo (tabbed panes). |

## How the bridge works

The agent library was written for a terminal REPL, so it talks to the user in
two ways: `print()` for progress and `input()` for edit approval.

- **Output.** The app calls `begin_capture_print()` on the conversation log, so
  everything the agent prints, from any thread, arrives as Textual `Print`
  events. `LineBuffer` reassembles the fragments into lines.
- **Approval.** `tools.py` imports `print_colored_diff`, `prompt_yes_no_skip`,
  and `prompt_reason` by name. `install_approval_hooks()` replaces those names on
  the `tools` module, so the worker thread blocks on a `Future` while the modal
  dialog collects the answer on the UI thread.

A model request cannot be cancelled once it has been sent; quitting with
`Ctrl+Q` exits immediately and abandons it.
