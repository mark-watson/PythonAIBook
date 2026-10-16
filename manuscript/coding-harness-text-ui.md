# A Text User Interface for the Coding Harness Agent

The last chapter built a coding agent that runs in a plain terminal REPL. It prints progress with `print()` and asks for edit approval with `input()`. That works, but a full screen UI makes long sessions easier to follow: the conversation scrolls in one pane, a side panel shows the provider, model, token counts, and cost, and a dialog shows each proposed diff with buttons to apply, reject, or skip it.

This chapter builds that UI with [Textual](https://textual.textualize.io/), a Python library for terminal apps. The new project uses the agent from the last chapter as a library and does not change one line of it. All agent behavior (providers, the tool calling loop, the tools, slash commands, skills, and context compaction) still lives in `source-code/coding-harness-agent`. The new code in `source-code/coding-harness-text-ui` is only the UI, in two files:

| File | Role |
| --- | --- |
| `agent_bridge.py` | Adapts the agent library's `print()` output and `input()` approval prompts to any UI. It does not import Textual. |
| `harness_tui.py` | The Textual app: conversation log, status panel, prompt, and approval dialog. |

Here is the screen layout:

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

## Using the Agent as a Library

The UI project's `pyproject.toml` lists the agent as an editable path dependency:

```toml
[project]
name = "coding-harness-text-ui"
version = "0.1.0"
description = "Textual terminal UI for the coding harness agent in ../coding-harness-agent"
requires-python = ">=3.14"
dependencies = [
    "textual>=8.2.4",
    "coding-harness-agent",
]

[tool.uv]
package = false

[tool.uv.sources]
coding-harness-agent = { path = "../coding-harness-agent", editable = true }
```

Since the install is editable, `uv sync` points the UI's virtual environment at the agent's source directory. A fix in `agent.py` or `tools.py` shows up in the UI the next time you start it, with no reinstall. The `package = false` setting tells uv that the UI project is an app to run, not a package to build.

## Setup and Running

Configuration is the agent's own: `~/.coding_harness.json` plus an optional `.local_coding_harness.json` in the working directory, as covered in the last chapter.

```
cd source-code/coding-harness-text-ui
uv sync
export FIREWORKS_API_KEY=your-key     # or configure a local MLX provider
uv run harness_tui.py                  # or: make run
uv run harness_tui.py --cwd ~/myproject --provider mlx
```

The command line flags match a subset of the agent's flags:

| Flag | Meaning |
| --- | --- |
| `--provider NAME` | Select a provider profile. |
| `--model ID` | Override the model. |
| `--cwd DIR` | Directory the agent works in (config loads from there). |
| `-y, --yes` | Auto-approve edits (diffs still show). |
| `--dry-run` | Show diffs but never write files. |
| `--debug` | Log raw requests and responses. |

Type a request and press Enter. The turn runs in a background thread, and tool calls, diffs, and the reply appear in the conversation log. When the agent proposes an edit, a dialog shows the diff: `y` applies it, `n` rejects it, and `s` skips it and asks for a reason that goes back to the model. All agent slash commands work (`/help`, `/context`, `/compact`, `/provider`, `/model`, `/tokens`, `/search`, `/skills`, `/<skill>`, `/reset`). The UI adds `/clear` to clear the log and `/quit` to exit. Up and Down recall input history, Right arrow accepts a suggested `/command`, `Ctrl+L` clears the log, and `Ctrl+Q` quits.

## The Problem: An Agent Written for a REPL

The agent library talks to the user two ways:

1. **Output.** Every module reports progress with `print()`: tool calls, the intent classifier's choice, errors, and the model's reply.
2. **Approval.** When the model proposes an edit, `gate_write` in `tools.py` shows a diff and asks the user to approve it:

```python
def gate_write(path, full, current, new, heading):
    """Shared approval flow: diff, then y/n/s, then write + `make check`."""
    diff_text = unified_diff(current, new, "a/" + path, "b/" + path)
    print("")
    if heading:
        print(heading)
    print_colored_diff(diff_text)

    if dry_run:
        return "dry-run: diff shown, file not written (use without --dry-run to apply)"
    if auto_approve:
        # Safety: still show diff above, then auto-apply without prompting
        if not quiet_mode:
            print("[auto-approve: applying change without prompt]")
        return apply_change(path, full, new).replace(
            "applied;", "applied (auto-approved);", 1)
    answer = prompt_yes_no_skip()
    if answer == "no":
        return "user rejected the change"
    if answer == "skip":
        reason = prompt_reason()
        return "user skipped: {}".format(reason)
    return apply_change(path, full, new)
```

In the REPL, `print_colored_diff` writes ANSI color codes and `prompt_yes_no_skip` and `prompt_reason` call `input()`. A Textual app owns the terminal, so neither raw ANSI output nor `input()` can work there.

We could add callback hooks to the agent library. Instead, the UI adapts both channels from outside, and the agent stays a plain REPL program that knows nothing about Textual.

**Output** is the easy half. Textual's `App.begin_capture_print(widget)` redirects `sys.stdout` so each `print()` call, from any thread, arrives at the widget as an `events.Print` event. The widget only has to join the fragments into lines.

**Approval** relies on a detail of Python imports. Line 18 of `tools.py` reads:

```python
from approval import unified_diff, print_colored_diff, prompt_yes_no_skip, prompt_reason
```

A `from ... import` binds the three names in the `tools` module's own namespace. When `gate_write` calls `prompt_yes_no_skip()`, Python looks the name up in `tools` at call time. So if we assign a new function to `tools.prompt_yes_no_skip`, every later call from `gate_write` runs our function, while `approval.prompt_yes_no_skip` stays as it was. This is monkey patching, and it works here because we patch the module that *uses* the names, not the module that defines them.

## The Bridge Module

`agent_bridge.py` holds all the code that connects the agent library to a UI. It never imports Textual. The caller passes in two plain functions, `show_diff(diff_text)` and `ask_approval(diff_text)`, so you could reuse this module for a different toolkit, such as a web UI.

The module has five parts:

* `LineBuffer` collects the text that `print()` writes and emits whole lines. One `print("a", "b")` call writes several pieces to stdout ("a", " ", "b", "\n"), so the UI would split lines if it wrote each piece as it arrived. The lock matters because the agent prints from a worker thread while the UI thread may call `flush()`.
* `install_approval_hooks` replaces the three approval names on the `tools` module. The REPL asks for a skip reason with a second prompt, but the UI collects the answer and reason in one dialog. A small `state` dict holds the reason from `prompt_yes_no_skip` until `gate_write` calls `prompt_reason` right after. The function also turns off ANSI color in the `approval` module, since the UI colors diffs itself.
* `start_session` repeats the startup steps of `agent.cli_main`: change to the working directory, load the config, apply config flags and environment overrides, pick the provider and model, set the edit mode, register the tools, and reset the conversation. It changes directory *before* it loads the config so the agent finds `.local_coding_harness.json` in the project it will work on. It returns an error string instead of exiting, so the caller decides how to report failures.
* `run_line` runs one line of input. `agent.handle_slash_command` returns `None` when the line is not a slash command, and only then does the line go to the model with `agent.send_to_model`. The call blocks for the whole turn, including every tool call and approval, so the UI must call it from a worker thread.
* `status` gathers what the side panel shows. It reads token counts from `mlx_serve` for local models (under its stats lock) or from `fireworks_ai` for hosted ones, and estimates context size as total message characters divided by four, the same rule of thumb the agent's `/context` command uses.

Here is the full listing:

```python
# agent_bridge.py -- connect the coding-harness-agent library to a text UI.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# The agent library in ../coding-harness-agent was written for a terminal
# REPL: it reports progress with print() and asks for edit approval with
# input(). This module adapts those two channels without changing the
# library:
#
#   * Output:   the UI captures stdout; LineBuffer turns the fragments that
#               print() writes into whole lines.
#   * Approval: tools.py imported print_colored_diff, prompt_yes_no_skip and
#               prompt_reason by name, so replacing those names on the tools
#               module reroutes every diff and approval prompt to the UI.
#
# Everything here is UI-toolkit agnostic. The caller supplies two callables:
#   show_diff(diff_text)        -- display a unified diff (any thread)
#   ask_approval(diff_text)     -- block until the user answers; returns
#                                  ("yes" | "no" | "skip", reason_str)

import os
import threading

import agent
import approval
import fireworks_ai
import harness_config as hc
import mlx_serve
import tools


class LineBuffer:
    """Accumulate printed text and emit complete lines to `emit`."""

    def __init__(self, emit):
        self._emit = emit
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self._partial += text
            *lines, self._partial = self._partial.split("\n")
        for line in lines:
            self._emit(line)

    def flush(self):
        with self._lock:
            rest, self._partial = self._partial, ""
        if rest:
            self._emit(rest)


def install_approval_hooks(show_diff, ask_approval):
    """Route the edit tools' diff display and y/n/s prompts to the UI."""
    state = {"diff": "", "reason": ""}

    def print_colored_diff(diff_text):
        state["diff"] = diff_text
        show_diff(diff_text)

    def prompt_yes_no_skip():
        answer, reason = ask_approval(state["diff"])
        state["reason"] = reason or ""
        return answer

    def prompt_reason():
        # The UI collects the reason together with the "skip" answer.
        return state["reason"]

    tools.print_colored_diff = print_colored_diff
    tools.prompt_yes_no_skip = prompt_yes_no_skip
    tools.prompt_reason = prompt_reason
    # Diffs are colored by the UI, never with raw ANSI escapes.
    approval.set_color_enabled(False)


def start_session(cwd=None, provider=None, model=None, auto_approve=False,
                  dry_run=False, debug=False):
    """Load config and initialize agent state, mirroring agent.cli_main.

    Returns an error string when the session cannot start, else None.
    """
    if cwd:
        if not os.path.isdir(cwd):
            return "--cwd directory does not exist: {}".format(cwd)
        os.chdir(cwd)
    # Load config after the chdir so .local_coding_harness.json is found in
    # the directory the agent will work in.
    hc.load_harness_config()
    agent.apply_harness_flags()
    agent.apply_env_overrides()
    if not hc.config_provider_names():
        return ("no providers configured. Add a \"providers\" section to "
                "~/.coding_harness.json or .local_coding_harness.json")
    if provider:
        name = provider.strip().lower()
        if not hc.config_provider(name):
            return "unknown provider '{}' (available: {})".format(
                provider, ", ".join(hc.config_provider_names()))
        agent.switch_provider(name)
    if model:
        agent.set_current_model(model.strip())
    tools.auto_approve = auto_approve
    tools.dry_run = dry_run
    if debug:
        fireworks_ai.debug_log = True
    tools.register_all()
    agent.reset_conversation()
    return None


def run_line(line):
    """Run one line of user input: a slash command or a model turn.

    Blocks (call it from a worker thread). Returns "quit" when the user asked
    to leave, else None. Output arrives through print().
    """
    cmd = agent.handle_slash_command(line)
    if cmd == "quit":
        return "quit"
    if cmd is None:
        agent.send_to_model(line)
    return None


def status():
    """-> dict describing the session, for a status panel."""
    if agent.using_mlx():
        with mlx_serve._stats_lock:
            pt = mlx_serve._session_prompt_tokens
            ct = mlx_serve._session_completion_tokens
        cost = 0.0
    else:
        pt, _cached, ct, _total = fireworks_ai.session_snapshot()
        cost = fireworks_ai.session_cost()
    chars = sum(agent.message_char_size(m) for m in agent.messages)
    if tools.dry_run:
        mode = "dry-run"
    elif tools.auto_approve:
        mode = "auto-approve"
    else:
        mode = "ask"
    return {
        "cwd": os.getcwd(),
        "provider": agent.current_provider_name_or_legacy(),
        "model": agent.current_model_id(),
        "edits": mode,
        "search": "{} ({})".format("on" if agent.search_enabled else "off",
                                   agent.search_engine),
        "debug": "on" if fireworks_ai.debug_log else "off",
        "messages": len(agent.messages),
        "context_tokens": chars // 4,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost": cost,
    }


def completion_words():
    """Slash commands and skill names, for input suggestions."""
    return agent.SLASH_COMMANDS + ["/" + s for s in agent.list_skills()]
```

## The Textual App

`harness_tui.py` builds the screen from Textual widgets and runs agent turns in a background thread. Here is the full listing; the sections after it walk through the parts that need explaining.

```python
# harness_tui.py -- Textual terminal UI for the coding harness agent.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
#
# All agent behavior (providers, the tool-calling loop, tools, slash
# commands, skills, compaction) comes from the coding-harness-agent library
# in ../coding-harness-agent. This file is only the UI:
#
#   +-------------------------------------------+----------------+
#   | Conversation log (agent output, diffs)    | Session status |
#   |                                           |                |
#   +-------------------------------------------+----------------+
#   | > prompt or /command                                       |
#   +------------------------------------------------------------+
#
# A model turn runs in a background thread so the UI stays responsive.
# When the agent proposes an edit, a modal shows the diff and asks
# Yes / No / Skip (with a reason), replacing the library's input() prompt.

import argparse
import sys
import threading
from concurrent.futures import Future

from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

import agent_bridge


# ---------------------------------------------------------------------------
# Rendering helpers

def diff_renderable(diff_text):
    """A unified diff as Rich Text: cyan headers, green adds, red removals."""
    out = Text()
    for line in diff_text.rstrip("\n").split("\n"):
        if line.startswith(("+++", "---", "@@")):
            style = "bold cyan"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = ""
        out.append(line + "\n", style=style)
    return out


def output_line_style(line):
    """Pick a style for one line the agent printed."""
    s = line.lstrip()
    if s.startswith("* "):
        return "cyan"                      # tool call
    if s.startswith("[intent:") or s.startswith("[auto-approve"):
        return "dim"
    if s.startswith("[DEBUG]"):
        return "dim yellow"
    if s.startswith(("Error", "[Search error", "[Classifier")):
        return "bold red"
    if s.startswith("(new file:"):
        return "bold"
    return ""


# ---------------------------------------------------------------------------
# Widgets

class HistoryInput(Input):
    """An Input with up/down command history."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.history_index = 0
        self.draft = ""

    def remember(self, line):
        if line and (not self.history or self.history[-1] != line):
            self.history.append(line)
        self.history_index = len(self.history)
        self.draft = ""

    def on_key(self, event: events.Key) -> None:
        if event.key not in ("up", "down"):
            return
        event.prevent_default()
        if event.key == "up":
            if self.history_index == len(self.history):
                self.draft = self.value
            if self.history_index > 0:
                self.history_index -= 1
                self.value = self.history[self.history_index]
        elif self.history_index < len(self.history):
            self.history_index += 1
            self.value = (self.draft if self.history_index == len(self.history)
                          else self.history[self.history_index])
        self.cursor_position = len(self.value)


class ConversationLog(RichLog):
    """The main log. It receives the agent's printed output.

    Printed text, diffs and end-of-turn markers all arrive as messages
    posted to this widget, so they are displayed in the order the agent
    produced them even though the agent runs in another thread.
    """

    class ShowDiff(Message):
        def __init__(self, diff_text):
            super().__init__()
            self.diff_text = diff_text

    class TurnFinished(Message):
        def __init__(self, quit_requested):
            super().__init__()
            self.quit_requested = quit_requested

    def __init__(self, **kwargs):
        super().__init__(wrap=True, markup=False, highlight=False, **kwargs)
        self.buffer = agent_bridge.LineBuffer(self.write_agent_line)

    def write_agent_line(self, line):
        self.write(Text(line, style=output_line_style(line)))

    def on_print(self, event: events.Print) -> None:
        self.buffer.write(event.text)

    def on_conversation_log_show_diff(self, message: ShowDiff) -> None:
        self.buffer.flush()
        self.write(diff_renderable(message.diff_text))

    def on_conversation_log_turn_finished(self, message: TurnFinished) -> None:
        self.buffer.flush()


class ApprovalScreen(ModalScreen):
    """Show a proposed edit and return ("yes" | "no" | "skip", reason)."""

    BINDINGS = [
        Binding("y", "answer('yes')", "Apply"),
        Binding("n", "answer('no')", "Reject"),
        Binding("s", "answer('skip')", "Skip + reason"),
        Binding("escape", "answer('no')", "Reject", show=False),
    ]

    def __init__(self, diff_text):
        super().__init__()
        self.diff_text = diff_text

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("The agent proposes this change. Apply it?",
                        id="approval-title")
            yield RichLog(id="approval-diff", wrap=False, markup=False)
            yield Input(placeholder="Why are you skipping it? (Enter to send)",
                        id="approval-reason")
            with Horizontal(id="approval-buttons"):
                yield Button("Yes, apply (y)", variant="success", id="yes")
                yield Button("No (n)", variant="error", id="no")
                yield Button("Skip + reason (s)", variant="warning", id="skip")

    def on_mount(self) -> None:
        self.query_one("#approval-diff", RichLog).write(
            diff_renderable(self.diff_text))
        self.query_one("#approval-reason", Input).display = False
        self.query_one("#yes", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.action_answer(event.button.id)

    def action_answer(self, answer):
        if answer != "skip":
            self.dismiss((answer, ""))
            return
        reason = self.query_one("#approval-reason", Input)
        reason.display = True
        reason.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(("skip", event.value.strip()))


# ---------------------------------------------------------------------------
# The application

class HarnessApp(App):
    TITLE = "Coding Harness"

    CSS = """
    #main { height: 1fr; }
    #conversation {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #status {
        width: 38;
        border: round $secondary;
        padding: 0 1;
    }
    #prompt { margin: 0 1; }
    ApprovalScreen { align: center middle; }
    #approval-dialog {
        width: 90%;
        height: 85%;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #approval-title { text-style: bold; margin-bottom: 1; }
    #approval-diff { height: 1fr; border: round $primary; }
    #approval-reason { margin-top: 1; }
    #approval-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #approval-buttons Button { margin-left: 2; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("escape", "focus_prompt", "Focus prompt", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.busy = False
        self.closing = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            log = ConversationLog(id="conversation")
            log.border_title = "Conversation"
            yield log
            status = Static(id="status")
            status.border_title = "Session"
            yield status
        yield HistoryInput(
            id="prompt",
            placeholder="Ask the agent, or type /help for commands",
            suggester=SuggestFromList(agent_bridge.completion_words(),
                                      case_sensitive=False))
        yield Footer()

    @property
    def conversation(self) -> ConversationLog:
        return self.query_one("#conversation", ConversationLog)

    def on_mount(self) -> None:
        agent_bridge.install_approval_hooks(self.show_diff, self.ask_approval)
        self.begin_capture_print(self.conversation)
        info = agent_bridge.status()
        self.sub_title = info["cwd"]
        log = self.conversation
        log.write(Text("Coding Agent.  /help for agent commands, /clear to "
                       "clear this log, Ctrl+Q to quit.", style="bold"))
        log.write(Text("Up/Down recalls history; Right arrow accepts a "
                       "suggested /command.", style="dim"))
        self.refresh_status()
        self.set_interval(1.0, self.refresh_status)
        self.query_one("#prompt").focus()

    def on_unmount(self) -> None:
        self.closing.set()

    # --- status panel ---

    def refresh_status(self) -> None:
        info = agent_bridge.status()
        t = Text()

        def row(label, value, style=""):
            t.append("{:<10}".format(label), style="bold")
            t.append(str(value) + "\n", style=style)

        row("state", "working…" if self.busy else "ready",
            "bold yellow" if self.busy else "bold green")
        row("provider", info["provider"])
        t.append("model\n", style="bold")
        t.append("  " + info["model"] + "\n", style="cyan")
        row("edits", info["edits"],
            "" if info["edits"] == "ask" else "bold magenta")
        row("search", info["search"])
        row("debug", info["debug"])
        t.append("\n")
        row("messages", info["messages"])
        row("context", "~{} tok".format(info["context_tokens"]))
        row("prompt", "{} tok".format(info["prompt_tokens"]))
        row("output", "{} tok".format(info["completion_tokens"]))
        cost = info["cost"]
        row("cost", "n/a" if cost is None else "${:.4f}".format(cost))
        t.append("\ncwd\n", style="bold")
        t.append(info["cwd"] + "\n", style="dim")
        self.query_one("#status", Static).update(t)

    # --- input ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, HistoryInput):
            return
        line = event.value.strip()
        prompt = event.input
        prompt.remember(line)
        prompt.value = ""
        if not line:
            return
        if line in ("/quit", "/exit"):
            self.exit()
            return
        if line == "/clear":
            self.action_clear_log()
            return
        if self.busy:
            self.notify("The agent is still working on the previous request.",
                        severity="warning")
            return
        self.conversation.write(Text("\n> " + line, style="bold"))
        self.set_busy(True)
        threading.Thread(target=self.run_turn, args=(line,), daemon=True).start()

    def set_busy(self, busy):
        self.busy = busy
        prompt = self.query_one("#prompt", HistoryInput)
        prompt.placeholder = ("Working… (wait for the reply)" if busy
                              else "Ask the agent, or type /help for commands")
        self.refresh_status()

    def run_turn(self, line):
        """Worker thread: run one line through the agent library.

        A daemon thread rather than a Textual worker, so quitting does not
        have to wait for an in-flight HTTP request to finish."""
        quit_requested = False
        try:
            quit_requested = agent_bridge.run_line(line) == "quit"
        except Exception as e:  # noqa: BLE001
            print("\nError talking to model: {}".format(e))
        if self.closing.is_set():
            return
        self.conversation.post_message(ConversationLog.TurnFinished(quit_requested))
        try:
            self.call_from_thread(self.finish_turn, quit_requested)
        except RuntimeError:
            pass  # app already shut down

    def finish_turn(self, quit_requested):
        self.set_busy(False)
        if quit_requested:
            self.exit()

    # --- approval bridge (called from the worker thread) ---

    def show_diff(self, diff_text):
        self.conversation.post_message(ConversationLog.ShowDiff(diff_text))

    def ask_approval(self, diff_text):
        """Block the worker thread until the user answers the modal."""
        answer = Future()
        try:
            self.call_from_thread(self.push_screen, ApprovalScreen(diff_text),
                                  answer.set_result)
        except RuntimeError:
            return ("no", "")
        while not self.closing.is_set():
            try:
                result = answer.result(timeout=0.25)
            except TimeoutError:
                continue
            verdict, reason = result
            label = {"yes": "approved", "no": "rejected",
                     "skip": "skipped: " + reason}[verdict]
            print("[change {}]".format(label))
            return result
        return ("no", "")

    # --- actions ---

    def action_clear_log(self) -> None:
        self.conversation.clear()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt").focus()


# ---------------------------------------------------------------------------
# CLI entry

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="harness-tui",
        description="Text UI for the coding harness agent.")
    parser.add_argument("--provider", metavar="NAME",
                        help="provider profile from the harness config")
    parser.add_argument("--model", metavar="ID",
                        help="override the provider's model")
    parser.add_argument("--cwd", metavar="DIR",
                        help="directory the agent works in")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="auto-approve edits (diffs are still shown)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show diffs but never write files")
    parser.add_argument("--debug", action="store_true",
                        help="log raw requests and responses")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    error = agent_bridge.start_session(
        cwd=args.cwd, provider=args.provider, model=args.model,
        auto_approve=args.yes, dry_run=args.dry_run, debug=args.debug)
    if error:
        sys.stderr.write("error: {}\n".format(error))
        sys.exit(5)
    HarnessApp().run()


if __name__ == "__main__":
    main()
```

### Rendering

Two helpers style text with Rich, the rendering library under Textual. `diff_renderable` colors a unified diff: cyan for headers, green for added lines, red for removed lines. `output_line_style` picks a style for each line the agent prints, based on the prefixes the agent library uses: `* ` marks a tool call, `[intent:` marks the classifier's choice, and lines that start with `Error` show in bold red. Since the UI matches on text prefixes, a change to the agent's print format changes the colors but never breaks the UI.

### Widgets

`HistoryInput` extends Textual's `Input` with Up and Down history. It saves the half typed line as `draft` when you first press Up, so pressing Down past the newest entry brings your draft back. `event.prevent_default()` stops `Input` from handling the arrow keys itself.

`ConversationLog` extends `RichLog` and receives all captured output in `on_print`. It also defines two custom messages, `ShowDiff` and `TurnFinished`. The reason for these messages is ordering. Captured `print()` output arrives as events in this widget's message queue. If the worker thread wrote a diff to the log by a different route, the diff could appear before or after text the agent printed just before it. Posting the diff as a message to the *same* widget puts it in the same queue as the printed text, so the log shows events in the order the agent produced them. Each handler flushes the `LineBuffer` first, so a partial line printed before the diff appears above it.

`ApprovalScreen` is a `ModalScreen`, which blocks input to the screen below until it closes. Key bindings `y`, `n`, and `s` call `action_answer`. Escape counts as "no", the safe default. Choosing "skip" does not close the dialog; it shows a hidden `Input` for the reason, and pressing Enter there closes the dialog with `("skip", reason)`. `dismiss` passes the result tuple to the callback given to `push_screen`.

### Threads

A model turn can take minutes, so `on_input_submitted` starts a daemon thread running `run_turn`. The app tracks one `busy` flag and refuses a new request while a turn runs. `/quit` and `/clear` never reach the agent; the app handles them on the UI thread.

The docstring on `run_turn` explains why it uses a plain daemon thread instead of a Textual worker: a model request cannot be cancelled once sent, and Textual waits for its workers on exit. A daemon thread lets `Ctrl+Q` exit at once and abandon the request.

When the turn ends, `run_turn` posts `TurnFinished` to the log (to flush the last partial line in order) and then uses `call_from_thread` to run `finish_turn` on the UI thread. Textual widgets are not thread safe, so every UI change from the worker goes through `call_from_thread` or `post_message`. If the app has already shut down, `call_from_thread` raises `RuntimeError`, which the code ignores.

### The Approval Handshake

`ask_approval` is the tricky part of the program. It runs on the *worker* thread, inside `gate_write`, which expects a blocking call that returns an answer. But the dialog must run on the *UI* thread. The handshake goes like this:

1. The worker creates a `concurrent.futures.Future`.
2. It asks the UI thread to push an `ApprovalScreen`, with `answer.set_result` as the dismiss callback.
3. The worker waits on the future in a loop with a quarter second timeout.
4. When the user answers, Textual calls `set_result` on the UI thread, and the worker's wait returns the `(verdict, reason)` tuple.

The timeout loop checks the `closing` event between waits. If the user quits while the dialog is open, `on_unmount` sets `closing`, and the worker returns `("no", "")` instead of blocking forever. The worker also prints `[change approved]`, `[change rejected]`, or `[change skipped: reason]` so the log records each decision.

### The Status Panel

`on_mount` sets a one second timer that calls `refresh_status`, which reads `agent_bridge.status()` and redraws the panel. The panel shows whether a turn is running, the provider and model, the edit mode (`ask`, `auto-approve`, or `dry-run`), search and debug settings, message count, context size, token counts, cost, and working directory. Cost shows `n/a` when the provider profile has no `pricing` section, since `fireworks_ai.session_cost` returns `None` then.

Polling once a second is simpler than having the agent push updates, and reading a few numbers costs nothing.

### Startup

`main` parses the flags and calls `agent_bridge.start_session` *before* it creates the app. Config errors, such as an unknown provider, print to stderr and exit with code 5 while the terminal is still in normal mode, so the user can read them. Only then does `HarnessApp().run()` take over the screen. In `on_mount`, the app installs the approval hooks and starts capturing print output, and from that point every word the agent prints appears in the conversation log.

## Wrap Up

The UI adds about 600 lines and needs no change to the agent. Two Python features make that possible: Textual can capture `print()` from any thread, and a `from ... import` binds names in the importing module, where we can replace them. The `Future` handshake turns a blocking `input()` call on a worker thread into a modal dialog on the UI thread.

Monkey patching has a cost. The UI depends on the names `tools.py` imports, so if a later version of the agent renamed `prompt_yes_no_skip` or called it as `approval.prompt_yes_no_skip`, the patch would silently stop working and the agent would call `input()` inside the Textual app. If you plan to maintain both projects, consider moving the hooks into the agent as explicit callbacks once the UI design settles.

## Exercises for the Reader

1. **Stream the reply.** `fireworks_ai.py` reads the response as an SSE stream but prints nothing until the full reply arrives. Make it print tokens as they arrive, and confirm that the conversation log shows them without any UI change.
2. **Cancel a turn.** Add a `Ctrl+C` binding that sets a flag the agentic loop checks between tool calls, so the user can stop a long turn without quitting the app.
3. **Show a side by side diff.** Change `ApprovalScreen` to show the old and new file in two panes instead of a unified diff.
4. **Save the log.** Add a `/save FILE` command that writes the conversation log to a Markdown file.
5. **Replace the patch with callbacks.** Add `set_approval_handlers(show_diff, ask_yes_no_skip)` to `approval.py`, call it from `install_approval_hooks`, and confirm the REPL agent still works unchanged.
