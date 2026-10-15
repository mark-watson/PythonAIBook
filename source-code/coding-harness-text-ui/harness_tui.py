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
