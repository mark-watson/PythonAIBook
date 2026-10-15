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
