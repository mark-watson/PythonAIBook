# line_input.py -- rlwrap-style line editing for the coding-agent REPL.
# Python port of line-input.rkt.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Uses Python's bundled `readline` module (GNU Readline on Linux, libedit on
# macOS) to give the REPL cursor editing, in-session history (arrows,
# Ctrl-R), and Tab completion.
#
# Two things can go wrong, and both degrade to a plain input() instead of
# failing the harness:
#
#   * `import readline` raises when no readline backend is present. This
#     module therefore loads it inside a handler, and every entry point works
#     without it.
#
#   * stdin may not be a terminal (pipes, --stdin, CI). Editing is skipped
#     and the prompt is printed by hand so piped output is byte-identical to
#     plain read-line behavior.

import os
import sys

readline = None
_backend_tried = False

MAX_HISTORY = 1000


def _load_backend():
    global readline, _backend_tried
    if not _backend_tried:
        _backend_tried = True
        try:
            import readline as _rl  # noqa: F401
            readline = _rl
        except ImportError:
            readline = None
    return readline is not None


def terminal_stdin():
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def line_input_available():
    """-> bool. Loads the backend on first use, and only engages editing when
    stdin is a terminal, so piped/--stdin runs never touch readline at all."""
    return terminal_stdin() and _load_backend()


# ---------------------------------------------------------------------------
# History file

history_file = None


def default_history_file():
    return os.path.expanduser("~/.coding_agent_history")


def set_history_file(path):
    """Point at `path`, loading any existing entries. None disables persistence."""
    global history_file
    history_file = path
    if readline and path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.strip() != "":
                        readline.add_history(line)
        except Exception:
            pass


def save_history():
    if not readline or not history_file:
        return
    try:
        n = readline.get_current_history_length()
        count = min(n, MAX_HISTORY)
        if count > 0:
            lines = []
            for i in range(n - count + 1, n + 1):
                item = readline.get_history_item(i)
                if item is not None:
                    lines.append(item)
            tmp = history_file + "~tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            os.replace(tmp, history_file)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Completion

# Readline asks for one candidate at a time via completer(text, state), so the
# filtered list is built once (at state 0) and cached here between calls.
_completions = []


def install_completer(candidates_fn):
    """candidates_fn: str -> list of completion strings, for the word under the
    cursor. Ignored when the backend is unavailable."""
    if not line_input_available():
        return

    def completer(text, state):
        # Pure string work: no I/O, threads, or subprocesses (libedit calls
        # this while the interpreter is effectively in atomic mode).
        if state == 0:
            _completions.clear()
            try:
                matches = candidates_fn(text) or []
            except Exception:
                matches = []
            # Readline replaces the word it passed us, so a candidate that
            # does not begin with that word would be inserted verbatim and
            # leave stray text on the line. Drop those instead of guessing.
            for m in matches:
                if isinstance(m, str) and m and m.startswith(text) and m not in _completions:
                    _completions.append(m)
        if state < len(_completions):
            return _completions[state]
        return None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n\"'")
    # macOS ships libedit behind the readline module; the bind syntax differs.
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


# ---------------------------------------------------------------------------
# Reading

def read_input_line(prompt):
    """-> str, or None on EOF.

    Ctrl-C cancels the current line and returns "" (the caller re-prompts);
    Ctrl-D returns None, which ends the REPL. Always prints the prompt, with
    readline handling editing on a terminal and a plain read-line used
    otherwise."""
    if line_input_available():
        try:
            line = input(prompt)
            # The backend records the line as it is accepted, so re-adding it
            # would put every command in the history twice.
            n = readline.get_current_history_length()
            if n == 0 or readline.get_history_item(n) != line:
                readline.add_history(line)
            return line
        except EOFError:
            return None
        except KeyboardInterrupt:
            print("")  # move past the ^C echo; the caller re-prompts
            return ""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        print("")
        return ""
    except Exception:
        return None
    if line == "":  # EOF
        return None
    return line.rstrip("\n")