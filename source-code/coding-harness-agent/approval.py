# approval.py -- colored diffs and y/n/s approval prompts
# Python port of approval.rkt (mirrors py-coding-agent/approval.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details

import difflib
import sys

# ANSI colours

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"

# When False, print diffs without ANSI (for --plain / --no-color / piped output)
color_enabled = True


def set_color_enabled(enabled):
    """Turn ANSI colors on/off (used by --plain and the plain/quiet config)."""
    global color_enabled
    color_enabled = bool(enabled)


# ---------------------------------------------------------------------------
# unified_diff : str str str str -> str
# difflib-based replacement for running `diff -u` on two temporary files.

def unified_diff(old_content, new_content, old_label, new_label):
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    # difflib omits the trailing newline on the last line; add one so the
    # rendered text matches `diff -u` output line for line.
    diff = difflib.unified_diff(old_lines, new_lines,
                                fromfile=old_label, tofile=new_label)
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


# ---------------------------------------------------------------------------
# print_colored_diff : str -> None

def print_colored_diff(diff_text):
    for line in diff_text.split("\n"):
        if color_enabled and (line.startswith("+++") or line.startswith("---")
                              or line.startswith("@@")):
            print(ANSI_CYAN + line + ANSI_RESET)
        elif color_enabled and line.startswith("+"):
            print(ANSI_GREEN + line + ANSI_RESET)
        elif color_enabled and line.startswith("-"):
            print(ANSI_RED + line + ANSI_RESET)
        else:
            print(line)


# ---------------------------------------------------------------------------
# prompt_yes_no_skip : -> 'yes' | 'no' | 'skip'

def prompt_yes_no_skip():
    """-> 'yes' | 'no' | 'skip'. EOF (no human present) counts as 'no'."""
    while True:
        sys.stdout.write("\nApply this change? [y]es / [n]o / [s]kip and tell the model why: ")
        sys.stdout.flush()
        try:
            line = input("")
        except EOFError:
            print("\n(no answer available: refusing the change)")
            return "no"
        norm = (line or "").strip().lower()
        if norm in ("y", "yes"):
            return "yes"
        if norm in ("n", "no"):
            return "no"
        if norm in ("s", "skip"):
            return "skip"
        print("Please answer y, n, or s.")


# ---------------------------------------------------------------------------
# prompt_reason : -> str

def prompt_reason():
    sys.stdout.write("Reason (one line): ")
    sys.stdout.flush()
    try:
        return input("")
    except EOFError:
        return ""
