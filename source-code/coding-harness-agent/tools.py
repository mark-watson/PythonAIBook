# tools.py -- tool registry and coding-agent tools
# Python port of tools.rkt (originally py-coding-agent/tools.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Five tools: read_file, list_dir, grep, run_shell, propose_edit
# propose_edit shows a colored diff, asks y/n/s, and gates on `make check`.

import json
import os
import re
import shlex
import subprocess

import approval
from approval import unified_diff, print_colored_diff, prompt_yes_no_skip, prompt_reason

# ---------------------------------------------------------------------------
# Registry

registry = {}

SHELL_WHITELIST = {"make", "ls", "pwd", "cat", "uv"}
MAX_CHECK_OUTPUT_CHARS = 2000

# CLI-controlled modes (mutated by agent.py, read by the tools)
auto_approve = False
dry_run = False
quiet_mode = False


def define_tool(name, params, description, handler):
    """params: list of (pname, ptype, pdesc) tuples."""
    registry[name] = {
        "name": name,
        "description": description,
        "parameters": params,
        "handler": handler,
    }


def render_tools(names):
    rendered = []
    for name in names:
        tool = registry.get(name)
        if tool is None:
            raise ValueError("Undefined tool: {}".format(name))
        props = {}
        required = []
        for (pname, ptype, pdesc) in tool["parameters"]:
            props[pname] = {"type": ptype, "description": pdesc}
            required.append(pname)
        rendered.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return rendered


# ---------------------------------------------------------------------------
# Tool dispatch

def call_tool(name, args):
    tool = registry.get(name)
    if tool is None:
        raise ValueError("Unknown tool: {}".format(name))
    params = tool["parameters"]
    # Missing required args? Return an actionable error describing the expected
    # argument list -- small models frequently emit malformed/truncated
    # arguments, and silently receiving None tends to send them into retry loops.
    # Match the Racket semantics: only a missing key or a JSON null counts as
    # missing -- the empty string is a VALID value (propose_edit passes ""
    # as `old` when creating a new file).
    missing = [p[0] for p in params if args.get(p[0]) is None]
    if missing:
        return ("Error: tool '{}' missing required argument(s): {}. "
                "Expected arguments (JSON object): {}".format(
                    name, ", ".join(missing),
                    ", ".join(p[0] for p in params)))
    positional = [args.get(p[0]) for p in params]
    try:
        result = tool["handler"](*positional)
        return str(result) if result is not None else ""
    except Exception as e:  # noqa: BLE001
        return "Error: tool '{}' raised: {}  (check argument types/values)".format(name, e)


def execute_tool_calls(tool_calls):
    """tool-calls: list of dicts with 'id', 'function' {name, arguments}.

    Returns list of (call_id, name, result_str) tuples.
    """
    results = []
    for call in tool_calls:
        call_id = call.get("id", "")
        func = call.get("function") or {}
        name = func.get("name", "")
        name = name if isinstance(name, str) else str(name)
        args_json = func.get("arguments", "{}")
        if args_json is None or not isinstance(args_json, str):
            args_json = "{}" if args_json is None else json.dumps(args_json)
        short = args_json if len(args_json) <= 120 else args_json[:117] + "..."
        if not quiet_mode:
            print("* {} {}".format(name, short))
        try:
            parsed = json.loads(args_json)
            args_parsed = parsed if isinstance(parsed, dict) else "NOT-OBJECT"
        except Exception:
            args_parsed = "BAD-JSON"
        if isinstance(args_parsed, dict):
            # Coerce non-string values to strings so handlers behave like the
            # Racket version's format-based rendering.
            args_parsed = {k: (v if isinstance(v, str) else json.dumps(v))
                           for k, v in args_parsed.items()}
        if str(name).strip() == "":
            # Truncated tool call -- the model stopped mid-generation, so no
            # function name survived. Feed that back instead of crashing.
            result = ("Error: the model's tool call was truncated mid-generation "
                      "(no function name provided). Received arguments: {}".format(short))
        elif args_parsed == "BAD-JSON":
            result = "Error: invalid JSON in arguments for tool '{}'. Received: {}".format(name, short)
        elif args_parsed == "NOT-OBJECT":
            result = "Error: arguments for tool '{}' must be a JSON object. Received: {}".format(name, short)
        else:
            # Unknown tool names, contract violations, etc. become feedback to the
            # model rather than an uncaught exception that aborts the loop.
            try:
                result = call_tool(name, args_parsed)
            except Exception as e:  # noqa: BLE001
                result = "Error: tool '{}' raised: {}".format(name, e)
        results.append((call_id, name, result))
    return results


# ---------------------------------------------------------------------------
# Helpers: run subprocess and capture combined output

def run_external(exe, args):
    """exe: str, args: list of str -> (combined_output, exit_code)."""
    proc = subprocess.run([exe] + list(args),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace")
    return (proc.stdout + proc.stderr), proc.returncode


def truncate_string(s, max_len):
    if len(s) > max_len:
        return s[:max_len] + "\n... (truncated, {} total chars)".format(len(s))
    return s


def strip_shell_quotes(s):
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


# Hidden files (ignored from listings and reject read attempts):
#   - names ending in ~  (e.g. foo.rkt~)
#   - names wrapped in #...#  (e.g. #foo.rkt#)
#   - names starting with .  (e.g. .git, .gitignore, .env)
def hidden_file(name):
    return (name.endswith("~")
            or (name.startswith("#") and name.endswith("#"))
            or name.startswith("."))


def hidden_arg(s):
    cleaned = strip_shell_quotes(s)
    return (not cleaned.startswith("-")) and hidden_file(os.path.basename(cleaned))


def filter_ls_output(out):
    lines = out.split("\n")
    result_lines = []
    for line in lines:
        t = line.strip()
        if t.startswith("total ") or t == "":
            continue
        tokens = t.split()
        if not tokens:
            continue
        if re.match(r"^[-d]", tokens[0]):
            fname = tokens[-1]
            if hidden_file(fname) or fname in (".", ".."):
                continue
        elif hidden_file(t):
            continue
        result_lines.append(line)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Tool implementations

def tool_read_file(path):
    try:
        fname = os.path.basename(path)
        if hidden_file(fname):
            return "refusing to read hidden/internal file: {}".format(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:  # noqa: BLE001
        return "Error reading {}: {}".format(path, e)


def tool_list_dir(path):
    try:
        entries = sorted(os.listdir(path))
        lines = []
        for e in entries:
            if hidden_file(e):
                continue
            full = os.path.join(path, e)
            lines.append(e + "/" if os.path.isdir(full) else e)
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return "Error listing {}: {}".format(path, e)


def tool_grep(pattern, path):
    try:
        out, _code = run_external("grep", ["-rnE", pattern, path])
        return out
    except Exception as e:  # noqa: BLE001
        return "Error running grep: {}".format(e)


def tool_run_shell(command):
    command = command.strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return "empty command"
    cmd = tokens[0]
    if cmd not in SHELL_WHITELIST:
        return "Command '{}' not whitelisted. Allowed: {}".format(
            cmd, ", ".join(sorted(SHELL_WHITELIST)))
    if cmd != "ls":
        for a in tokens[1:]:
            if hidden_arg(a):
                return "refusing to run command referencing hidden/internal file: {}".format(a)
    try:
        out, code = run_external(cmd, tokens[1:])
        filtered = filter_ls_output(out) if cmd == "ls" else out
        return "{}(exit {})".format(filtered, code)
    except Exception as e:  # noqa: BLE001
        return "Error running command: {}".format(e)


def run_make_check():
    try:
        return run_external("make", ["check"])
    except Exception as e:  # noqa: BLE001
        return "make check error: {}".format(e), 1


def tool_propose_edit(path, old, new):
    exists = os.path.isfile(path)
    if exists:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                current = f.read()
        except Exception as e:  # noqa: BLE001
            return "Error reading {}: {}".format(path, e)
    else:
        current = ""

    if exists and current != old:
        return ("stale base: on-disk contents of {} do not match the 'old' you "
                "provided. Read the file again and retry.".format(path))
    if exists and current == new:
        return "no changes (proposed content matches current file)"
    if not exists and new == "":
        return "refused: cannot create an empty file"

    diff_text = unified_diff(current, new, "a/" + path, "b/" + path)
    print("")
    if not exists:
        print("(new file: {})".format(path))
    print_colored_diff(diff_text)

    def _apply():
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
        out, status = run_make_check()
        if status == 0:
            return "applied; make check passed"
        return "applied; make check FAILED (exit {}):\n{}".format(
            status, truncate_string(out, MAX_CHECK_OUTPUT_CHARS))

    if dry_run:
        return "dry-run: diff shown, file not written (use without --dry-run to apply)"
    if auto_approve:
        # Safety: still show diff above, then auto-apply without prompting
        if not quiet_mode:
            print("[auto-approve: applying change without prompt]")
        result = _apply()
        return result.replace("applied;", "applied (auto-approved);", 1)
    answer = prompt_yes_no_skip()
    if answer == "no":
        return "user rejected the change"
    if answer == "skip":
        reason = prompt_reason()
        return "user skipped: {}".format(reason)
    return _apply()


# ---------------------------------------------------------------------------
# Registration

def register_all():
    define_tool(
        "read_file",
        [("path", "string", "File path relative to the working directory.")],
        "Read and return the contents of a file. Refuses to read hidden/internal files (~, #...#, and dotfiles).",
        tool_read_file)
    define_tool(
        "list_dir",
        [("path", "string", 'Directory path. Use "." for the working directory.')],
        "List files and subdirectories (with trailing /) in a directory. Hidden/internal files (~, #...#, and dotfiles) are excluded.",
        tool_list_dir)
    define_tool(
        "grep",
        [("pattern", "string", "Extended regex pattern to search for."),
         ("path", "string", "Directory or file path to search.")],
        "Recursively grep files for PATTERN. Wraps `grep -rnE`.",
        tool_grep)
    define_tool(
        "run_shell",
        [("command", "string", "Shell command. Only whitelisted commands may run: make, ls, pwd, cat, uv.")],
        "Run a whitelisted shell command and return its combined output. Refuses commands that reference hidden/internal files.",
        tool_run_shell)
    define_tool(
        "propose_edit",
        [("path", "string", "Path to the file to edit or create."),
         ("old", "string", "For an existing file: the exact current contents. For a new file: pass empty string."),
         ("new", "string", "The proposed new contents of the file, in full.")],
        "Propose an edit or new-file creation. The user is shown a unified diff and asked to approve. On approval the file is written and `make check` is run.",
        tool_propose_edit)


ENABLED_TOOLS = ["read_file", "list_dir", "grep", "run_shell", "propose_edit"]
