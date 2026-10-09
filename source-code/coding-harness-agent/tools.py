# tools.py -- tool registry and coding-agent tools
# Python port of tools.rkt (originally py-coding-agent/tools.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Six tools: read_file, list_dir, grep, run_shell, propose_edit,
# replace_in_file. The two edit tools show a colored diff and ask y/n/s;
# propose_edit then gates on `make check`.

import json
import os
import re
import shlex
import subprocess

from approval import unified_diff, print_colored_diff, prompt_yes_no_skip, prompt_reason

# ---------------------------------------------------------------------------
# Registry

registry = {}

SHELL_WHITELIST = {"make", "ls", "pwd", "cat", "uv"}
MAX_CHECK_OUTPUT_CHARS = 2000
MAX_SHELL_OUTPUT_CHARS = 20000
GREP_MAX_MATCHES = 200
GREP_MAX_LINE_CHARS = 300
SHELL_TIMEOUT = 300
CHECK_TIMEOUT = 600
TIMEOUT_EXIT_CODE = 124

# CLI-controlled modes (mutated by agent.py, read by the tools)
auto_approve = False
dry_run = False
quiet_mode = False

# Set by tool_propose_edit / tool_replace_in_file when an applied change leaves
# `make check` failing. agent.py reads it to pick the process exit code for a
# one-shot run instead of re-scanning old tool output.
make_check_failed = False


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
        return "Error: unknown tool '{}'. Available tools: {}".format(
            name, ", ".join(sorted(registry)))
    params = tool["parameters"]
    # Missing required args? Return an actionable error describing the expected
    # argument list -- small models frequently emit malformed/truncated
    # arguments, and silently receiving None tends to send them into retry loops.
    # Match the Racket semantics: only a missing key or a JSON null counts as
    # missing -- the empty string is a VALID value (propose_edit passes ""
    # and replace_in_file may pass an empty replacement).
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
        bad_json = None
        try:
            parsed = json.loads(args_json)
            args_parsed = parsed if isinstance(parsed, dict) else "NOT-OBJECT"
        except Exception as e:  # noqa: BLE001
            # The decode error itself ("Expecting ',' delimiter: line 3 ...")
            # tells the model what to fix, so pass it through.
            bad_json = str(e)
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
        elif bad_json is not None:
            result = "Error: invalid JSON in arguments for tool '{}': {}. Received: {}".format(
                name, bad_json, short)
        elif args_parsed == "NOT-OBJECT":
            result = "Error: arguments for tool '{}' must be a JSON object. Received: {}".format(name, short)
        else:
            result = call_tool(name, args_parsed)
        # Guarantee a usable id: some local servers omit tool_call ids, and the
        # matching role:"tool" message needs one to line up.
        if not isinstance(call_id, str) or call_id == "":
            call_id = "call_{}".format(len(results))
        results.append((call_id, name, result))
    return results


# ---------------------------------------------------------------------------
# Helpers: run subprocess and capture combined output

def run_external(exe, args, timeout=SHELL_TIMEOUT):
    """exe: str, args: list of str -> (combined_output, exit_code).

    A timeout is reported as exit code 124 rather than raised, so a hung
    command becomes tool feedback instead of an exception.
    """
    try:
        proc = subprocess.run([exe] + list(args),
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("command timed out after {}s: {} {}".format(
            timeout, exe, " ".join(str(a) for a in args)), TIMEOUT_EXIT_CODE)
    return (proc.stdout + proc.stderr), proc.returncode


def truncate_string(s, max_len):
    if len(s) > max_len:
        return s[:max_len] + "\n... (truncated, {} total chars)".format(len(s))
    return s


def strip_shell_quotes(s):
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


# Hidden files (ignored from listings and rejected on read/search/run):
#   - names ending in ~  (e.g. foo.rkt~)
#   - names wrapped in #...#  (e.g. #foo.rkt#)
#   - names starting with .  (e.g. .git, .gitignore, .env)
def hidden_file(name):
    return (name.endswith("~")
            or (name.startswith("#") and name.endswith("#"))
            or name.startswith("."))


def hidden_in_path(path):
    """True when any component of `path` names a hidden/internal file.

    Checking every component (not just the basename) is what keeps the tools
    out of directories such as .git/ or ~/.ssh/. "." and ".." are ordinary
    path syntax, not hidden names, so they are skipped here.
    """
    for part in re.split(r"[\\/]+", str(path)):
        if part in ("", ".", ".."):
            continue
        if hidden_file(part):
            return True
    return False


def hidden_arg(s):
    cleaned = strip_shell_quotes(s)
    return (not cleaned.startswith("-")) and hidden_in_path(cleaned)


def working_dir():
    """The directory the agent is allowed to touch (established by --cwd)."""
    return os.path.realpath(os.getcwd())


def path_within_working_dir(path):
    """True when `path` resolves to the working directory or something inside it."""
    root = working_dir()
    full = os.path.realpath(os.path.join(root, os.path.expanduser(path)))
    return full == root or full.startswith(root + os.sep)


def resolve_tool_path(path):
    """-> (absolute path, None) or (None, refusal message).

    Every tool that touches the filesystem funnels through this: hidden
    components are refused, and anything outside the working directory (../,
    absolute paths, symlinks that escape) is refused as well.
    """
    if not isinstance(path, str) or path.strip() == "":
        return None, "refusing: empty path"
    if hidden_in_path(path):
        return None, "refusing to touch hidden/internal path: {}".format(path)
    full = os.path.realpath(os.path.join(working_dir(), os.path.expanduser(path)))
    if full != working_dir() and not full.startswith(working_dir() + os.sep):
        return None, ("refusing: {} is outside the working directory {}. "
                      "Use a path relative to it.").format(path, working_dir())
    return full, None


def looks_like_long_ls_line(line):
    """True for `ls -l` rows: permissions, links, owner, group, size, date, name."""
    parts = line.split(None, 1)
    if not parts:
        return False
    mode = parts[0]
    return len(mode) >= 10 and mode[0] in "-dlbcps" and set(mode[1:10]) <= set("rwxSsTt-")


def filter_ls_output(out):
    """Drop `total N` headers, the . / .. rows, and hidden names.

    Long-format rows are split into at most eight fields so that a filename
    containing spaces survives intact; short-format rows are split into one
    name per whitespace-separated token.
    """
    result_lines = []
    for line in out.split("\n"):
        t = line.strip()
        if t == "" or t.startswith("total "):
            continue
        if looks_like_long_ls_line(t):
            fields = t.split(None, 8)
            fname = fields[8] if len(fields) > 8 else fields[-1]
            if fname in (".", "..") or hidden_file(fname):
                continue
        else:
            names = [n for n in t.split()
                     if n not in (".", "..") and not hidden_file(n)]
            if not names:
                continue
            line = " ".join(names)
        result_lines.append(line)
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Tool implementations

def tool_read_file(path):
    full, refusal = resolve_tool_path(path)
    if refusal:
        return refusal
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:  # noqa: BLE001
        return "Error reading {}: {}".format(path, e)


def tool_list_dir(path):
    full, refusal = resolve_tool_path(path)
    if refusal:
        return refusal
    try:
        entries = sorted(os.listdir(full))
        lines = []
        for e in entries:
            if hidden_file(e):
                continue
            child = os.path.join(full, e)
            lines.append(e + "/" if os.path.isdir(child) else e)
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return "Error listing {}: {}".format(path, e)


def tool_grep(pattern, path):
    """Recursive regex search that skips hidden files and directories.

    Implemented with os.walk + re instead of shelling out to `grep -rnE` for
    three reasons: hidden directories (.git/) stay out of the model's context,
    a pattern beginning with "-" cannot be mistaken for a flag, and the result
    can be capped and truncated instead of flooding the window.
    """
    root, refusal = resolve_tool_path(path)
    if refusal:
        return refusal
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return "Error: invalid regular expression '{}': {}".format(pattern, e)

    targets = [root] if os.path.isfile(root) else None
    if targets is None and not os.path.isdir(root):
        return "Error: no such file or directory: {}".format(path)

    matches = []
    truncated = False

    def add_match(text):
        nonlocal truncated
        if len(matches) >= GREP_MAX_MATCHES:
            truncated = True
            return
        matches.append(truncate_string(text, GREP_MAX_LINE_CHARS))

    if targets:
        files = targets
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not hidden_file(d))
            for fname in sorted(filenames):
                if not hidden_file(fname):
                    files.append(os.path.join(dirpath, fname))

    for fname in files:
        try:
            with open(fname, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if rx.search(line):
                        rel = os.path.relpath(fname, working_dir())
                        add_match("{}:{}:{}".format(rel, lineno, line.rstrip("\n")))
                        if truncated:
                            break
        except (OSError, UnicodeError):
            continue
        if truncated:
            break
    if not matches:
        return "no matches for {!r} under {}".format(pattern, os.path.relpath(root, working_dir()))
    if truncated:
        matches.append("... (stopped after {} matches; narrow the pattern or path)".format(
            GREP_MAX_MATCHES))
    return "\n".join(matches)


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
    for a in tokens[1:]:
        if hidden_arg(a):
            return "refusing to run command referencing hidden/internal file: {}".format(a)
        # Note: the whitelist limits WHICH programs run. It does not sandbox
        # what they do -- `uv run` and `make` execute arbitrary code by design.
        try:
            inside = path_within_working_dir(strip_shell_quotes(a))
        except (OSError, ValueError):
            inside = False
        if not inside:
            return ("refusing to run command referencing a path outside the working "
                    "directory: {}".format(a))
    try:
        out, code = run_external(cmd, tokens[1:], timeout=SHELL_TIMEOUT)
        filtered = filter_ls_output(out) if cmd == "ls" else out
        return "{}(exit {})".format(truncate_string(filtered, MAX_SHELL_OUTPUT_CHARS), code)
    except Exception as e:  # noqa: BLE001
        return "Error running command: {}".format(e)


def run_make_check():
    try:
        return run_external("make", ["check"], timeout=CHECK_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return "make check error: {}".format(e), 1


def apply_change(path, full, new):
    """Write the approved contents and report the `make check` verdict."""
    global make_check_failed
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(new)
    out, status = run_make_check()
    if status == 0:
        return "applied; make check passed"
    make_check_failed = True
    return "applied; make check FAILED (exit {}):\n{}".format(
        status, truncate_string(out, MAX_CHECK_OUTPUT_CHARS))


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


def tool_propose_edit(path, old, new):
    """Whole-file edit: `old` must match the current contents exactly ("" to create)."""
    full, refusal = resolve_tool_path(path)
    if refusal:
        return refusal
    exists = os.path.isfile(full)
    if exists:
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
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

    heading = "(new file: {})".format(path) if not exists else None
    return gate_write(path, full, current, new, heading)


def tool_replace_in_file(path, old_string, new_string):
    """Compact edit: replace one unique snippet instead of echoing the whole file."""
    full, refusal = resolve_tool_path(path)
    if refusal:
        return refusal
    if old_string == "":
        return "refused: old_string is empty; use propose_edit to create a file"
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            current = f.read()
    except Exception as e:  # noqa: BLE001
        return "Error reading {}: {}".format(path, e)

    hits = current.count(old_string)
    if hits == 0:
        return ("old_string not found in {}. Read the file again and copy a unique "
                "snippet exactly, including indentation.".format(path))
    if hits > 1:
        return ("old_string appears {} times in {}. Include more surrounding "
                "context so the snippet is unique.".format(hits, path))
    new = current.replace(old_string, new_string, 1)
    if new == current:
        return "no changes (replacement is identical to the matched text)"
    return gate_write(path, full, current, new, None)


# ---------------------------------------------------------------------------
# Registration

def register_all():
    define_tool(
        "read_file",
        [("path", "string", "File path relative to the working directory.")],
        "Read and return the contents of a file. Refuses hidden/internal paths (~, #...#, dotfiles) and paths outside the working directory.",
        tool_read_file)
    define_tool(
        "list_dir",
        [("path", "string", 'Directory path. Use "." for the working directory.')],
        "List files and subdirectories (with trailing /) in a directory. Hidden/internal files (~, #...#, and dotfiles) are excluded.",
        tool_list_dir)
    define_tool(
        "grep",
        [("pattern", "string", "Python regular expression to search for."),
         ("path", "string", "File or directory path to search, relative to the working directory.")],
        "Recursively search files for PATTERN and return matching lines as path:line:text. Hidden/internal files and directories (dotfiles, .git) are skipped, and the result is capped at {} matches.".format(GREP_MAX_MATCHES),
        tool_grep)
    define_tool(
        "run_shell",
        [("command", "string", "Shell command. Only whitelisted commands may run: make, ls, pwd, cat, uv.")],
        "Run a whitelisted shell command and return its combined output. Refuses hidden/internal paths and paths outside the working directory. The whitelist limits which programs run, not what they do: make and uv execute project code.",
        tool_run_shell)
    define_tool(
        "propose_edit",
        [("path", "string", "Path to the file to edit or create."),
         ("old", "string", "For an existing file: the exact current contents. For a new file: pass empty string."),
         ("new", "string", "The proposed new contents of the file, in full.")],
        "Propose a whole-file edit or new-file creation. The user is shown a unified diff and asked to approve. On approval the file is written and `make check` is run. Prefer replace_in_file for small changes to an existing file.",
        tool_propose_edit)
    define_tool(
        "replace_in_file",
        [("path", "string", "Path to an existing file inside the working directory."),
         ("old_string", "string", "The exact snippet to replace. It must appear exactly once in the file."),
         ("new_string", "string", "The replacement text. Pass empty string to delete the snippet.")],
        "Propose a surgical edit: replace one unique snippet of an existing file. Far cheaper than propose_edit when the file is large. The user is shown a unified diff and asked to approve; on approval the file is written and `make check` is run.",
        tool_replace_in_file)


ENABLED_TOOLS = ["read_file", "list_dir", "grep", "run_shell", "propose_edit",
                 "replace_in_file"]
