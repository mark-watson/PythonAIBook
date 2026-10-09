# Building a Coding Harness Agent

Large language models can talk about code. They cannot read your files, run your tests, or edit your programs. The bridge between those two worlds is a piece of software called a *harness*. This chapter builds a complete, small harness from scratch: a terminal coding agent that reads a project, proposes edits as diffs, asks for your approval, runs `make check` after every change, and works with both hosted APIs and local models.

Along the way we will cover the four problems every harness must solve:

1. **Tool calling.** How do you describe actions to a model so that it emits machine-executable instructions instead of prose?
2. **The agentic loop.** How do you run model, tools, and feedback until the task is done, without letting a confused model loop forever?
3. **Control and safety.** How do you gate file writes behind human approval, restrict shell access, and detect a stuck model?
4. **Configuration.** How do you keep every provider-specific value (endpoint, model, API key name, pricing) in data files so that adding a provider never requires a code change?

The finished program is a Python project managed by [uv](https://docs.astral.sh/uv/). All source files are listed in full, and each listing is introduced and followed by discussion.

## The Program at a Glance

The agent is nine modules. Each has one job:

| File | Role |
| --- | --- |
| `agent.py` | CLI parsing, REPL, slash commands, intent routing, context management |
| `harness_config.py` | Hierarchical JSON configuration: global file, local override file, provider profiles |
| `chat_loop.py` | The provider-agnostic agentic loop (the heart of the program) |
| `tools.py` | Tool registry: `read_file`, `list_dir`, `grep`, `run_shell`, `propose_edit` |
| `approval.py` | Unified diffs, ANSI coloring, y/n/s approval prompts |
| `fireworks_ai.py` | Client for OpenAI-compatible hosted APIs, with SSE streaming and cost tracking |
| `mlx_serve.py` | Client for local OpenAI-compatible servers (MLX, oMLX, sushi, Ollama) |
| `search.py` | Brave and Exa web search backends |
| `line_input.py` | Readline editing, persistent history, Tab completion, with graceful fallback |

Data flows through the program like this:

```
user input --> agent.py --> intent classifier --> chat_loop.py (agentic loop)
                                                    |
                                fireworks_ai.py or mlx_serve.py  (HTTP transport)
                                                    |
                                       provider endpoint (hosted or local)
                                                    |
                                       tool_calls? --> tools.py executes
                                                    |
                                       approval.py gates writes
                                                    |
                                       results appended as tool messages, loop repeats
```

Two design decisions shape everything else. First, the loop in `chat_loop.py` knows nothing about providers: each client module passes it a `post_fn` callback that performs the actual HTTP work. Second, no provider-specific value is compiled into the code: endpoints, models, generation parameters, and pricing all live in JSON configuration files.

## The Agentic Loop, in Theory

An OpenAI-style chat request sends the model a list of messages plus a list of tool definitions. The model replies in one of two shapes. It can answer with plain text, which ends the loop. Or it can reply with `tool_calls`, a list of function names plus JSON arguments:

```json
{
  "role": "assistant",
  "content": "Let me read the file first.",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"search.py\"}"
      }
    }
  ]
}
```

The harness must then execute the call, append the result as a new message of role `tool`, and send the whole (longer) conversation back to the model:

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "name": "read_file",
  "content": "def normalize(s):\n    return s.strip().lower()\n"
}
```

The model sees its own action and its outcome, and decides the next step. This read-act-observe cycle, repeated until the model stops calling tools, is the agentic loop. It is the same pattern behind Claude Code, Pi, and the other coding agents you may have used.

A naive implementation of this loop fails in practice for a predictable reason: weak models get stuck. They re-issue the identical failing call dozens of times, or they stop generating mid-call. A serious harness detects these states and bails out with a message. We will see exactly how in `chat_loop.py`.

## Configuration: The Global and Local Files

Before any code runs, you need to tell the agent which models to talk to. The harness uses a two-layer configuration system, in the style of the Pi coding harness:

- **Global file:** `~/.coding_harness.json`. Your machine-wide defaults: every provider profile you use, your lifestyle flags (quiet, plain, debug), and search settings.
- **Local override file:** `.local_coding_harness.json`. A per-project file, loaded from the current working directory at startup. It changes only what this project needs: a different default model, a different temperature, search enabled.

The two files are deep-merged, with the local file winning. A project can therefore pin its own provider and generation settings without touching your global preferences.

### The Global File

Create `~/.coding_harness.json`. This is a realistic setup with one hosted provider and one local provider:

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
      "model": "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
      "generation": { "temperature": 0.6, "max_tokens": 32768 }
    }
  },
  "search": { "engine": "brave", "enabled": false },
  "debug": false,
  "quiet": false,
  "plain": false
}
```

Each provider is a *profile*, a named bag of fields:

- `type` selects the wire protocol. `"openai"` means OpenAI-compatible chat completions with a Bearer key (Fireworks and any compatible endpoint). `"mlx"` means a local OpenAI-compatible server that needs no key. The strings `"ollama"`, `"omlx"`, and `"sushi"` are also accepted and map to `"mlx"`, because those servers all expose the same route on different ports: `mlx_lm.server` on 11434, oMLX on 8000, sushi on 12345.
- `endpoint` is the full chat-completions URL.
- `model` is the model identifier the server expects.
- `api_key_env` names the *environment variable* that holds the key. The config file never contains secrets, only the name of the variable to read.
- `generation` holds `temperature`, `max_tokens`, and any other parameter to pass through. Parameters absent here are left out of the request entirely, so the server's own default applies.
- `pricing` holds USD rates per 1,000,000 tokens, used by the `/tokens` command. A profile without `pricing` reports token counts and prints cost as `n/a`. Nothing is guessed.

The top-level `search` block picks the engine (`"brave"` or `"exa"`) and whether search starts enabled. The three booleans `debug`, `quiet`, and `plain` are lifestyle flags applied at startup.

### The Local Override File

Suppose a project works best with the local model, and you want web search on for it. Put `.local_coding_harness.json` in the project root and override only those keys:

```json
{
  "default_provider": "mlx",
  "search": { "enabled": true },
  "providers": {
    "fireworks": {
      "model": "accounts/fireworks/models/kimi-k2-instruct"
    }
  }
}
```

After merging with the global file above, the project runs with:

- active provider `mlx` (the local `default_provider` wins),
- search on, using the global engine `brave` (the nested merge kept it),
- the `mlx` profile exactly as the global file defines it,
- the `fireworks` profile with the global endpoint, key name, generation, and pricing, but the *local* model.

That last point is the payoff of recursive merging: you can change one field inside a profile and inherit everything else.

### Merge Rules and Precedence

Three layers of precedence exist, applied in this order:

1. **CLI flags** (`--provider`, `--model`, `--quiet`, ...) are strongest. `agent.py` applies them last.
2. **Environment variables** `CODING_AGENT_PROVIDER`, `CODING_AGENT_MODEL`, `CODING_AGENT_QUIET`, `CODING_AGENT_PLAIN`, `CODING_AGENT_DEBUG` override config values. An env var naming an unknown provider produces a warning, not a crash.
3. **Config files**: local over global, key by key.

Anything the layers do not set falls back to profile defaults, and a request parameter that the profile does not declare is omitted from the wire payload. The only hard requirement is that at least one provider profile exists: the CLI refuses to start with an error message rather than silently falling back to a compiled-in default endpoint.

The module that implements all of this is short and worth reading in full:

```python
# coding_harness -- hierarchical JSON configuration for the coding harness,
# in the rough style of the Pi coding harness config.  Python port of
# harness-config.rkt.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Two config layers are merged (local wins over global on conflicts):
#
#   Global:  ~/.coding_harness.json            (base configuration)
#   Local:   .local_coding_harness.json        (optional per-project override,
#                                              loaded from the current
#                                              directory at startup)
#
# Rough format (all sections optional):
#
# {
#   "default_provider": "mlx",
#   "providers": {
#     "mlx":       { "type": "mlx", "endpoint": "http://localhost:11434/v1/chat/completions",
#                    "model": "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
#                    "generation": { "temperature": 0.6, "max_tokens": 32768 } },
#     "fireworks": { "type": "openai", "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
#                    "api_key_env": "FIREWORKS_API_KEY",
#                    "model": "accounts/fireworks/models/deepseek-v4p1-flash",
#                    "generation": { "temperature": 0.6, "max_tokens": 32768 },
#                    "pricing": { "input": 0.14, "cached_input": 0.028, "output": 0.28 } }
#   },
#   "search": { "engine": "brave", "enabled": false },
#   "debug": false, "quiet": false, "plain": false
# }
#
# Provider "type" is either "mlx" (the local mlx-serve backend -- OpenAI-style
# /v1/chat/completions served by mlx_lm.server on localhost:11434, oMLX on
# port 8000, sushi on port 12345) or "openai" (OpenAI-style chat completions;
# Fireworks.ai and any compatible endpoint). The type strings "ollama",
# "omlx", and "sushi" are also accepted and mapped to "mlx".
#
# Every provider-specific value lives here: endpoint, model, api_key_env,
# generation parameters, and the per-1M-token USD "pricing" rates (input,
# cached_input, output) that /tokens uses. Nothing provider-specific is
# compiled into the Python code, so adding or changing a provider needs no
# code change. A profile with no "pricing" block reports token counts without
# a cost estimate rather than guessing a rate.
#
# Merge rules: nested dicts merge recursively, local keys override global
# keys; anything that is not a dict (strings, numbers, booleans, lists) is
# replaced wholesale by the local value when present.

import json
import os

GLOBAL_CONFIG_PATH = os.path.expanduser("~/.coding_harness.json")
LOCAL_CONFIG_PATH = os.path.join(os.getcwd(), ".local_coding_harness.json")

# The merged config, loaded once at startup (reloadable via load_harness_config).
harness_config = {}

# Mutable cell: the name of the provider section currently in use.
_active_provider_name = None


# ---------------------------------------------------------------------------
# JSON loading helpers

def _read_json_file(path):
    """-> config dict, or None if the file is missing / unreadable / not an object."""
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Recursive dict merge (local overrides global)

def deep_merge(global_cfg, local_cfg):
    if not isinstance(global_cfg, dict):
        return local_cfg
    if not isinstance(local_cfg, dict):
        return local_cfg
    acc = dict(global_cfg)
    for k, v in local_cfg.items():
        if isinstance(acc.get(k), dict) and isinstance(v, dict):
            acc[k] = deep_merge(acc[k], v)
        else:
            acc[k] = v
    return acc


def load_harness_config():
    """Load global then local, deep-merge, store, and return the result."""
    global harness_config, _active_provider_name
    global_cfg = _read_json_file(GLOBAL_CONFIG_PATH) or {}
    # The local override is always looked up in the *current* directory.
    local_path = os.path.join(os.getcwd(), ".local_coding_harness.json")
    local_cfg = _read_json_file(local_path) or {}
    harness_config = deep_merge(global_cfg, local_cfg)
    _active_provider_name = None  # re-resolve the default profile
    return harness_config


# ---------------------------------------------------------------------------
# Providers

def config_providers():
    p = harness_config.get("providers", {})
    return p if isinstance(p, dict) else {}


def config_provider_names():
    return sorted(config_providers().keys())


def config_provider(name):
    """-> provider dict for profile `name`, or None."""
    if name is None:
        return None
    return config_providers().get(str(name))


def _pick_default_provider_name(cfg):
    declared = cfg.get("default_provider")
    names = sorted(config_providers().keys())
    if isinstance(declared, str) and declared in config_providers():
        return declared
    if "fireworks" in names:
        return "fireworks"
    if names:
        return names[0]
    return None


def config_active_provider_name():
    global _active_provider_name
    if _active_provider_name is None:
        _active_provider_name = _pick_default_provider_name(harness_config)
    return _active_provider_name


def config_set_active_provider(name):
    """Switch the active profile. Returns the resulting active name."""
    global _active_provider_name
    if name is not None and config_provider(name):
        _active_provider_name = str(name)
    return _active_provider_name


def config_active_provider():
    """-> provider dict of the active profile, or None when there is no config."""
    n = config_active_provider_name()
    return config_provider(n) if n else None


# ---------------------------------------------------------------------------
# Provider field accessors (all tolerant of missing keys)

def provider_type(provider):
    """-> 'mlx' | 'openai' -- defaults to 'openai'.

    "mlx" selects the local mlx-serve backend (formerly "ollama"); "ollama",
    "omlx", and "sushi" are also accepted and mapped to 'mlx for compatibility.
    """
    t = provider.get("type") if provider else None
    if isinstance(t, str):
        low = t.strip().lower()
        if low in ("mlx", "ollama", "omlx", "sushi"):
            return "mlx"
    return "openai"


def provider_endpoint(provider):
    e = provider.get("endpoint") if provider else None
    return e if isinstance(e, str) and e != "" else None


def provider_model(provider):
    m = provider.get("model") if provider else None
    return m if isinstance(m, str) and m != "" else None


def provider_api_key_env(provider):
    """Name of the env var holding the Bearer key for this endpoint, or None.

    Absent/empty/null means "no key" (plain local MLX).
    """
    k = provider.get("api_key_env") if provider else None
    return k if isinstance(k, str) and k != "" else None


def provider_generation(provider):
    g = provider.get("generation") if provider else None
    return g if isinstance(g, dict) else {}


def provider_pricing(provider):
    """-> dict of per-1M-token USD rates ('input', 'cached_input', 'output'),
    or an empty dict when the profile declares none."""
    g = provider.get("pricing") if provider else None
    return g if isinstance(g, dict) else {}


def pricing_ref(pricing, key):
    """-> number, or None when the profile does not declare that rate.

    The None result means "unknown", which callers report rather than
    guessing a value.
    """
    if not isinstance(pricing, dict):
        return None
    if key in pricing:
        return pricing[key]
    return None


def generation_ref(generation, key, default=None):
    """Fetch a generation parameter ("temperature", "max_tokens", "think", ...)."""
    if not isinstance(generation, dict):
        return default
    return generation.get(key, default)


# ---------------------------------------------------------------------------
# Debug helper

def print_config_summary():
    print("Config files: {} {} / {} {}".format(
        GLOBAL_CONFIG_PATH,
        "(loaded)" if os.path.isfile(GLOBAL_CONFIG_PATH) else "(absent)",
        os.path.join(os.getcwd(), ".local_coding_harness.json"),
        "(loaded)" if os.path.isfile(os.path.join(os.getcwd(), ".local_coding_harness.json")) else "(absent)"))
    print("Providers:    {}".format(", ".join(config_provider_names())))
    print("Active:       {}".format(config_active_provider_name() or "(defaults)"))
```

Three details are worth pointing out.

`_read_json_file` returns `None` for a missing file, unreadable file, *or* a file whose top-level JSON is not an object. Bad configuration degrades to "absent", never to an exception at import time. This is deliberate: `harness_config` is imported by every other module, so a crash here would make even `--help` fail.

`deep_merge` walks the two dictionaries in parallel. When both sides have a dict for the same key, it recurses. Otherwise the local value replaces the global value wholesale. That is why a local `"search": { "enabled": true }` keeps the global `"engine": "brave"`: the two search dicts merge. But a list is replaced, not appended, because there is no sane general rule for merging lists.

`config_active_provider_name` caches the chosen profile in a module-level variable, and `load_harness_config` resets that cache. The selection rule when `default_provider` is missing or wrong: prefer a profile named `fireworks`, else the alphabetically first profile.

## Tool Calling: The Registry in tools.py

The model can only call tools that the harness declared in the request. `tools.py` keeps a registry mapping tool names to a description, a parameter list, and a Python handler. `render_tools` converts each entry into the OpenAI JSON-schema form. For `read_file` the rendered definition sent over the wire looks like this:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read and return the contents of a file. Refuses to read hidden/internal files (~, #...#, and dotfiles).",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "File path relative to the working directory."
        }
      },
      "required": ["path"]
    }
  }
}
```

Here is the whole module:

```python
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
```

The five tools divide into two groups. The *observation* tools (`read_file`, `list_dir`, `grep`) are safe and unrestricted apart from a hidden-file filter. The *action* tools (`run_shell`, `propose_edit`) can change the world, so both are gated: `run_shell` accepts only `make`, `ls`, `pwd`, `cat`, and `uv`, and `propose_edit` requires your approval.

Four ideas in this file carry most of the teaching value.

**Errors are feedback, not exceptions.** Look at `call_tool` and `execute_tool_calls`. Missing arguments, malformed JSON, truncated tool calls, and unknown tool names all return an error *string* that goes back to the model in a `tool` message. Small models produce these failures constantly. Returning actionable text ("missing required argument(s): path; Expected arguments: path") lets the model correct itself. Raising an exception would abort the whole session.

**The empty string is a valid argument.** When the model creates a new file it passes `""` as `old`. So `call_tool` treats only a *missing key or JSON null* as a missing argument. A careless `if not value` check would reject the one argument value that matters most.

**Hidden files are invisible.** The predicate `hidden_file` treats names ending in `~`, names wrapped in `#...#`, and dotfiles as hidden. Listings skip them, reads refuse them, and shell arguments referencing them are rejected. Secrets in `.env` and repository internals in `.git` stay out of the model's context.

**Edits are compared against a fresh read of the file.** In `tool_propose_edit`, if the model's `old` string does not match the current on-disk contents, the tool returns a `stale base` error telling the model to read the file again and retry. This kills an entire class of hallucinated-edit bugs, where a model edits from memory of a file it read twenty messages ago.

When the user approves, `_apply` writes the file and runs `make check`. A failed check returns its output (truncated to 2000 characters) as the tool result, so the model sees the compile or test errors and can react. The harness also turns that failure into a nonzero exit code in one-shot mode, which lets shell scripts detect a bad edit.

## Human Approval: approval.py

Nothing reaches your disk without passing through the approval prompt. `approval.py` computes a unified diff with `difflib`, colors it with ANSI codes, and asks `y`, `n`, or `s`. The `s` answer is the interesting one: it skips the change *and* collects a one-line reason, which becomes part of the tool result:

```
user skipped: do not touch the public API, add a wrapper instead
```

The model reads that sentence and changes its plan. Free-text feedback is cheaper for the model to use than a bare rejection.

```python
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
    while True:
        sys.stdout.write("\nApply this change? [y]es / [n]o / [s]kip and tell the model why: ")
        sys.stdout.flush()
        try:
            line = input("")
        except EOFError:
            line = ""
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
```

The coloring logic is intentionally trivial: cyan for headers and `@@` hunk markers, green for `+` lines, red for `-` lines. The module-level `color_enabled` flag is what `--plain` and `--no-color` turn off.

One small fidelity detail: `unified_diff` appends a trailing newline when `difflib` omits one, so the rendered text matches what the command-line `diff -u` produces line for line.

## The Provider-Agnostic Loop: chat_loop.py

This module is the heart of the harness and is fully independent of any provider. Its interface is the `post_fn` callback: given a chat-completions payload, return the parsed response as a dictionary with `choices[0].message` and `usage`. Both clients provide that shape, so both share one loop:

```python
# chat_loop.py -- provider-agnostic agentic tool-calling loop, shared by
# fireworks_ai.py and mlx_serve.py. Python port of chat-loop.rkt.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# The `post_fn` argument adapts an OpenAI-style chat-completions payload to a
# specific backend and returns a normalized response dict:
#   { 'choices': [ { 'message': <assistant msg>, 'finish_reason': ... } ],
#     'usage':   { 'prompt_tokens': n, 'completion_tokens': n, ... } }
# where <assistant msg> is { 'role': "assistant", 'content': <string> }
# plus, when the model called tools, 'tool_calls' -- a list of
#   { 'id': <string>, 'type': "function",
#     'function': { 'name': <string>, 'arguments': <json string> } }

from tools import render_tools, execute_tool_calls

# Repetition detection: track the last few tool-call signatures so a model
# stuck issuing the identical failing call is stopped early rather than
# burning all max-iterations. A signature is (name + args-json) per call,
# sorted, so multi-call batches compare as a set.
REPEAT_WINDOW = 5   # remember the last N batches
REPEAT_LIMIT = 2    # >= 2 identical batches in the window => stuck


# ---------------------------------------------------------------------------
# Helpers

def response_message(data):
    """Extract the assistant message dict from a normalized response, or None."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice.get("message")
    return None


def msg_content(msg):
    """The assistant message's 'content', coerced to a string.

    Some OpenAI-compatible servers (e.g. mlx_lm.server) emit "content": null
    on tool-only responses; coerce any non-string value to "" here.
    """
    c = msg.get("content", "") if msg else ""
    return c if isinstance(c, str) else ""


# ---------------------------------------------------------------------------
# Request payload

def request_payload(model_id, max_tokens, temperature, messages):
    """Build a request body, omitting generation parameters the active
    provider profile did not declare. No max_tokens/temperature default is
    compiled in here; the provider config is the only source for them."""
    payload = {"model": model_id, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


# ---------------------------------------------------------------------------
# chat* : post_fn messages ... -> str

def chat(post_fn, messages, model_id, max_tokens=None, temperature=None):
    payload = request_payload(model_id, max_tokens, temperature, messages)
    data = post_fn(payload)
    msg = response_message(data)
    content = msg_content(msg)
    return content if content else "No response content"


# ---------------------------------------------------------------------------
# chat_with_tools : -> (final_text, final_messages)
# Multi-turn agentic loop.

def chat_with_tools(post_fn, messages, tools, model_id,
                    max_tokens=None, temperature=None, max_iterations=20):
    tools_rendered = render_tools(tools)
    current_messages = list(messages)
    recent_signatures = []

    def call_signature(tool_calls):
        sig = []
        for tc in tool_calls:
            f = tc.get("function") or {}
            sig.append("{}|{}".format(f.get("name", ""), f.get("arguments", "")))
        return sorted(sig)

    def seen_too_often(sig):
        # True when the same batch of calls appeared >= REPEAT_LIMIT times in
        # the recent window (i.e. once already, before this repeat).
        return sum(1 for s in recent_signatures if s == sig) >= REPEAT_LIMIT - 1

    def append_tool_results(results):
        for (call_id, name, result_str) in results:
            current_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result_str,
            })

    _it = 0
    while True:
        iter_ = _it
        _it += 1
        if iter_ >= max_iterations:
            # Max iterations -- one final no-tools call for summary
            payload = request_payload(model_id, max_tokens, temperature, current_messages)
            try:
                data = post_fn(payload)
                msg = response_message(data)
                content = msg.get("content", "") if msg else ""
                if not isinstance(content, str) or content == "":
                    content = "(no summary from model)"
                if msg:
                    current_messages.append(msg)
                return content, current_messages
            except Exception:
                return "(max tool iterations reached)", current_messages

        payload = request_payload(model_id, max_tokens, temperature, current_messages)
        if tools_rendered:
            payload["tools"] = tools_rendered
            payload["tool_choice"] = "auto"
        data = post_fn(payload)
        msg = response_message(data)
        if msg is None:
            raise RuntimeError("response has no 'message'. Raw: {}".format(data))
        tool_calls = msg.get("tool_calls")
        content = msg_content(msg)
        # Append the assistant message
        current_messages.append(msg)

        if (isinstance(tool_calls, list) and tool_calls
                and seen_too_often(call_signature(tool_calls))):
            # The model is stuck re-issuing the identical call(s) -- bail out
            # with an explanation instead of looping to max-iterations.
            return ("(stopped: the model repeated the identical tool call(s) "
                    "{} times without making progress; it may be too weak for "
                    "this task or its arguments are malformed)".format(REPEAT_LIMIT),
                    current_messages)

        if content and content.strip() and isinstance(tool_calls, list) and tool_calls:
            print("")
            print(content.strip())
            recent_signatures.append(call_signature(tool_calls))
            recent_signatures = recent_signatures[-REPEAT_WINDOW:]
            append_tool_results(execute_tool_calls(tool_calls))
            continue
        if not tool_calls:
            return (content if content else "(empty response from model)"), current_messages
        recent_signatures.append(call_signature(tool_calls))
        recent_signatures = recent_signatures[-REPEAT_WINDOW:]
        append_tool_results(execute_tool_calls(tool_calls))
```

Follow `chat_with_tools` iteration by iteration:

1. Render the tool definitions and attach them to the payload with `"tool_choice": "auto"`.
2. POST. Extract the assistant message.
3. Append the assistant message to the history, exactly as the model sent it. The history grows to be a faithful transcript: user text, assistant text, tool calls, tool results.
4. If there are no `tool_calls`, the turn is over. Return the text and the new history.
5. If there are tool calls, print any accompanying text, execute the calls, append one `tool` message per call, and go back to step 1.

Two safeguards stand out.

**Repetition detection.** Each batch of tool calls is reduced to a *signature*: a sorted list of `name|arguments-json` strings. Sorting makes multi-call batches compare as a set, so the order the model emits them does not hide a repeat. The loop remembers the last `REPEAT_WINDOW = 5` signatures. When the same signature appears `REPEAT_LIMIT = 2` times in that window, the loop stops and says so. A model stuck on a malformed argument no longer burns twenty iterations of your token budget.

**Bounded iterations with a graceful ending.** After `max_iterations = 20`, the loop makes one final request *without* tools, forcing the model to produce a text summary instead of another tool call. If even that fails, it returns a placeholder string. The loop always returns a `(text, messages)` pair; the session never hangs on the loop itself.

The code also patches over real-world server bugs. `msg_content` coerces `"content": null` to `""`, because `mlx_lm.server` sends exactly that on tool-only responses. The payload builder omits `max_tokens` and `temperature` unless they were configured, because some local servers reject unknown or unwanted parameters.

## A Hosted Client with Streaming and Cost Accounting: fireworks_ai.py

The name is historical; the module talks to *any* OpenAI-compatible endpoint named by the active profile. It adds two things the shared loop does not do: it reads the config for endpoint, key variable, model, and generation parameters, and it consumes SSE streaming responses.

A streaming response arrives as a series of `data:` lines, each holding a small JSON chunk. Content arrives in fragments, and so do tool calls, split *across chunks by index*:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"grep"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pat"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tern\": \"foo\"}"}}]}}]}
data: [DONE]
```

`parse_sse_response` reassembles these fragments into one ordinary, non-streaming-shaped response dict, so the rest of the program never knows streaming happened. It accumulates `content`, `reasoning_content` (emitted by DeepSeek-style models), and per-index tool-call fragments, and keeps the last `usage` object, which the server sends at the end of the stream when `stream_options: {"include_usage": true}` is set.

```python
# fireworks_ai.py -- OpenAI-compatible (Fireworks AI) streaming API client,
# session stats, chat helpers. Python port of fireworks-ai.rkt
# (originally py-coding-agent/fireworks_ai.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Endpoint, model, api_key_env, generation parameters, and pricing all come
# from the active provider profile in the harness config (~/.coding_harness.json
# and/or .local_coding_harness.json); no provider-specific value is compiled
# in here.

import json
import threading

import requests

import harness_config as hc
from harness_config import (config_active_provider, provider_api_key_env,
                            provider_endpoint, provider_generation,
                            provider_model, provider_pricing, pricing_ref,
                            generation_ref)
from chat_loop import chat as chat_star
from chat_loop import chat_with_tools as chat_with_tools_star

# ---------------------------------------------------------------------------
# Modes and timeouts

debug_log = False  # shared /debug toggle (agent.py flips this)

# Requests use SSE streaming ("stream": true), so there is NO total
# wall-clock cap on generation: a long response that keeps producing
# tokens simply keeps streaming. The only remaining timeouts are:
#   CONNECT_MAX_TIME    -- seconds to establish the TCP connection.
#   HEADER_MAX_TIME     -- seconds to wait for response headers (TTFT).
#   STREAM_IDLE_TIMEOUT -- seconds of *silence* from the server before we
#                          give up (requests' read timeout applies per-chunk).
#                          Tokens arriving periodically never trip this; only
#                          a genuinely stalled connection does.
CONNECT_MAX_TIME = 10
HEADER_MAX_TIME = 600
STREAM_IDLE_TIMEOUT = 300

# ---------------------------------------------------------------------------
# Pricing -- read from the active provider profile's "pricing" block (USD per
# 1M tokens). A profile that declares no pricing yields None rates, and
# callers report the cost as unknown instead of inventing a number.

def active_pricing():
    return provider_pricing(config_active_provider())


# ---------------------------------------------------------------------------
# Session stats (thread-safe)

_stats_lock = threading.Lock()
_session_prompt_tokens = 0
_session_completion_tokens = 0
_session_total_tokens = 0
_session_cached_tokens = 0


def reset_session_stats():
    global _session_prompt_tokens, _session_completion_tokens
    global _session_total_tokens, _session_cached_tokens
    with _stats_lock:
        _session_prompt_tokens = 0
        _session_completion_tokens = 0
        _session_total_tokens = 0
        _session_cached_tokens = 0


# Costs use the active provider's configured rates. Each returns None when
# the profile declares no such rate, so callers can report "unknown" instead
# of a misleading $0.00.

def rate_cost(tokens, rate):
    return (tokens * rate / 1_000_000) if rate is not None else None


def prompt_cost(tokens):
    return rate_cost(tokens, pricing_ref(active_pricing(), "input"))


def cached_cost(tokens):
    return rate_cost(tokens, pricing_ref(active_pricing(), "cached_input"))


def completion_cost(tokens):
    return rate_cost(tokens, pricing_ref(active_pricing(), "output"))


# Cached input tokens are reported by the server in
# usage.prompt_tokens_details.cached_tokens and are part of prompt_tokens;
# bill them at the discounted rate and subtract them from the uncached pool.
def session_cost():
    """-> number, or None when the active profile declares no pricing at all."""
    rates = active_pricing()
    inp = pricing_ref(rates, "input")
    cached = pricing_ref(rates, "cached_input")
    out = pricing_ref(rates, "output")
    if inp is None and cached is None and out is None:
        return None
    with _stats_lock:
        pt = _session_prompt_tokens
        ca = _session_cached_tokens
        ct = _session_completion_tokens
    return ((rate_cost(max(0, pt - ca), inp) or 0)
            + (rate_cost(ca, cached) or 0)
            + (rate_cost(ct, out) or 0))


def print_session_stats():
    with _stats_lock:
        pt = _session_prompt_tokens
        ct = _session_completion_tokens
        tt = _session_total_tokens
        ca = _session_cached_tokens
    cost = session_cost()
    rates = active_pricing()
    print("")
    print("Session token usage:")
    print("  Prompt tokens:     {}".format(pt))
    print("  Completion tokens: {}".format(ct))
    print("  Total tokens:      {}".format(tt))
    if ca > 0:
        pct = 100.0 * ca / max(1, pt)
        print("  Cached tokens:     {} ({:.1f}% of prompt)".format(ca, pct))
    if cost is not None:
        print("  Estimated cost:    ${:.6f}  (${:.4f}/M input, ${:.4f}/M cached input, ${:.4f}/M output)".format(
            cost,
            pricing_ref(rates, "input") or 0,
            pricing_ref(rates, "cached_input") or 0,
            pricing_ref(rates, "output") or 0))
    else:
        print('  Estimated cost:    n/a (no "pricing" block for this provider)')


def accumulate_usage(data):
    global _session_prompt_tokens, _session_completion_tokens
    global _session_total_tokens, _session_cached_tokens
    usage = data.get("usage") or {}
    if isinstance(usage, dict) and usage:
        with _stats_lock:
            _session_prompt_tokens += usage.get("prompt_tokens", 0)
            _session_completion_tokens += usage.get("completion_tokens", 0)
            _session_total_tokens += usage.get("total_tokens", 0)
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                _session_cached_tokens += details.get("cached_tokens", 0)


# ---------------------------------------------------------------------------
# API key
#
# The env var name comes from the active provider profile's api_key_env when
# a harness config is loaded; falls back to FIREWORKS_API_KEY.

import os  # noqa: E402


def get_api_key():
    p = config_active_provider()
    env_name = (provider_api_key_env(p) if p else None) or "FIREWORKS_API_KEY"
    key = os.environ.get(env_name, "")
    if not key:
        raise RuntimeError("fireworks-ai: {} environment variable not set".format(env_name))
    return key


# ---------------------------------------------------------------------------
# SSE streaming helpers

def parse_sse_chunk(body):
    """Parse one SSE "data: {...}" body into a dict (or None on bad JSON)."""
    try:
        return json.loads(body)
    except Exception:
        return None


def parse_sse_response(resp):
    """Reconstruct the equivalent non-streaming chat-completions response from
    an SSE stream:

        { 'id': ..., 'model': ...,
          'choices': [ { 'message': <assistant msg>, 'finish_reason': ... } ],
          'usage': {...} }

    `message` carries accumulated 'content', (deepseek) 'reasoning_content',
    and (when present) a list of 'tool_calls' dicts exactly like a
    non-streaming response.
    """
    content_parts = []
    reasoning_parts = []
    tool_calls_by_index = {}  # index -> {'id','type','name','arguments'}
    usage = None
    finish_reason = None
    message_id = ""
    message_model = ""

    for raw in resp.iter_lines():
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        trimmed = line.strip()
        if trimmed == "" or trimmed.startswith(":"):  # comment / keep-alive
            continue
        if not trimmed.startswith("data:"):
            continue
        body = trimmed[5:].strip()
        if body == "[DONE]":
            continue
        chunk = parse_sse_chunk(body)
        if not isinstance(chunk, dict):
            continue
        # API-level error inside the stream
        if "error" in chunk:
            err = chunk["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError("API error: {}".format(msg))
        if "id" in chunk:
            message_id = chunk.get("id", "")
        if "model" in chunk:
            message_model = chunk.get("model", "")
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage
        for c in chunk.get("choices") or []:
            delta = c.get("delta") or {}
            fr = c.get("finish_reason")
            if fr is not None and fr != finish_reason:
                finish_reason = fr
            c_delta = delta.get("content")
            if isinstance(c_delta, str):
                content_parts.append(c_delta)
            r_delta = delta.get("reasoning_content")
            if isinstance(r_delta, str):
                reasoning_parts.append(r_delta)
            tc = delta.get("tool_calls")
            if isinstance(tc, list):
                for t in tc:
                    idx = t.get("index", 0)
                    entry = tool_calls_by_index.get(idx)
                    if entry is None:
                        entry = {"id": "", "type": "function", "name": "", "arguments": ""}
                        tool_calls_by_index[idx] = entry
                    t_id = t.get("id")
                    if isinstance(t_id, str) and t_id != "":
                        entry["id"] = t_id
                    t_type = t.get("type")
                    if isinstance(t_type, str):
                        entry["type"] = t_type
                    f = t.get("function")
                    if isinstance(f, dict):
                        f_name = f.get("name")
                        if isinstance(f_name, str) and f_name != "":
                            entry["name"] = f_name
                        f_args = f.get("arguments")
                        if isinstance(f_args, str) and f_args != "":
                            entry["arguments"] += f_args

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    idxs = sorted(tool_calls_by_index.keys())
    tool_calls = None
    if idxs:
        tool_calls = []
        for idx in idxs:
            e = tool_calls_by_index[idx]
            tool_calls.append({
                "id": e["id"],
                "type": e["type"],
                "function": {"name": e["name"], "arguments": e["arguments"]},
            })
    if tool_calls:
        message = {"role": "assistant", "content": content,
                   "reasoning_content": reasoning, "tool_calls": tool_calls}
    else:
        message = {"role": "assistant", "content": content,
                   "reasoning_content": reasoning}
    return {
        "id": message_id,
        "model": message_model,
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage or {},
    }


# ---------------------------------------------------------------------------
# Low-level POST (streaming)

def post_fireworks(payload):
    api_key = get_api_key()
    p = config_active_provider()
    endpoint = (provider_endpoint(p) if p else None)
    if not endpoint:
        raise RuntimeError('fireworks-ai: active provider profile has no "endpoint"; '
                           "set it in the harness config")
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "authorization": "Bearer " + api_key,
    }
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    stream_payload["stream_options"] = {"include_usage": True}
    if debug_log:
        dbg = {k: v for k, v in stream_payload.items() if k != "messages"}
        print("[DEBUG] request: {}".format(json.dumps(dbg)))
    try:
        resp = requests.post(endpoint, headers=headers, json=stream_payload,
                             stream=True,
                             timeout=(CONNECT_MAX_TIME, STREAM_IDLE_TIMEOUT))
        data = parse_sse_response(resp)
        resp.close()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("fireworks-ai: HTTP error: {}".format(e))
    if debug_log:
        print("[DEBUG] response: {}".format(json.dumps(data)))
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError("API error: {}".format(msg))
    accumulate_usage(data)
    if "choices" not in data:
        raise RuntimeError("fireworks-ai: response has no 'choices'. Raw: {}".format(json.dumps(data)))
    return data


# ---------------------------------------------------------------------------
# chat / chat_with_tools -- thin wrappers over the shared provider-agnostic
# loop in chat_loop.py (also used by mlx_serve.py).
#
# Generation defaults come from the active provider profile's "generation"
# section when a harness config is loaded; explicit keyword args win.
# A missing model is an error; missing generation parameters are left out of
# the request so the server's own default applies.

def active_model_id():
    p = config_active_provider()
    m = provider_model(p) if p else None
    if not m:
        raise RuntimeError('fireworks-ai: active provider profile has no "model"; '
                           "set it in the harness config")
    return m


def gen_param(key):
    return generation_ref(provider_generation(config_active_provider()), key, None)


def chat(messages, model_id=None, max_tokens=None, temperature=None):
    return chat_star(post_fireworks, messages,
                     model_id=model_id or active_model_id(),
                     max_tokens=max_tokens if max_tokens is not None else gen_param("max_tokens"),
                     temperature=temperature if temperature is not None else gen_param("temperature"))


def chat_with_tools(messages, tools, model_id=None, max_tokens=None,
                    temperature=None, max_iterations=20):
    return chat_with_tools_star(post_fireworks, messages, tools,
                                model_id=model_id or active_model_id(),
                                max_tokens=max_tokens if max_tokens is not None else gen_param("max_tokens"),
                                temperature=temperature if temperature is not None else gen_param("temperature"),
                                max_iterations=max_iterations)
```

Two aspects deserve attention.

**Timeouts for streaming.** A total wall-clock timeout would kill long generations, which is wrong for a model that is still producing tokens. The module instead uses three separate bounds: `CONNECT_MAX_TIME = 10` seconds to open the TCP connection, `HEADER_MAX_TIME = 600` as the time-to-first-token allowance, and `STREAM_IDLE_TIMEOUT = 300` as the read timeout, which `requests` applies *per chunk*. A response that keeps producing tokens never trips it. Only a truly silent connection does.

**Cost accounting.** Each response's `usage` is added to thread-safe session counters. The profile's `pricing` block gives USD rates per 1,000,000 tokens. For `p`$ prompt tokens, `a`$ of those cached, and `o`$ completion tokens, with input, cached-input, and output rates `r_i`$, `r_c`$, and `r_o`$, the session cost is

```$
C = \frac{(p - a)\, r_i + a\, r_c + o\, r_o}{10^{6}}
```

Cached prompt tokens are part of `prompt_tokens`, so the formula bills them at the discounted rate by subtracting them from the uncached pool. This matters in an agentic loop because the same long message prefix is resent on every iteration, and prompt caching typically turns 60 to 90 percent of a session's prompt tokens into cached ones, at one fifth the price.

The functions `prompt_cost`, `cached_cost`, and `completion_cost` each return `None` when the profile omits that rate. `session_cost` returns `None` when the whole `pricing` block is missing, and the printer reports `n/a`. The lesson: a missing price is *unknown*, not zero, and reporting an unknown cost as `$0.00` is a bug users will not forgive.

## The Local Client: mlx_serve.py

Local inference needs neither keys nor streaming. `mlx_lm.server` and its siblings already speak the OpenAI protocol that `chat_loop.py` consumes, so this module is thin: forward the payload, return the JSON, and accumulate usage. It mirrors `fireworks_ai.py` function for function so `agent.py` can swap providers by changing which module it calls.

```python
# mlx_serve.py -- local MLX (mlx_lm.server) API client, session stats, chat
# helpers. Python port of mlx-serve.rkt. Mirrors the interface of
# fireworks_ai.py so agent.py can swap providers with /provider or
# AGENT_PROVIDER=mlx.
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# The endpoint is the OpenAI-compatible /v1/chat/completions route served by
# mlx_lm.server (port 11434), oMLX (port 8000), sushi (port 12345), and
# Ollama's own OpenAI shim. That protocol already matches what chat_loop.py
# consumes (choices[].message with optional tool_calls, usage.prompt_tokens /
# completion_tokens), so there is no message re-shaping here: we forward the
# OpenAI payload and return the response as-is. A Bearer key from the provider
# profile is passed along but ignored by a local server.
#
# Endpoint, model, api_key_env, and generation parameters all come from the
# active provider profile in the harness config; nothing provider-specific is
# compiled in here.

import os
import threading

import requests

from harness_config import (config_active_provider, provider_api_key_env,
                            provider_endpoint, provider_generation,
                            provider_model, generation_ref)
from chat_loop import chat as chat_star
from chat_loop import chat_with_tools as chat_with_tools_star
import fireworks_ai  # for the shared debug_log toggle

# The provider dict MLX requests consult for endpoint/model/generation.
# agent.py sets this to the active profile (None = use the harness config's
# active provider).
mlx_active_provider = None

# The OpenAI-compatible endpoint returns reasoning in the assistant message's
# 'reasoning' field, which chat_loop.py ignores (it only reads 'content' and
# 'tool_calls'), so there is no separate thinking toggle to wire up here.
MLX_THINK = False

# Non-streaming request: the whole generation must complete within this
# window. Local models on large weights can be slow, so be generous.
MLX_MAX_TIME = 900
MLX_CONNECT_TIME = 10


def current_provider_json():
    return mlx_active_provider or config_active_provider() or {}


# ---------------------------------------------------------------------------
# Session stats (thread-safe). mlx_lm.server reports prompt_tokens /
# completion_tokens on every /v1/chat/completions response. Local inference is
# free, so stats are informational only -- estimated cost is always $0.

_stats_lock = threading.Lock()
_session_prompt_tokens = 0
_session_completion_tokens = 0


def mlx_reset_session_stats():
    global _session_prompt_tokens, _session_completion_tokens
    with _stats_lock:
        _session_prompt_tokens = 0
        _session_completion_tokens = 0


def mlx_accumulate_usage(usage):
    global _session_prompt_tokens, _session_completion_tokens
    if isinstance(usage, dict) and usage:
        with _stats_lock:
            _session_prompt_tokens += usage.get("prompt_tokens", 0)
            _session_completion_tokens += usage.get("completion_tokens", 0)


def mlx_print_session_stats():
    with _stats_lock:
        pt = _session_prompt_tokens
        ct = _session_completion_tokens
    print("")
    print("Session token usage (local MLX -- no API cost):")
    print("  Prompt tokens:     {}".format(pt))
    print("  Completion tokens: {}".format(ct))
    model = provider_model(current_provider_json()) or "?"
    print("  Estimated cost:    $0  (local model {})".format(model))


# ---------------------------------------------------------------------------
# Low-level POST (non-streaming, OpenAI-compatible)

def mlx_api_key(provider):
    env_name = provider_api_key_env(provider)
    return os.environ.get(env_name) if env_name else None


def post_mlx(payload):
    provider = current_provider_json()
    # chat_loop.py already builds a complete OpenAI-shaped body, including
    # tools/tool_choice when present, so it is forwarded as-is.
    request_body = payload
    endpoint = provider_endpoint(provider)
    if not endpoint:
        raise RuntimeError('mlx-serve: active provider profile has no "endpoint"; '
                           "set it in the harness config")
    key = mlx_api_key(provider)
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = "Bearer " + key
    if fireworks_ai.debug_log:
        print("[DEBUG] mlx request ({}): {}".format(endpoint, json_dumps(request_body)))
    try:
        resp = requests.post(endpoint, headers=headers, json=request_body,
                             timeout=(MLX_CONNECT_TIME, MLX_MAX_TIME))
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("mlx-serve: HTTP error: {}".format(e))
    if fireworks_ai.debug_log:
        print("[DEBUG] mlx response: {}".format(json_dumps(data)))
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError("MLX API error: {}".format(msg))
    if "choices" not in data:
        raise RuntimeError("MLX response has no 'choices'. Raw: {}".format(json_dumps(data)))
    mlx_accumulate_usage(data.get("usage") or {})
    return data


def json_dumps(x):
    import json
    try:
        return json.dumps(x)
    except Exception:
        return str(x)


# ---------------------------------------------------------------------------
# mlx_chat / mlx_chat_with_tools -- same signatures as fireworks_ai.py.
# Generation parameters resolve from the active profile; a missing model is
# an error, and missing generation parameters are simply left out of the
# request.

def m_gen_param(key):
    return generation_ref(provider_generation(current_provider_json()), key, None)


def m_model_id():
    m = provider_model(current_provider_json())
    if not m:
        raise RuntimeError('mlx-serve: active provider profile has no "model"; '
                           "set it in the harness config")
    return m


def mlx_chat(messages, model_id=None, max_tokens=None, temperature=None):
    return chat_star(post_mlx, messages,
                     model_id=model_id or m_model_id(),
                     max_tokens=max_tokens if max_tokens is not None else m_gen_param("max_tokens"),
                     temperature=temperature if temperature is not None else m_gen_param("temperature"))


def mlx_chat_with_tools(messages, tools, model_id=None, max_tokens=None,
                        temperature=None, max_iterations=20):
    return chat_with_tools_star(post_mlx, messages, tools,
                                model_id=model_id or m_model_id(),
                                max_tokens=max_tokens if max_tokens is not None else m_gen_param("max_tokens"),
                                temperature=temperature if temperature is not None else m_gen_param("temperature"),
                                max_iterations=max_iterations)
```

Note `MLX_MAX_TIME = 900`. A non-streaming local request must finish generation inside one read window, and large quantized models on a laptop can be slow. Be generous.

Note too that `print_session_stats` here always reports `$0`. Local inference is free per token; you paid for it in hardware and electricity. The counts are still printed, because they tell you whether the context window is getting large.

## Web Search: search.py

Coding questions are often really documentation questions ("what changed in this library's API since version 2"). The harness can optionally search the web and insert the results into the prompt. Two backends are implemented, both returning `(url, title, snippet)` triples.

The Brave API returns results under a nested `web.results` key:

```json
{
  "web": {
    "results": [
      { "url": "https://docs.example.com/api",
        "title": "HTTP Client Reference",
        "description": "Configuration options, timeouts, and session reuse." }
    ]
  }
}
```

Exa returns a flat `results` list with a `highlights` array per result, and the code takes the first highlight as the snippet. Here is the module:

```python
# search.py -- Brave Search and Exa AI search backends
# Python port of search.rkt (originally py-coding-agent/search.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details

import os

import requests

EXA_ENDPOINT = "https://api.exa.ai/search"


# ---------------------------------------------------------------------------
# Brave Search
# Returns list of (url, title, description)

def brave_search(query, num_results=5):
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        raise RuntimeError("brave-search: BRAVE_SEARCH_API_KEY environment variable not set")
    url = ("https://api.search.brave.com/res/v1/web/search?q={}&count={}"
           .format(requests.utils.quote(query), num_results))
    headers = {
        "X-Subscription-Token": api_key,
        "content-type": "application/json",
        "accept": "application/json",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    data = resp.json()
    web = data.get("web", {})
    results = web.get("results", [])
    return [(r.get("url", ""), r.get("title", ""), r.get("description", ""))
            for r in results]


# ---------------------------------------------------------------------------
# Exa AI Search
# Returns list of (url, title, highlight)

def exa_search(query, num_results=5):
    api_key = os.environ.get("EXA_SEARCH_API_KEY", "")
    if not api_key:
        raise RuntimeError("exa-search: EXA_SEARCH_API_KEY environment variable not set")
    payload = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {"highlights": True},
    }
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + api_key,
    }
    resp = requests.post(EXA_ENDPOINT, headers=headers, json=payload, timeout=30)
    data = resp.json()
    results = data.get("results", [])
    out = []
    for r in results:
        hl = r.get("highlights") or []
        out.append((r.get("url", ""), r.get("title", ""),
                    hl[0] if isinstance(hl, list) and hl else ""))
    return out
```

When search is on, `format_search_results` in `agent.py` turns the triples into a numbered block and prepends it to the user's question, so the model receives the evidence and the ask together.

## The REPL: agent.py

The main module wires everything together: config loading, environment overrides, CLI parsing, the REPL, slash commands, intent routing, and context management. Read it in full, then we will take the tour.

```python
# agent.py -- main REPL loop and CLI entry point
# Python port of agent.rkt (originally py-coding-agent/agent.py).
#
# Copyright (C) 2026 Mark Watson <markw@markwatson.com>
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
# See LICENSE file for details
#
# Configuration comes from the hierarchical harness config:
#   global  ~/.coding_harness.json
#   local   .local_coding_harness.json  (overrides the global file)
# Precedence: CLI flags > env > harness/legacy config file > profile defaults.

import argparse
import os
import sys

import approval
import fireworks_ai
import harness_config as hc
import line_input
import mlx_serve
import search as search_mod
import tools
from harness_config import (config_active_provider, config_active_provider_name,
                            config_provider, config_provider_names,
                            config_set_active_provider, generation_ref,
                            provider_generation, provider_model, provider_type)
from tools import ENABLED_TOOLS, register_all

VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# Prompts

SYSTEM_PROMPT_TEMPLATE = (
    "You are an interactive coding assistant working in the directory {cwd}.\n\n"
    "Rules:\n"
    "- Use read_file, list_dir, and grep to understand the code BEFORE proposing edits.\n"
    "- To EDIT an existing file: read_file it first, then pass its exact current contents\n"
    "  as `old` to propose_edit.\n"
    "- To CREATE a new file: call propose_edit with the empty string \"\" as `old` and\n"
    "  the full desired contents as `new`. Do not call read_file first for a file that\n"
    "  does not exist yet.\n"
    "- One file per propose_edit call. Keep diffs small and focused.\n"
    "- If the user rejects an edit or `make check` fails, ask for clarification instead\n"
    "  of retrying blindly.\n"
    "- run_shell only accepts whitelisted commands: make, ls, pwd, cat, uv.\n"
    "- When you are done, reply with a short natural-language summary of what changed."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question clearly and concisely "
    "using the web search results provided. Do not reference files, directories, or "
    "code editing tools unless the user explicitly asks about code."
)

COMPACT_SYSTEM_PROMPT = (
    "You are a context compactor for a coding assistant. Summarize the conversation "
    "transcript into a compact brief that will replace it. Preserve: the user's goals "
    "and instructions, decisions made, files created or modified (with paths), "
    "important code and tool-output details, and outstanding tasks. "
    "Write dense bullets, no preamble."
)

GENERAL_KEYWORDS = [
    "movie", "film", "cinema", "theater", "theatre", "showing", "playing", "showtime",
    "weather", "forecast", "rain", "snow", "temperature outside",
    "restaurant", "recipe", "menu", "where to eat",
    "news", "sports", "score", "standings",
    "near me", "nearby", "directions to",
    "hotel", "flight", "travel", "vacation",
    "population of", "history of", "capital of",
    "who is ", "who was ", "where is ", "when is ", "when does ",
    "price of", "cost of", "how much does",
]

CODING_KEYWORDS = [
    ".lisp", ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb", ".rs", ".c ",
    "def ", "class ", "function ", "refactor", "implement ", "compile", "makefile",
    "stacktrace", "segfault", "git commit", "git push", "git pull",
    "unit test", "pull request", "fix the bug", "add a function", "write a function",
]

# ---------------------------------------------------------------------------
# LLM provider dispatch
#
# Providers are named profiles declared in ~/.coding_harness.json and/or
# .local_coding_harness.json (see harness_config.py). Two wire types are
# supported: "mlx" (local mlx_lm.server -- OpenAI-compatible
# /v1/chat/completions, no API key) and "openai" (Fireworks.ai and compatible
# endpoints). Session-level model override set by /model or --model; None
# means "use the model declared by the active provider profile".

_model_override = None


def config_loaded():
    return bool(hc.harness_config)


def active_provider_hash():
    return config_active_provider() if config_loaded() else None


def require_provider(who):
    """-> provider dict, or a clear error when nothing is configured."""
    p = active_provider_hash()
    if p is None:
        raise RuntimeError(
            "{}: no active provider profile; define \"providers\" in "
            "~/.coding_harness.json or .local_coding_harness.json".format(who))
    return p


def active_provider_type():
    """-> 'mlx' | 'openai'  (wire format of the active chat provider)"""
    p = active_provider_hash()
    return provider_type(p) if p else "openai"


def using_mlx():
    return active_provider_type() == "mlx"


def current_provider_name_or_legacy():
    return config_active_provider_name() or "?"


def current_model_id():
    return (_model_override
            or (provider_model(active_provider_hash()) if active_provider_hash() else None)
            or "?")


def set_current_model(m):
    """/model <id> or --model: override the active profile's model this session."""
    global _model_override
    _model_override = m


def switch_provider(name):
    """Select a profile and drop any model override so the new profile's own
    model takes effect (provider selection is always applied before --model)."""
    global _model_override
    active = config_set_active_provider(name)
    _model_override = None
    return active


def profile_gen(provider, key, explicit):
    return explicit if explicit is not None else generation_ref(provider_generation(provider), key, None)


# Mutable session state
messages = []
search_enabled = False
search_engine = "brave"  # "brave" or "exa"

# CLI / config state
cli_quiet = False
cli_plain = False

# Exit codes
EXIT_OK = 0
EXIT_MODEL_ERROR = 1
EXIT_CHECK_FAILED = 2
EXIT_REJECTED = 3
EXIT_BAD_ARGS = 5


def exit_with_code(code):
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


# ---------------------------------------------------------------------------
# Config file + env
# Precedence: CLI flags > env > harness config > compiled defaults
# Legacy env namespaced aliases (override config, still below CLI):
#   AGENT_PROVIDER / CODING_AGENT_PROVIDER, CODING_AGENT_MODEL,
#   CODING_AGENT_QUIET, CODING_AGENT_PLAIN, CODING_AGENT_DEBUG

def apply_harness_flags():
    """Apply top-level lifestyle flags from the harness JSON config
    (quiet/plain/debug/search). Provider sections are consumed lazily by the
    dispatch functions; here we just set the display/session flags."""
    global cli_quiet, cli_plain, search_enabled, search_engine
    cfg = hc.harness_config
    if cfg.get("quiet"):
        cli_quiet = True
        tools.quiet_mode = True
    if cfg.get("plain"):
        cli_plain = True
        approval.color_enabled = False
    if cfg.get("debug"):
        fireworks_ai.debug_log = True
    s = cfg.get("search")
    if isinstance(s, dict):
        se = s.get("engine")
        if isinstance(se, str) and se.lower() in ("brave", "exa"):
            search_engine = se.lower()
        if "enabled" in s:
            search_enabled = bool(s.get("enabled"))


def apply_env_overrides():
    global cli_quiet, cli_plain
    env_provider = os.environ.get("CODING_AGENT_PROVIDER") or os.environ.get("AGENT_PROVIDER")
    if env_provider and env_provider.strip() != "":
        low = env_provider.strip().lower()
        if config_provider(low):
            switch_provider(low)
        else:
            names = config_provider_names()
            sys.stderr.write(
                "warning: ignoring unknown provider '{}' from the environment{}\n".format(
                    env_provider,
                    " (available: {})".format(", ".join(names)) if names else " (no providers configured)"))
    env_model = os.environ.get("CODING_AGENT_MODEL")
    if env_model:
        set_current_model(env_model)
    if os.environ.get("CODING_AGENT_QUIET"):
        cli_quiet = True
        tools.quiet_mode = True
    if os.environ.get("CODING_AGENT_PLAIN"):
        cli_plain = True
        approval.color_enabled = False
    if os.environ.get("CODING_AGENT_DEBUG"):
        fireworks_ai.debug_log = True


# ---------------------------------------------------------------------------
# CLI helpers

def read_all_stdin():
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def build_prompt(positional, prompt_parts, stdin_text):
    """Join all sources with newlines; trim; return "" if nothing."""
    parts = []
    if stdin_text and stdin_text.strip() != "":
        parts.append(stdin_text.strip())
    parts.extend(prompt_parts)
    parts.extend(positional)
    parts = [p for p in parts if isinstance(p, str) and p.strip() != ""]
    return "\n\n".join(parts).strip()


def resolve_cwd(dir_str):
    if dir_str:
        if not os.path.isdir(dir_str):
            sys.stderr.write("error: --cwd directory does not exist: {}\n".format(dir_str))
            exit_with_code(EXIT_BAD_ARGS)
        os.chdir(dir_str)


# ---------------------------------------------------------------------------
# Skills:  ~/.agents/skills/<name>/SKILL.md

SKILLS_DIR = os.path.expanduser("~/.agents/skills")


def list_skills():
    if not os.path.isdir(SKILLS_DIR):
        return []
    names = []
    for e in sorted(os.listdir(SKILLS_DIR)):
        if os.path.isfile(os.path.join(SKILLS_DIR, e, "SKILL.md")):
            names.append(e)
    return names


def skill_file(name):
    return os.path.join(SKILLS_DIR, name, "SKILL.md")


def skill_exists(name):
    return os.path.isfile(skill_file(name))


def skill_description(name):
    """Parse the YAML frontmatter for a `description:` field; return None if not found."""
    try:
        with open(skill_file(name), "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        lines = text.split("\n")
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if line.startswith("description:"):
                    return line[len("description:"):].strip()
        return None
    except Exception:
        return None


def show_skills():
    skills = list_skills()
    if not skills:
        print("No skills found in {}".format(SKILLS_DIR))
    else:
        print("Available skills (from {}):".format(SKILLS_DIR))
        for s in skills:
            desc = skill_description(s)
            print("  /{} — {}".format(s, desc) if desc else "  /{}".format(s))
        print("\nType /<skill-name> to load a skill into the conversation.")


def load_skill(name):
    if not skill_exists(name):
        print("Unknown command or skill: /{}  (try /skills)".format(name))
        return
    try:
        with open(skill_file(name), "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:  # noqa: BLE001
        print("Error loading skill '{}': {}".format(name, e))
        return
    system_msg = {
        "role": "system",
        "content": ("The user has loaded the following skill: '{}'. ".format(name)
                    + "Use it as authoritative reference and guidance for subsequent "
                      "responses in this conversation.\n\n" + content),
    }
    messages.append(system_msg)
    desc = skill_description(name)
    print("Loaded skill: {}  ({} chars)".format(name, len(content)))
    if desc:
        print("  {}".format(desc))
    print("\nAsking the model to load the skill into context...")
    try:
        reply = llm_chat(list(messages))
    except Exception as e:  # noqa: BLE001
        reply = "(model call failed: {})".format(e)
    print("\n{}".format(clean(reply)))


# ---------------------------------------------------------------------------
# Helpers

def clean(text):
    return text.strip() if isinstance(text, str) else str(text).strip()


def reset_conversation():
    cwd = os.getcwd()
    prompt = SYSTEM_PROMPT_TEMPLATE.replace("{cwd}", cwd)
    messages.clear()
    messages.append({"role": "system", "content": prompt})


def print_banner():
    if not cli_quiet:
        print("")
        print("Coding Agent REPL.  /help for commands, /quit to exit.")
        print("  cwd:      {}".format(os.getcwd()))
        print("  provider: {}".format(current_provider_name_or_legacy()))
        print("  model:    {}".format(current_model_id()))
        print("")


def show_history():
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or "(no content)"
        print("\n--- {} ---\n{}".format(role, content))


def message_char_size(msg):
    """Approximate size of a message's contribution to the model context."""
    def slen(v):
        return len(v) if isinstance(v, str) else 0
    total = slen(msg.get("content", "")) + slen(msg.get("reasoning_content", ""))
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            f = tc.get("function") or {}
            total += slen(f.get("name", "")) + slen(f.get("arguments", ""))
    return total


def message_preview(msg):
    """Full single-line preview text (whitespace collapsed); wrapping is done
    by wrap_preview at display time."""
    if msg.get("role") == "tool":
        raw = "[{}] {}".format(msg.get("name", "?"), msg.get("content") or "")
    else:
        content = msg.get("content", "")
        tcs = msg.get("tool_calls")
        if isinstance(content, str) and content.strip() != "":
            raw = content
        elif isinstance(tcs, list) and tcs:
            raw = "[tool calls: {}]".format(
                ", ".join((tc.get("function") or {}).get("name", "?") for tc in tcs))
        else:
            raw = "(no content)"
    return " ".join(str(raw).split())


PREVIEW_WIDTH = 60
PREVIEW_MAX_LINES = 3


def wrap_preview(s):
    """Wrap s at PREVIEW_WIDTH (breaking on the last space in the window when
    possible) into at most PREVIEW_MAX_LINES lines; "…" marks text that still
    does not fit."""
    n = len(s)
    lines = []
    start = 0
    while True:
        if start >= n:
            return lines
        if n - start <= PREVIEW_WIDTH:
            lines.append(s[start:])
            return lines
        if len(lines) == PREVIEW_MAX_LINES - 1:
            lines.append(s[start:start + PREVIEW_WIDTH - 1] + "…")
            return lines
        window = s[start:start + PREVIEW_WIDTH]
        bp = None
        for i, ch in enumerate(window):
            if ch == " ":
                bp = i + 1
        use = bp if bp else PREVIEW_WIDTH
        lines.append(s[start:start + use].strip())
        start += use


def show_context():
    total = sum(message_char_size(m) for m in messages)
    print("")
    print("Context: {} message{}, {} chars, {} tokens (est.)".format(
        len(messages), "" if len(messages) == 1 else "s", total, total // 4))
    print("")
    print(" {:>3}  {:<9}  {:>7}  {}".format("#", "role", "chars", "preview"))
    print(" {:<3}  {:<9}  {:<7}  {:<50}".format("---", "---------", "-------", "-" * 50))
    for i, m in enumerate(messages, 1):
        lines = wrap_preview(message_preview(m))
        print(" {:>3}  {:<9}  {:>7}  {}".format(i, m.get("role", "?"),
                                                message_char_size(m), lines[0]))
        for extra in lines[1:]:
            print(" {:<3}  {:<9}  {:<7}  {}".format("", "", "", extra))
    print("")


def compact_context():
    if len(messages) <= 2:
        print("Nothing to compact — conversation is already short.")
        return
    before = sum(message_char_size(m) for m in messages)
    print("Compacting {} messages ({} chars)…".format(len(messages), before))
    transcript_parts = []
    for m in messages:
        tcs = m.get("tool_calls")
        header = "### {}".format(m.get("role", "?"))
        if isinstance(tcs, list) and tcs:
            header += " (tool calls: {})".format(
                ", ".join((tc.get("function") or {}).get("name", "?") for tc in tcs))
        content = m.get("content", "")
        transcript_parts.append("{}\n{}".format(header, content if isinstance(content, str) else ""))
    transcript = "\n\n".join(transcript_parts)
    try:
        summary = llm_chat([
            {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ])
    except Exception as e:  # noqa: BLE001
        print("Compaction failed: {}".format(e))
        return
    if summary:
        # Keep the original system prompt; replace the rest with the summary.
        messages[:] = [
            messages[0],
            {"role": "user",
             "content": ("[Earlier conversation compacted to this summary. "
                         "Continue from where it left off.]\n\n" + summary)},
        ]
        after = sum(message_char_size(m) for m in messages)
        print("Compacted: {} → {} chars.".format(before, after))
        show_context()


# ---------------------------------------------------------------------------
# Slash commands

HELP_TEXT = """
Commands:
  /reset            clear conversation
  /history          dump message log
  /context          show a formatted summary of the current context
  /compact          compact history into a summary, then show the new context
  /model <id>       switch model (for the current provider)
  /provider         show current LLM provider and available profiles
  /provider <name>  switch provider profile (or fireworks/mlx w/o config)
  /debug            toggle raw request/response logging
  /search           toggle web search on/off
  /search brave     enable Brave search
  /search exa       enable Exa search
  /tokens           show session token usage and estimated cost
  /skills           list available skills in ~/.agents/skills
  /<skill-name>     load that skill into the conversation
  /quit             exit
"""


def handle_slash_command(line):
    """Returns 'quit', 'continue', or None (not a command)."""
    global search_enabled, search_engine
    if line == "" or line == "/quit":
        return "quit"
    if line == "/reset":
        reset_conversation()
        print("Conversation reset.")
        return "continue"
    if line == "/history":
        show_history()
        return "continue"
    if line == "/context":
        show_context()
        return "continue"
    if line == "/compact":
        compact_context()
        return "continue"
    if line.startswith("/model "):
        new_model = line[len("/model "):].strip()
        set_current_model(new_model)
        print("Model set to {}".format(new_model))
        return "continue"
    if line == "/provider":
        print("Current provider: {} (model: {})".format(
            current_provider_name_or_legacy(), current_model_id()))
        if config_loaded():
            print("Available profiles: {}".format(", ".join(config_provider_names())))
        return "continue"
    if line.startswith("/provider "):
        p = line[len("/provider "):].strip().lower()
        if config_provider(p):
            switch_provider(p)
            print("Provider set to profile '{}' (model: {})".format(p, current_model_id()))
        else:
            names = config_provider_names()
            print("Unknown provider '{}'{}".format(
                p,
                " -- use one of: {}".format(", ".join(names)) if names else " -- no providers configured"))
        return "continue"
    if line == "/debug":
        fireworks_ai.debug_log = not fireworks_ai.debug_log
        print("Debug logging {}".format("ON" if fireworks_ai.debug_log else "OFF"))
        return "continue"
    if line == "/tokens":
        if using_mlx():
            mlx_serve.mlx_print_session_stats()
        else:
            fireworks_ai.print_session_stats()
        return "continue"
    if line == "/search":
        search_enabled = not search_enabled
        print("Web search {} (engine: {})".format("ON" if search_enabled else "OFF", search_engine))
        return "continue"
    if line.startswith("/search "):
        engine = line[len("/search "):].strip().lower()
        if engine in ("brave", "exa"):
            search_engine = engine
            search_enabled = True
            print("Web search ON (engine: {})".format(engine))
        else:
            print("Unknown engine '{}' — use 'brave' or 'exa'".format(engine))
        return "continue"
    if line == "/help":
        print(HELP_TEXT)
        return "continue"
    if line == "/skills":
        show_skills()
        return "continue"
    if line.startswith("/"):
        # Any other /xxx  --  treat as a skill name lookup.
        load_skill(line[1:])
        return "continue"
    return None


# ---------------------------------------------------------------------------
# Intent classification

def heuristic_classify(lower):
    if any(kw in lower for kw in GENERAL_KEYWORDS):
        return "general"
    if any(kw in lower for kw in CODING_KEYWORDS):
        return "coding"
    return None


def llm_classify(user_line):
    try:
        msgs = [
            {"role": "system",
             "content": "You are a one-word query classifier. Reply with exactly one word and nothing else."},
            {"role": "user",
             "content": ("Classify this query as exactly one word — GENERAL, CODING, or HYBRID:\n"
                         "GENERAL = factual or informational; nothing to do with writing, editing, or debugging code.\n"
                         "CODING  = writing, editing, refactoring, or debugging code or files.\n"
                         "HYBRID  = coding question that benefits from web docs or library references.\n"
                         "Query: {}\n"
                         "One-word answer:").format(user_line)},
        ]
        raw = llm_chat(msgs, max_tokens=10, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        print("[Classifier LLM error: {} — defaulting to coding]".format(e))
        return "coding"
    up = raw.strip().upper()
    if "GENERAL" in up:
        return "general"
    if "HYBRID" in up:
        return "hybrid"
    return "coding"


def classify_intent(user_line):
    return heuristic_classify(user_line.lower()) or llm_classify(user_line)


# ---------------------------------------------------------------------------
# Search integration

def run_search(query):
    if search_engine == "exa":
        return search_mod.exa_search(query)
    return search_mod.brave_search(query)


def format_search_results(query, results):
    lines = ['[Web search results for: "{}"]'.format(query)]
    for i, r in enumerate(results, 1):
        lines.append("{}. {}\n   {}\n   {}".format(i, r[1], r[0], r[2] or ""))
    return "\n".join(lines) + "\n---"


def maybe_search(user_line, force):
    if not (force or search_enabled):
        return None
    try:
        results = run_search(user_line)
        if results:
            return format_search_results(user_line, results) + "\n" + user_line
        return None
    except Exception as e:  # noqa: BLE001
        print("[Search error ({}): {}]".format(search_engine, e))
        return None


# ---------------------------------------------------------------------------
# LLM dispatch (provider-profile-aware)

def llm_chat(msgs, max_tokens=None, temperature=None):
    """Plain (no-tools) chat: dispatches on the active provider profile."""
    p = require_provider("llm-chat")
    mt = profile_gen(p, "max_tokens", max_tokens)
    tp = profile_gen(p, "temperature", temperature)
    if provider_type(p) == "mlx":
        mlx_serve.mlx_active_provider = p
        return mlx_serve.mlx_chat(msgs, model_id=current_model_id(),
                                  max_tokens=mt, temperature=tp)
    return fireworks_ai.chat(msgs, model_id=current_model_id(),
                             max_tokens=mt, temperature=tp)


def llm_chat_with_tools(msgs, tools_list):
    """Agentic (tool-calling) chat -- uses the same active provider as plain chat."""
    p = require_provider("llm-chat-with-tools")
    if provider_type(p) == "mlx":
        mlx_serve.mlx_active_provider = p
        return mlx_serve.mlx_chat_with_tools(msgs, tools_list, model_id=current_model_id())
    return fireworks_ai.chat_with_tools(msgs, tools_list, model_id=current_model_id())


# ---------------------------------------------------------------------------
# Send to model (intent-routed)

INTENT_LABELS = {
    "general": "web search, no coding tools",
    "coding": "coding tools, no search",
    "hybrid": "coding tools + web search if /search is on",
}


def send_to_model(user_line):
    global messages
    intent = classify_intent(user_line)
    if not cli_quiet:
        print("[intent: {} → {}]".format(intent, INTENT_LABELS[intent]))
    if intent == "general":
        content = maybe_search(user_line, True) or user_line
        msgs = [
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        reply = llm_chat(msgs)
        print("\n{}".format(clean(reply)))
    elif intent == "coding":
        updated = messages + [{"role": "user", "content": user_line}]
        reply, new_messages = llm_chat_with_tools(updated, ENABLED_TOOLS)
        messages = new_messages
        print("\n{}".format(clean(reply)))
    else:  # hybrid
        content = maybe_search(user_line, False) or user_line
        updated = messages + [{"role": "user", "content": content}]
        reply, new_messages = llm_chat_with_tools(updated, ENABLED_TOOLS)
        messages = new_messages
        print("\n{}".format(clean(reply)))


# ---------------------------------------------------------------------------
# One-shot helpers: infer exit code from tool results

def tool_messages_contain(substr):
    sub = substr.lower()
    return any(m.get("role") == "tool"
               and sub in str(m.get("content", "")).lower()
               for m in messages)


def infer_exit_code():
    if tool_messages_contain("make check failed"):
        return EXIT_CHECK_FAILED
    if tool_messages_contain("user rejected") or tool_messages_contain("user skipped"):
        return EXIT_REJECTED
    return EXIT_OK


def run_one_shot(prompt):
    """Prompt is already non-empty string. Run single task, print reply, exit."""
    register_all()
    reset_conversation()
    # One-shot still respects quiet/plain but banner is suppressed anyway
    try:
        send_to_model(prompt)
        exit_code = infer_exit_code()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("Error talking to model: {}\n".format(e))
        exit_with_code(EXIT_MODEL_ERROR)
    if exit_code == EXIT_CHECK_FAILED:
        if not cli_quiet:
            print("\n[make check failed]")
        exit_with_code(EXIT_CHECK_FAILED)
    if exit_code == EXIT_REJECTED:
        if not cli_quiet:
            print("\n[change rejected or skipped]")
        exit_with_code(EXIT_REJECTED)
    exit_with_code(EXIT_OK)


# ---------------------------------------------------------------------------
# Interactive line editing (readline when available)

SLASH_COMMANDS = ["/reset", "/history", "/context", "/compact", "/model", "/provider",
                  "/debug", "/search", "/tokens", "/help", "/skills", "/quit"]

SEARCH_ENGINES = ["brave", "exa"]


def configured_model_ids():
    if config_loaded():
        ids = []
        for name in config_provider_names():
            p = config_provider(name)
            m = provider_model(p) if p else None
            if m:
                ids.append(m)
        return ids
    return []


def completion_candidates(word):
    """Readline completes the word under the cursor rather than the whole
    line, so the command set and the argument sets are merged and filtered by
    prefix. Tab in the middle of prose only reacts to words that actually
    begin a known name (command, provider profile, engine, or model id)."""
    matching = lambda xs: [x for x in xs if x.startswith(word)]  # noqa: E731
    if word.startswith("/"):
        return matching(SLASH_COMMANDS)
    return (matching(config_provider_names())
            + matching(SEARCH_ENGINES)
            + matching(configured_model_ids()))


def setup_line_input():
    """Enable readline editing, persisted history, and Tab completion when
    stdin is a terminal and the readline backend is present. Returns True
    when active."""
    if not line_input.line_input_available():
        return False
    line_input.set_history_file(line_input.default_history_file())
    line_input.install_completer(completion_candidates)
    return True


# ---------------------------------------------------------------------------
# Main REPL + CLI dispatch

def run_repl():
    if not config_loaded():
        hc.load_harness_config()
        apply_harness_flags()
    register_all()
    reset_conversation()
    setup_line_input()
    print_banner()
    while True:
        line = line_input.read_input_line("\n> ")
        if line is None:
            line_input.save_history()
            print("")
            return
        trimmed = line.strip()
        if trimmed == "":
            continue
        cmd = handle_slash_command(trimmed)
        if cmd == "quit":
            line_input.save_history()
            return
        if cmd == "continue":
            continue
        try:
            send_to_model(trimmed)
        except Exception as e:  # noqa: BLE001
            print("\nError talking to model: {}".format(e))
            sys.stdout.flush()


def run():
    """Back-compat alias."""
    run_repl()


# ---------------------------------------------------------------------------
# CLI entry

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="AI pair programmer in your terminal: multi-turn agentic "
                    "coding loop with approved colored diffs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--prompt", action="append", default=[],
                        metavar="TEXT", help="prompt text, repeatable, joined with newlines")
    parser.add_argument("--stdin", action="store_true",
                        help="read prompt from stdin (pipe/heredoc)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="auto-approve propose_edit after showing diff")
    parser.add_argument("--dry-run", action="store_true",
                        help="show diffs but do not write files")
    parser.add_argument("--provider", metavar="NAME",
                        help="provider profile from the harness config (e.g. fireworks, mlx)")
    parser.add_argument("--model", metavar="ID",
                        help="override model for this provider")
    parser.add_argument("--cwd", metavar="DIR",
                        help="chdir before running")
    parser.add_argument("--debug", action="store_true",
                        help="enable debug logging (same as /debug)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="no banner, no [intent] line, less tool chatter")
    parser.add_argument("--plain", "--no-color", dest="plain", action="store_true",
                        help="no ANSI colors in diffs")
    parser.add_argument("-v", "--version", action="store_true",
                        help="show version and exit")
    parser.add_argument("positional", nargs="*", metavar="PROMPT",
                        help="one-shot prompt text")
    return parser


def cli_main(argv=None):
    global cli_quiet, cli_plain
    # 1a) harness JSON config: ~/.coding_harness.json + .local_coding_harness.json
    hc.load_harness_config()
    apply_harness_flags()
    # 1b) env overrides (harness JSON takes precedence for provider unless
    #     CODING_AGENT_PROVIDER/AGENT_PROVIDER names a known profile)
    apply_env_overrides()
    # 2) parse CLI; if a prompt is present, run one-shot, else REPL
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # --version early exit
    if args.version:
        print("coding-agent {}".format(VERSION))
        exit_with_code(EXIT_OK)
    # Provider data lives only in the harness config, so refuse to run with none
    # rather than falling back to compiled-in defaults.
    if not config_provider_names():
        sys.stderr.write(
            "error: no providers configured.\n"
            "  Add a \"providers\" section to ~/.coding_harness.json or\n"
            "  .local_coding_harness.json (see README.md for the format).\n")
        exit_with_code(EXIT_BAD_ARGS)
    # Apply CLI overrides (highest precedence)
    if args.debug:
        fireworks_ai.debug_log = True
    if args.quiet:
        cli_quiet = True
        tools.quiet_mode = True
    if args.plain:
        cli_plain = True
        approval.color_enabled = False
    if args.yes:
        tools.auto_approve = True
    if args.dry_run:
        tools.dry_run = True
    if args.provider:
        low = args.provider.strip().lower()
        if config_provider(low):
            switch_provider(low)
        else:
            names = config_provider_names()
            sys.stderr.write("error: unknown provider '{}'{}\n".format(
                args.provider,
                " — available profiles: {}".format(", ".join(names)) if names else " — no providers configured"))
            exit_with_code(EXIT_BAD_ARGS)
    if args.model:
        if args.model.strip() == "":
            sys.stderr.write("error: --model requires a non-empty value\n")
            exit_with_code(EXIT_BAD_ARGS)
        set_current_model(args.model.strip())
    if args.cwd:
        resolve_cwd(args.cwd)
    # Build prompt from all sources
    stdin_text = read_all_stdin() if args.stdin else None
    prompt = build_prompt(args.positional, args.prompt, stdin_text)
    if prompt != "":
        run_one_shot(prompt)
    else:
        run_repl()


if __name__ == "__main__":
    cli_main()
```

### The system prompt is a policy document

`SYSTEM_PROMPT_TEMPLATE` is where the behavioral contract lives. It tells the model to explore before editing, to `read_file` immediately before proposing an edit so the `old` string matches disk, to pass `""` as `old` for new files, to keep diffs small, and to ask rather than retry after a rejection. Compare that text with `tools.py`: every rule in the prompt is backed by enforcement in code. Stale bases are rejected by the tool, whitelisted commands are enforced by `run_shell`. The prompt teaches the model the rules; the tools make breaking them impossible. That pairing is the single most important habit in harness design.

### Intent routing

Not every message belongs in a coding loop. Before calling the model, `send_to_model` classifies the input:

- `general` (weather, movies, "who is"): a stateless call with web search forced on and `GENERAL_SYSTEM_PROMPT`. No coding tools, no conversation history.
- `coding`: the full agentic loop with `ENABLED_TOOLS` over the persistent `messages` history.
- `hybrid`: the coding loop, but with search results prepended when `/search` is on.

`classify_intent` uses a cheap two-stage design: `heuristic_classify` scans the lower-cased input for keyword lists (`GENERAL_KEYWORDS`, `CODING_KEYWORDS`). Only when neither list matches does it spend one LLM call, with `max_tokens=10` and `temperature=0.0`, asking for one word. Errors in that classifier call default to `coding`, so the assistant degrades to useful behavior instead of refusing to act.

### Context management: /context and /compact

An agentic session accumulates messages fast: each tool call adds the assistant message and at least one tool result. `message_char_size` counts characters of content, reasoning, and tool arguments, and `/context` prints a table with one line per message. The token estimate divides characters by 4, the standard heuristic `\hat{t} \approx c / 4`$ for English text and code.

`/compact` is the interesting one. It serializes the whole conversation into a transcript, sends it with `COMPACT_SYSTEM_PROMPT` asking for a dense summary, and then replaces everything except the original system prompt with the summary as a single user message. A 40,000-character session becomes a 2,000-character brief. The model keeps working from the summary: goals, decisions, files touched, and open tasks survive; the giant file dumps do not.

### Slash commands, skills, and Tab completion

`handle_slash_command` dispatches `/reset`, `/history`, `/context`, `/compact`, `/model`, `/provider`, `/debug`, `/search`, `/tokens`, `/help`, `/skills`, and `/quit`. Any *other* `/name` is treated as a skill lookup under `~/.agents/skills/<name>/SKILL.md`: the file is read, injected as a system message, and the model is asked to acknowledge it. This is the same skill mechanism the bigger agents use, in about sixty lines.

`completion_candidates` powers Tab completion. Readline completes the word under the cursor, so the function merges slash commands (when the word starts with `/`) with provider profile names, search engines, and configured model IDs, filtered by prefix.

### One-shot mode and exit codes

With a prompt (from `-p`, positional words, or `--stdin`), the agent runs one task and exits. `infer_exit_code` scans the tool results for the phrases `"make check failed"`, `"user rejected"`, and `"user skipped"` and maps them to distinct exit codes:

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | model/API error |
| 2 | edit applied but `make check` failed |
| 3 | user rejected or skipped the change |
| 5 | bad arguments or no providers configured |

This turns the agent into a Unix citizen. A CI script can run `coding-agent -p "fix the lint errors"; test $? -eq 0` and know not just that the agent finished, but whether the result compiles.

## Line Editing: line_input.py

A REPL without cursor keys feels broken. Python ships a `readline` module, but there are two failure modes: the import can be absent, and stdin may not be a terminal (piped input, `--stdin`, CI). This module handles both, degrading to a plain hand-printed prompt.

```python
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

def install_completer(candidates_fn):
    """candidates_fn: str -> list of completion strings. Ignored when the
    backend is unavailable."""
    if not line_input_available():
        return

    def completer(text, state):
        # Pure string work: no I/O, threads, or subprocesses (libedit calls
        # this while the interpreter is effectively in atomic mode).
        try:
            matches = candidates_fn(text) or []
        except Exception:
            matches = []
        if state < len(matches):
            return matches[state]
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
    """-> str, or None on EOF/KeyboardInterrupt. Always prints the prompt,
    with readline handling editing on a terminal and a plain read-line used
    otherwise."""
    if line_input_available():
        try:
            line = input(prompt)
            readline.add_history(line)
            return line
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
    except Exception:
        return None
    if line == "":  # EOF
        return None
    return line.rstrip("\n")
```

Two portability details. On macOS the `readline` module is backed by *libedit*, not GNU Readline, and libedit uses different bind syntax, which is why `install_completer` checks `"libedit" in readline.__doc__` and issues `bind ^I rl_complete` there instead of `tab: complete`. And history saving writes to a temporary file and calls `os.replace`, which is atomic: a crash mid-save cannot leave you with a half-written history. The completer wrapper keeps the callback free of I/O and subprocesses, because libedit calls it while the interpreter is effectively in atomic mode.

## Running the Agent

The project is managed by uv. `pyproject.toml` declares one runtime dependency, `requests`, and a console script `coding-agent` pointing at `agent:cli_main`.

```
cd source-code/coding-harness-agent
uv sync                       # creates .venv, installs requests
export FIREWORKS_API_KEY=your-key-here
uv run agent.py               # interactive REPL
```

The Makefile wraps the usual work:

```
make run          # uv run agent.py
make sync         # create/update .venv
make check        # byte-compile all sources
make install      # uv tool install .  -> global coding-agent command
make build        # uv build -> sdist + wheel in dist/
make clean
```

Try the modes, in order:

```
coding-agent --provider mlx -q -p "add a docstring to normalize in search.py"
echo "what is the capital of Norway" | coding-agent --stdin --provider fireworks
coding-agent --dry-run --debug -p "refactor grep tool to exclude binary files"
```

### A REPL Session, Start to Finish

Below is a transcript of a real session shape, with a Fireworks provider configured as shown earlier. User input begins after the `>` prompt.

```
Coding Agent REPL.  /help for commands, /quit to exit.
  cwd:      /Users/markw/GITHUB/demo/normalize-util
  provider: fireworks
  model:    accounts/fireworks/models/deepseek-v4p1-flash

> add a docstring to normalize in search.py and run make check

[intent: coding → coding tools, no search]

I'll read the file first, then propose the edit.
* read_file search.py
* propose_edit {"path": "search.py", "old": "def normalize(s):\n    return s.strip().l…

--- a/search.py
+++ b/search.py
@@ -1,4 +1,7 @@
 import os

 def normalize(s):
+    """Lower-case, whitespace-stripped form of s.
+
+    Used as the comparison key for cache lookups.
+    """
     return s.strip().lower()

Apply this change? [y]es / [n]o / [s]kip and tell the model why: y
make check passed. Added a three-line docstring to normalize in search.py.

> /tokens

Session token usage:
  Prompt tokens:     18432
  Completion tokens: 921
  Total tokens:      19353
  Cached tokens:     12288 (66.7% of prompt)
  Estimated cost:    $0.001462  ($0.1400/M input, $0.0280/M cached input, $0.2800/M output)

> /quit
```

Reading the transcript against the code: `[intent: coding ...]` is `send_to_model` reporting the classifier. The `* read_file search.py` lines come from `execute_tool_calls` in quiet-off mode. The assistant text printed *before* each `*` line is `content` from the same message that carried the tool calls, which the loop prints before executing. The diff and the y/n/s prompt are `approval.py`. The final sentence is the last iteration's content, the one that carried no tool calls.

### Reading /context

After a few exchanges, `/context` prints the message table:

```
Context: 5 messages, 15234 chars, 3808 tokens (est.)

   #  role       chars  preview
  ---  ---------  -------  --------------------------------------------------
   1  system       712  You are an interactive coding assistant working in t…
   2  user         184  add a docstring to normalize in search.py and run ma…
   3  assistant   9840  [tool calls: read_file, propose_edit]
   4  tool       4412  [read_file] import os\n\ndef normalize(s):\n    retur…
   5  assistant    86  make check passed. Added a three-line docstring to no…
```

The table immediately shows where your tokens go. Message 4 is a whole file echo. The system prompt (712 characters, resent every turn) is *why prompt caching pays off here*: it is a fixed prefix, so the server bills it at the cached rate.

### One-shot and piped output

One-shot mode is REPL output minus the banner, plus an exit code:

```
$ coding-agent -q --dry-run -p "rename normalize to norm_key in search.py"
[dry-run only] I would change the function name in search.py and update its
two call sites...
$ echo $?
0
```

Pipe input with `--stdin` to feed the agent command output:

```
$ git diff origin/main | coding-agent --stdin --provider fireworks \
    -p "summarize these changes for the release notes"
```

If the user rejects an edit in a one-shot run, the exit code is 3 and the phrase `[change rejected or skipped]` prints, unless `-q` suppressed it.

## Wrap Up

This chapter built a complete coding agent in nine small Python modules. The finished harness does four things that generalize to every agent you will ever build or use:

1. **The loop is separate from the transport.** `chat_loop.py` holds all agent intelligence and knows nothing about HTTP or providers. Swapping Fireworks for MLX was a `post_fn` argument, not an edit. If you want to add Anthropic- or Google-style clients later, you write one function, not a new loop.
2. **Behavior lives in two places: the prompt and the tools.** Instructions tell the model what to do; code guarantees what may happen. Whitelisted commands, hidden-file filters, stale-base rejection, and the approval gate mean the model cannot damage your project even if it misreads the instructions. Whenever you write a rule in a system prompt, ask which tool enforces it.
3. **Failures of small models are design inputs, not exceptions.** Malformed JSON, null content, truncated tool calls, and repetition loops all have explicit handling that turns them into feedback the model can act on. Stuck-model detection with signatures bounds the cost of confusion.
4. **Configuration is data.** Endpoints, models, keys-by-name, generation parameters, and pricing all live in the global and local JSON files, merged with a twelve-line `deep_merge`. Unknown values are reported as unknown. New providers need no code change, which is exactly the test of whether you separated policy from mechanism.

You now understand the core architecture of the coding agents in this book at a level that a user of those agents does not reach by reading their documentation. The agentic loop, the approval gate, and the hierarchical config are all reusable in your own projects, from CI bots to home-lab automation.

## Exercises for the Reader

1. **Add a `write_file` tool.** Register a tool taking `(path, contents)` that creates a new file. Reuse the approval flow from `propose_edit`. Make sure `""` remains a valid argument and hidden paths are refused. Test it by asking the agent to generate a `.gitignore`.
2. **Implement an Ollama provider profile** in your global config (`type: "ollama"`, endpoint on port 11434). Confirm that no code changes are needed and that `/provider` lists the new profile.
3. **Widen the repetition window.** Change `REPEAT_WINDOW` to 10 and `REPEAT_LIMIT` to 3. Run an intentionally weak small model on a task that fails, and compare the token usage (`/tokens`) with the default settings. What is the tradeoff?
4. **Make `/compact` budget-driven.** Add an auto-compaction trigger: when the total of `message_char_size` exceeds a threshold from the config file (say `"auto_compact_chars": 50000`), run compaction automatically before the next model call.
5. **Add a third search engine** (for example Tavily) to `search.py`, return the same `(url, title, snippet)` triples, and wire `/search tavily` plus config support for the new engine name.
6. **Estimate cost without pricing.** `session_cost` currently returns `None` when a profile declares no `pricing`. Change `print_session_stats` to also show the estimated token count of the *next* request using `c / 4`$ so readers can see context growth even for unpriced profiles.
7. **Harden the shell whitelist.** `run_shell` currently allows `cat`, which can read any non-dotfile by absolute path. Add a check that rejects arguments containing `..` or an absolute path outside the working directory, and return the reason as the tool result.
8. **Port the loop to streaming for MLX.** `mlx_lm.server` also supports SSE. Reuse `parse_sse_response` from `fireworks_ai.py` to stream local responses, and print tokens as they arrive.
