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
