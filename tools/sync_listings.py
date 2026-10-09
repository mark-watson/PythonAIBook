#!/usr/bin/env python3
# sync_listings.py -- keep the fenced source listings in a chapter in sync with
# the modules they were copied from.
#
# The book prints whole modules. That is only honest if the printed listing is
# the file the reader can also run, so this script owns the copy step:
#
#   ./tools/sync_listings.py --check    # report drift, exit 1 if any
#   ./tools/sync_listings.py --write    # rewrite the listings from the sources
#
# How a listing is identified
# ---------------------------
# Every listing in the chapter starts with the module's own header comment,
# e.g. "# tools.py -- tool registry ..." or "# agent.py -- main REPL loop ...".
# This script reads that first line and looks for the matching source file in
# the example directory. Nothing is positional, so reordering sections or
# inserting a new module does not silently pair the wrong code with the wrong
# prose: an unmatched listing is an error, not a guess.
#
# Every listing in the chapter must be matched, and every module in
# MODULE_ORDER must appear exactly once. That keeps both directions honest: a
# chapter can neither print stale code nor quietly drop a module.
#
# Usage:
#   python tools/sync_listings.py                       # check the default pair
#   python tools/sync_listings.py --check
#   python tools/sync_listings.py --write
#   python tools/sync_listings.py --manuscript X.md --source-dir Y --write

import argparse
import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_MANUSCRIPT = os.path.join(ROOT, "manuscript", "coding-harness-agent.md")
DEFAULT_SOURCE_DIR = os.path.join(ROOT, "source-code", "coding-harness-agent")

# The chapter introduces the modules in this order. Each name must appear as a
# python listing exactly once.
MODULE_ORDER = [
    "harness_config.py",
    "tools.py",
    "approval.py",
    "chat_loop.py",
    "fireworks_ai.py",
    "mlx_serve.py",
    "search.py",
    "agent.py",
    "line_input.py",
]

# Every listing starts with a header comment naming either the module file
# ("# tools.py -- ...") or an older internal name ("# coding_harness -- ...").
HEADER_RE = re.compile(r"^#\s+([A-Za-z_][A-Za-z0-9_.]*)\s+--")

# Older internal names that still appear in a header line.
MODULE_ALIASES = {
    "coding_harness": "harness_config.py",
}


def normalize(text):
    """Compare and write listings the same way: strip trailing blank lines."""
    return text.rstrip("\n")


def find_listings(lines):
    """-> list of (start, end, body) for every fenced block, 0-based, exclusive end."""
    listings = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("```"):
            lang = lines[i][3:].strip()
            start = i + 1
            j = start
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j >= len(lines):
                raise SystemExit("error: unterminated code fence at line {}".format(i + 1))
            listings.append((start, j, lang, "\n".join(lines[start:j])))
            i = j + 1
        else:
            i += 1
    return listings


def module_for_listing(body, source_dir):
    """The source file a listing came from, identified by its header comment."""
    first = body.split("\n", 1)[0]
    m = HEADER_RE.match(first)
    if not m:
        return None, first
    name = m.group(1)
    if not name.endswith(".py"):
        name += ".py"
    name = MODULE_ALIASES.get(name[:-3], name)
    return (name if os.path.isfile(os.path.join(source_dir, name)) else None), first


def read_source(source_dir, name):
    with open(os.path.join(source_dir, name), "r", encoding="utf-8") as f:
        return normalize(f.read())


def check_or_write(manuscript, source_dir, write):
    with open(manuscript, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    listings = find_listings(lines)
    python_listings = [l for l in listings if l[2] == "python"]

    matched = {}          # module name -> listing index
    problems = []
    for idx, (start, end, _lang, body) in enumerate(python_listings):
        name, first = module_for_listing(body, source_dir)
        if name is None:
            problems.append("listing at line {}: cannot identify a module from {!r}"
                            .format(start + 1, first[:60]))
            continue
        if name in matched:
            problems.append("listing at line {}: {} appears more than once"
                            .format(start + 1, name))
            continue
        matched[name] = idx

    for name in MODULE_ORDER:
        if name not in matched:
            problems.append("no listing found for {}".format(name))
    for name in matched:
        if name not in MODULE_ORDER:
            problems.append("{} is listed in the chapter but not in MODULE_ORDER".format(name))
    if problems:
        for p in problems:
            sys.stderr.write("error: {}\n".format(p))
        return 1

    stale = []
    rewrites = []
    for name in MODULE_ORDER:
        start, end, _lang, body = python_listings[matched[name]]
        source = read_source(source_dir, name)
        if normalize(body) == source:
            continue
        stale.append(name)
        rewrites.append((start, end, source))

    if not stale:
        sys.stdout.write("listings up to date: {} modules match {}\n".format(
            len(MODULE_ORDER), os.path.relpath(source_dir, ROOT)))
        return 0

    if not write:
        sys.stderr.write("drift detected in {} listing(s):\n".format(len(stale)))
        for name in stale:
            start, end, _lang, body = python_listings[matched[name]]
            source = read_source(source_dir, name)
            diff = difflib.unified_diff(normalize(body).split("\n"),
                                        source.split("\n"),
                                        "manuscript:{}".format(name),
                                        "source:{}".format(name),
                                        lineterm="", n=1)
            for line in list(diff)[:12]:
                sys.stderr.write("  {}\n".format(line))
        sys.stderr.write("rerun with --write to update\n")
        return 1

    for start, end, source in sorted(rewrites, reverse=True):
        lines[start:end] = source.split("\n")
    with open(manuscript, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    sys.stdout.write("rewrote {} listing(s) in {}\n".format(
        len(stale), os.path.relpath(manuscript, ROOT)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sync the chapter's fenced Python listings with the source modules.")
    parser.add_argument("--manuscript", default=DEFAULT_MANUSCRIPT,
                        help="chapter markdown file (default: %(default)s)")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR,
                        help="directory holding the modules (default: %(default)s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true",
                       help="report drift and exit 1 (the default)")
    group.add_argument("--write", action="store_true",
                       help="rewrite the listings from the source files")
    args = parser.parse_args(argv)
    return check_or_write(args.manuscript, args.source_dir, args.write)


if __name__ == "__main__":
    sys.exit(main())
