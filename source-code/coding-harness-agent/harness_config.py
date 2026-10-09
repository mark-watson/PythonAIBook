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
LOCAL_CONFIG_BASENAME = ".local_coding_harness.json"

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


def local_config_path():
    """The local override file, always looked up in the *current* directory
    (--cwd may have changed it since this module was imported)."""
    return os.path.join(os.getcwd(), LOCAL_CONFIG_BASENAME)


def load_harness_config():
    """Load global then local, deep-merge, store, and return the result."""
    global harness_config, _active_provider_name
    global_cfg = _read_json_file(GLOBAL_CONFIG_PATH) or {}
    local_cfg = _read_json_file(local_config_path()) or {}
    harness_config = deep_merge(global_cfg, local_cfg)
    _active_provider_name = None  # re-resolve the default profile
    return harness_config


# ---------------------------------------------------------------------------
# Providers

def config_providers():
    p = harness_config.get("providers", {})
    return p if isinstance(p, dict) else {}


def provider_names(cfg):
    """-> sorted profile names declared in `cfg` (the caller's dict, not the
    module global, so loading and defaulting can be reasoned about separately)."""
    p = cfg.get("providers") if isinstance(cfg, dict) else None
    return sorted(p.keys()) if isinstance(p, dict) else []


def config_provider_names():
    return provider_names(harness_config)


def config_provider(name):
    """-> provider dict for profile `name`, or None."""
    if name is None:
        return None
    return config_providers().get(str(name))


def _pick_default_provider_name(cfg):
    """-> name of the profile to start with, or None.

    The declared "default_provider" wins; otherwise "fireworks" if present,
    otherwise the first name alphabetically.
    """
    names = provider_names(cfg)
    declared = cfg.get("default_provider") if isinstance(cfg, dict) else None
    if isinstance(declared, str) and declared in names:
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
    def loaded(path):
        return "(loaded)" if os.path.isfile(path) else "(absent)"
    print("Config files: {} {} / {} {}".format(
        GLOBAL_CONFIG_PATH, loaded(GLOBAL_CONFIG_PATH),
        local_config_path(), loaded(local_config_path())))
    print("Providers:    {}".format(", ".join(config_provider_names())))
    print("Active:       {}".format(config_active_provider_name() or "(defaults)"))
