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
