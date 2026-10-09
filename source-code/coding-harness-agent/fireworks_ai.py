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
