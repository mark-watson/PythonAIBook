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
