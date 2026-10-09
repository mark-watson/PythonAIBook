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
REPEAT_LIMIT = 2    # the same batch seen this many times => stuck

MAX_ITERATIONS_DEFAULT = 20


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


def clean_assistant_message(msg):
    """A copy of an assistant message safe to send back to a strict server.

    Two normalizations happen here. A null or non-string 'content' becomes ""
    (mlx_lm.server sends "" on tool-only responses, the OpenAI spec sends
    null, and replaying the null back can be rejected). A missing tool-call id
    gets a generated one at index 0, 1, ... -- the matching role:"tool" reply
    is addressed by that id, so an empty id breaks the pairing.
    """
    tool_calls = msg.get("tool_calls")
    has_calls = isinstance(tool_calls, list) and bool(tool_calls)
    out = {"role": "assistant", "content": msg_content(msg)}
    if not has_calls:
        return out
    cleaned = []
    for i, tc in enumerate(tool_calls):
        func = tc.get("function") or {}
        call_id = tc.get("id")
        cleaned.append({
            "id": call_id if isinstance(call_id, str) and call_id != ""
                  else "call_{}".format(i),
            "type": tc.get("type") or "function",
            "function": {"name": func.get("name", ""),
                         "arguments": func.get("arguments", "{}")},
        })
    out["tool_calls"] = cleaned
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning != "":
        out["reasoning_content"] = reasoning
    return out


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
                    max_tokens=None, temperature=None,
                    max_iterations=MAX_ITERATIONS_DEFAULT):
    tools_rendered = render_tools(tools)
    current_messages = list(messages)
    recent_signatures = []

    def call_signature(tool_calls):
        sig = []
        for tc in tool_calls:
            f = tc.get("function") or {}
            sig.append("{}|{}".format(f.get("name", ""), f.get("arguments", "")))
        return sorted(sig)

    def remember_and_count(sig):
        """Record this batch and return how often it has now been seen.

        The batch is recorded BEFORE the count, so the second identical batch
        reports 2 and trips REPEAT_LIMIT.
        """
        recent_signatures.append(sig)
        del recent_signatures[:-REPEAT_WINDOW]
        return sum(1 for s in recent_signatures if s == sig)

    def append_tool_results(results):
        for (call_id, name, result_str) in results:
            current_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result_str,
            })

    iter_ = 0
    while True:
        if iter_ >= max_iterations:
            # Max iterations -- one final no-tools call for a summary.
            payload = request_payload(model_id, max_tokens, temperature, current_messages)
            try:
                data = post_fn(payload)
                msg = response_message(data)
                content = msg_content(msg)
                if content == "":
                    content = "(no summary from model)"
                if msg:
                    current_messages.append(clean_assistant_message(msg))
                return content, current_messages
            except Exception:
                return "(max tool iterations reached)", current_messages
        iter_ += 1

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
        assistant_msg = clean_assistant_message(msg)
        current_messages.append(assistant_msg)

        if not tool_calls:
            return (content if content else "(empty response from model)"), current_messages

        # Echo any reasoning the model emitted alongside its tool calls, then
        # run the calls. Results are appended before the stuck-model check so
        # the transcript keeps a role:"tool" reply for every call.
        if content.strip():
            print("")
            print(content.strip())
        batch = call_signature(tool_calls)
        results = execute_tool_calls(tool_calls)
        append_tool_results(results)
        seen = remember_and_count(batch)
        if seen >= REPEAT_LIMIT:
            # The model is stuck re-issuing the identical call(s) -- bail out
            # with an explanation instead of looping to max-iterations.
            return ("(stopped: the model repeated the identical tool call(s) "
                    "{} times without making progress; it may be too weak for "
                    "this task or its arguments are malformed)".format(seen),
                    current_messages)
