# test_chat_loop.py -- smoke tests for the provider-agnostic agentic loop.
#
# The loop talks to a `post_fn` callback instead of a network socket, so a
# fake model is just a list of canned responses. That seam is the point of the
# design, and these tests prove it: repetition detection, malformed tool
# calls, missing ids, and max-iteration summaries are all exercised offline.
#
# Run from the project root:  uv run pytest -q

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat_loop  # noqa: E402
import harness_config as hc  # noqa: E402
import tools  # noqa: E402


def assistant(content="", tool_calls=None, **extra):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    msg.update(extra)
    return {"choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def call(name, arguments, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class FakeModel:
    """Returns canned responses in order and records every payload it sees."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("fake model ran out of responses")
        return self.responses.pop(0)


TOOLS = ["list_dir"]


class ChatLoopTestCase(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self.tmp = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.chdir(self.tmp)
        tools.register_all()
        tools.quiet_mode = True


class TestPlainChat(ChatLoopTestCase):
    def test_returns_content(self):
        model = FakeModel([assistant("hello")])
        self.assertEqual(chat_loop.chat(model, [{"role": "user", "content": "hi"}],
                                        "m", max_tokens=None, temperature=None), "hello")

    def test_null_content_falls_back(self):
        model = FakeModel([{"choices": [{"message": {"role": "assistant", "content": None}}]}])
        self.assertEqual(chat_loop.chat(model, [], "m"), "No response content")

    def test_undeclared_generation_params_are_omitted(self):
        model = FakeModel([assistant("hello")])
        chat_loop.chat(model, [], "m")
        self.assertNotIn("max_tokens", model.payloads[0])
        self.assertNotIn("temperature", model.payloads[0])


class TestToolLoop(ChatLoopTestCase):
    def test_executes_tool_calls_and_returns_final_text(self):
        model = FakeModel([
            assistant("looking", [call("list_dir", '{"path": "."}')]),
            assistant("done"),
        ])
        text, messages = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertEqual(text, "done")
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["tool_call_id"], "c1")

    def test_missing_tool_call_id_is_filled_in_before_the_reply(self):
        model = FakeModel([
            assistant("looking", [{"type": "function",
                                   "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]),
            assistant("done"),
        ])
        _, messages = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_0")
        self.assertEqual(messages[1]["tool_call_id"], "call_0")

    def test_repeated_batch_stops_after_the_second_attempt(self):
        stuck = assistant("trying", [call("list_dir", '{"path": "."}')])
        model = FakeModel([stuck, stuck, stuck, stuck])
        text, messages = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertIn("repeated the identical tool call", text)
        self.assertEqual(len(model.payloads), 2)
        # Every assistant tool_calls message still has a matching tool reply,
        # so the transcript stays valid for a strict server.
        calls = sum(len(m.get("tool_calls") or []) for m in messages)
        replies = sum(1 for m in messages if m["role"] == "tool")
        self.assertEqual(calls, replies)
        self.assertEqual(replies, 2)

    def test_a_changed_batch_is_not_treated_as_repetition(self):
        model = FakeModel([
            assistant("one", [call("list_dir", '{"path": "."}')]),
            assistant("two", [call("list_dir", '{"path": "tests"}')]),
            assistant("done"),
        ])
        text, _ = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertEqual(text, "done")

    def test_bad_json_arguments_become_tool_feedback(self):
        model = FakeModel([
            assistant("", [call("list_dir", "{oops")]),
            assistant("done"),
        ])
        _, messages = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertIn("invalid JSON", messages[1]["content"])

    def test_max_iterations_asks_for_a_summary_without_tools(self):
        # Two distinct batches, so the repetition guard stays quiet and the
        # iteration budget is what runs out.
        model = FakeModel([
            assistant("still working", [call("list_dir", '{"path": "."}')]),
            assistant("still working", [call("list_dir", '{"path": "tests"}')]),
            assistant("summary"),
        ])
        text, _ = chat_loop.chat_with_tools(model, [], TOOLS, "m", max_iterations=2)
        self.assertEqual(text, "summary")
        self.assertNotIn("tools", model.payloads[-1])

    def test_empty_assistant_content_with_tool_calls_does_not_stop_the_loop(self):
        model = FakeModel([
            assistant("", [call("list_dir", '{"path": "."}')]),
            assistant("done"),
        ])
        text, _ = chat_loop.chat_with_tools(model, [], TOOLS, "m")
        self.assertEqual(text, "done")


class TestConfig(unittest.TestCase):
    def tearDown(self):
        hc.harness_config = {}

    def test_deep_merge_overrides_only_the_named_keys(self):
        merged = hc.deep_merge(
            {"providers": {"a": {"model": "one", "generation": {"temperature": 0.1}}}},
            {"providers": {"a": {"generation": {"temperature": 0.9}}}})
        self.assertEqual(merged["providers"]["a"]["model"], "one")
        self.assertEqual(merged["providers"]["a"]["generation"]["temperature"], 0.9)

    def test_declared_default_provider_wins(self):
        cfg = {"default_provider": "local", "providers": {"fireworks": {}, "local": {}}}
        self.assertEqual(hc._pick_default_provider_name(cfg), "local")

    def test_fireworks_is_the_fallback(self):
        cfg = {"providers": {"aaa": {}, "fireworks": {}}}
        self.assertEqual(hc._pick_default_provider_name(cfg), "fireworks")

    def test_missing_provider_yields_none(self):
        self.assertIsNone(hc._pick_default_provider_name({}))

    def test_provider_type_aliases_map_to_mlx(self):
        for value in ("mlx", "ollama", "omlx", "sushi", " MLX "):
            self.assertEqual(hc.provider_type({"type": value}), "mlx")
        self.assertEqual(hc.provider_type({}), "openai")
        self.assertEqual(hc.provider_type(None), "openai")

    def test_pricing_ref_reports_unknown_rather_than_zero(self):
        self.assertIsNone(hc.pricing_ref({}, "input"))
        self.assertEqual(hc.pricing_ref({"input": 0.14}, "input"), 0.14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
