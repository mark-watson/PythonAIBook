# test_agent.py -- smoke tests for CLI dispatch, intent routing, and the
# one-shot exit codes.
#
# The network and the terminal are both stubbed: `post_fireworks` is replaced
# by a fake model, and stdin by a StringIO. That keeps the whole CLI path --
# config loading, provider selection, tool execution, exit code -- testable
# offline.

import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import agent  # noqa: E402
import fireworks_ai  # noqa: E402
import harness_config as hc  # noqa: E402
import tools  # noqa: E402


def assistant(content="", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}


def call(name, arguments, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class FakeModel:
    """A stub transport: scripted assistant replies plus the intent classifier.

    `classify_intent` falls back to a one-word LLM call for queries the
    heuristics do not recognize, and those calls would otherwise eat a scripted
    reply. Answering them here keeps every test's response list exactly as long
    as the list of tool-calling turns it means to script.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.classifier_calls = 0

    def __call__(self, payload):
        text = json.dumps(payload.get("messages", []))
        if "one-word query classifier" in text:
            self.classifier_calls += 1
            return assistant("CODING")
        if not self.responses:
            raise AssertionError("fake model ran out of responses")
        return self.responses.pop(0)


class FakeStdin(io.StringIO):
    """A text stream that reports itself as a pipe, like `git diff | agent`."""

    def isatty(self):
        return False


class AgentTestCase(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._old_argv = list(sys.argv)
        self._old_stdin = sys.stdin
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        hc.harness_config = {
            "default_provider": "fake",
            "providers": {"fake": {"type": "openai", "endpoint": "http://x",
                                   "model": "fake-model", "generation": {}}},
        }
        hc._active_provider_name = None
        # cli_main loads the real ~/.coding_harness.json; in a test we want the
        # fixture config only, so the loader becomes a no-op.
        self._real_loader = hc.load_harness_config
        hc.load_harness_config = lambda: hc.harness_config
        agent.cli_quiet = True
        agent.cli_plain = True
        agent.set_current_model(None)
        tools.register_all()
        tools.auto_approve = True
        tools.dry_run = False
        tools.quiet_mode = True
        self.write("Makefile", "check:\n\t@true\n")

    def tearDown(self):
        sys.argv = self._old_argv
        sys.stdin = self._old_stdin
        hc.load_harness_config = self._real_loader
        hc.harness_config = {}
        hc._active_provider_name = None
        os.chdir(self._old_cwd)
        self.tmp.cleanup()

    def write(self, name, text):
        with open(name, "w", encoding="utf-8") as f:
            f.write(text)

    def stub(self, responses):
        transport = FakeModel(responses)
        fireworks_ai.post_fireworks = transport
        return transport

    def feed_stdin(self, text):
        sys.stdin = FakeStdin(text)


class TestIntent(AgentTestCase):
    def test_general_keyword_is_classified_without_the_model(self):
        self.assertEqual(agent.heuristic_classify("what is the capital of norway"), "general")

    def test_coding_keyword(self):
        self.assertEqual(agent.heuristic_classify("refactor normalize in search.py"), "coding")

    def test_unmatched_query_falls_through_to_the_model(self):
        self.assertIsNone(agent.heuristic_classify("tell me about the universe"))


class TestOneShot(AgentTestCase):
    def test_plain_reply_exits_zero(self):
        self.stub([assistant("all done")])
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["-p", "refactor normalize"])
        self.assertEqual(ctx.exception.code, agent.EXIT_OK)
        self.assertEqual(agent.messages[-1]["content"], "all done")

    def test_rejected_edit_exits_three(self):
        # Auto-approve is off, so the approval prompt reads from stdin and the
        # human answers "n" to both the first attempt and the one retry.
        tools.auto_approve = False
        self.write("mod.py", "x = 1\n")
        self.stub([
            assistant("editing", [call("replace_in_file",
                                       '{"path": "mod.py", "old_string": "x = 1", "new_string": "x = 2"}')]),
            assistant("I see, stopping"),
            assistant("smaller change", [call("replace_in_file",
                                              '{"path": "mod.py", "old_string": "x = 1", "new_string": "x = 3"}')]),
            assistant("ok"),
        ])
        self.feed_stdin("n\nn\n")
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["-p", "change x"])
        self.assertEqual(ctx.exception.code, agent.EXIT_REJECTED)
        self.assertEqual(open("mod.py").read(), "x = 1\n")

    def test_failed_check_exits_two(self):
        self.write("Makefile", "check:\n\t@exit 1\n")
        self.stub([
            assistant("editing", [call("replace_in_file",
                                       '{"path": "Makefile", "old_string": "@exit 1", "new_string": "@exit 1 # x"}')]),
            assistant("done"),
        ])
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["-p", "touch the Makefile"])
        self.assertEqual(ctx.exception.code, agent.EXIT_CHECK_FAILED)

    def test_model_error_exits_one(self):
        def boom(payload):
            raise RuntimeError("no route to host")
        fireworks_ai.post_fireworks = boom
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["-p", "refactor normalize"])
        self.assertEqual(ctx.exception.code, agent.EXIT_MODEL_ERROR)

    def test_empty_stdin_with_flag_is_a_usage_error(self):
        self.feed_stdin("")
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["--stdin"])
        self.assertEqual(ctx.exception.code, agent.EXIT_BAD_ARGS)

    def test_piped_stdin_without_the_flag_is_still_a_prompt(self):
        self.stub([assistant("summarized")])
        self.feed_stdin("a diff on stdin")
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["-p", "summarize this diff"])
        self.assertEqual(ctx.exception.code, agent.EXIT_OK)

    def test_unknown_provider_exits_with_bad_args(self):
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["--provider", "nope", "-p", "hi"])
        self.assertEqual(ctx.exception.code, agent.EXIT_BAD_ARGS)

    def test_version_prints_and_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            agent.cli_main(["--version"])
        self.assertEqual(ctx.exception.code, agent.EXIT_OK)


class TestSlashCommands(AgentTestCase):
    def test_unknown_command_tries_a_skill_then_returns_continue(self):
        agent.reset_conversation()
        self.assertEqual(agent.handle_slash_command("/no-such-skill"), "continue")

    def test_provider_switch_reports_the_new_model(self):
        hc.harness_config["providers"]["other"] = {"type": "openai", "model": "other-model"}
        agent.reset_conversation()
        agent.handle_slash_command("/provider other")
        self.assertEqual(agent.current_model_id(), "other-model")

    def test_tokens_reset_zeroes_the_counters(self):
        fireworks_ai.accumulate_usage({"usage": {"prompt_tokens": 10,
                                                 "completion_tokens": 5,
                                                 "total_tokens": 15}})
        agent.reset_conversation()
        agent.handle_slash_command("/tokens reset")
        self.assertEqual(fireworks_ai.session_snapshot(), (0, 0, 0, 0))

    def test_compact_keeps_the_system_prompt(self):
        self.stub([assistant("a dense summary")])
        agent.reset_conversation()
        agent.messages.append({"role": "user", "content": "do the thing"})
        agent.messages.append(assistant("working", [call("list_dir", '{"path": "."}')]))
        agent.messages.append({"role": "tool", "name": "list_dir",
                               "content": "Makefile", "tool_call_id": "c1"})
        agent.compact_context()
        self.assertEqual(len(agent.messages), 2)
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertIn("a dense summary", agent.messages[1]["content"])

    def test_general_question_leaves_the_coding_transcript_alone(self):
        self.stub([assistant("Oslo")])
        agent.reset_conversation()
        before = list(agent.messages)
        agent.send_to_model("what is the capital of norway")
        self.assertEqual(agent.messages, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
