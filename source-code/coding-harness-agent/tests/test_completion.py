# test_completion.py -- smoke tests for Tab completion.
#
# These exist because of a real bug. On macOS, Python's readline module is
# libedit, and libedit splits the word differently than GNU Readline: it treats
# a leading "/" as a word separator, so completing "/t" hands the completer the
# word "t". A completer that only understood "/t" returned nothing, libedit
# fell back to filename completion, and it inserted a file name in the middle
# of the command -- producing lines like "/ttokens". Completion must therefore
# accept both spellings: the full word and the separator-stripped one.
#
# Runs offline: no readline backend, no terminal, no network.

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import agent  # noqa: E402
import harness_config as hc  # noqa: E402
import line_input  # noqa: E402


class TestCompletionCandidates(unittest.TestCase):
    def setUp(self):
        hc.harness_config = {
            "default_provider": "fireworks",
            "providers": {
                "fireworks": {"type": "openai", "model": "accounts/fireworks/models/deepseek-v4p1-flash"},
                "mlx": {"type": "mlx", "model": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"},
            },
        }
        hc._active_provider_name = None

    def tearDown(self):
        hc.harness_config = {}
        hc._active_provider_name = None

    def test_full_word_from_gnu_readline(self):
        # GNU Readline keeps the slash: it passes "/t" and inserts the result.
        self.assertEqual(agent.completion_candidates("/t"), ["/tokens"])

    def test_stripped_word_from_libedit_gives_the_suffix_only(self):
        # libedit passes "t"; the slash is already on the line, so the
        # candidate must NOT repeat it or the line becomes "/ttokens".
        self.assertEqual(agent.completion_candidates("t"), ["tokens"])

    def test_every_command_is_reachable_from_its_first_letter(self):
        for command in agent.SLASH_COMMANDS:
            letter = command[1]
            full = agent.completion_candidates("/" + letter)
            stripped = agent.completion_candidates(letter)
            self.assertIn(command, full, "with slash: {}".format(command))
            self.assertIn(command[1:], stripped, "separator-stripped: {}".format(command))

    def test_empty_slash_lists_every_command(self):
        self.assertEqual(agent.completion_candidates("/"), list(agent.SLASH_COMMANDS))

    def test_non_command_words_do_not_offer_slash_commands(self):
        # "fireworks" must complete to the profile, not to "/fireworks".
        self.assertNotIn("/fireworks", agent.completion_candidates("fire"))
        self.assertIn("fireworks", agent.completion_candidates("fire"))

    def test_ambiguous_prefix_merges_the_sets_without_duplicates(self):
        got = agent.completion_candidates("m")
        self.assertEqual(sorted(got), sorted(set(got)))
        self.assertIn("model", got)        # from "/model"
        self.assertIn("mlx", got)          # provider profile
        self.assertIn("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", got)

    def test_unknown_prefix_is_empty(self):
        self.assertEqual(agent.completion_candidates("/zzz"), [])
        self.assertEqual(agent.completion_candidates("zzz"), [])


class TestReadlineCompleterContract(unittest.TestCase):
    """Exercise line_input's completer directly, without a readline backend."""

    def _completer_for(self, candidates_fn):
        captured = {}

        class FakeReadline:
            __doc__ = "Importing this module enables command line editing using GNU readline."

            def set_completer(self, fn):
                captured["completer"] = fn

            def set_completer_delims(self, delims):
                pass

            def parse_and_bind(self, line):
                pass

        real = line_input.readline
        try:
            line_input.readline = FakeReadline()
            line_input.line_input_available = lambda: True
            line_input.install_completer(candidates_fn)
        finally:
            line_input.readline = real
        return captured["completer"]

    def test_state_walks_the_match_list_then_stops(self):
        completer = self._completer_for(lambda word: ["/aa", "/ab"])
        self.assertEqual(completer("/a", 0), "/aa")
        self.assertEqual(completer("/a", 1), "/ab")
        self.assertIsNone(completer("/a", 2))

    def test_candidates_that_do_not_extend_the_word_are_dropped(self):
        # "/tokens" can never be inserted over the word "/x"; offering it would
        # leave stray text, which is the class of bug we are guarding against.
        completer = self._completer_for(lambda word: ["/tokens", "/xray"])
        self.assertEqual(completer("/x", 0), "/xray")
        self.assertIsNone(completer("/x", 1))

    def test_duplicate_candidates_are_collapsed(self):
        completer = self._completer_for(lambda word: ["/aa", "/aa"])
        self.assertEqual(completer("/a", 0), "/aa")
        self.assertIsNone(completer("/a", 1))

    def test_a_raising_candidates_function_yields_no_completions(self):
        def boom(word):
            raise RuntimeError("no")
        completer = self._completer_for(boom)
        self.assertIsNone(completer("/a", 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
