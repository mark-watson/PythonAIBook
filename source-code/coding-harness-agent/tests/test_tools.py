# test_tools.py -- smoke tests for the tool layer.
#
# Run from the project root:  uv run pytest -q
# Or without pytest:          uv run python tests/test_tools.py
#
# These tests exercise the safety properties the chapter claims: hidden paths
# are refused, nothing outside the working directory is touched, `ls` output
# keeps filenames containing spaces, and grep never walks into .git.

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402


class ToolTestCase(unittest.TestCase):
    """Every test runs inside a fresh temporary working directory."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        tools.register_all()
        tools.auto_approve = False
        tools.dry_run = False
        tools.quiet_mode = True
        tools.make_check_failed = False

    def tearDown(self):
        os.chdir(self._old_cwd)
        self.tmp.cleanup()

    def write(self, name, text):
        with open(name, "w", encoding="utf-8") as f:
            f.write(text)
        return name


class TestPathSafety(ToolTestCase):
    def test_reads_normal_file(self):
        self.write("a.py", "x = 1\n")
        self.assertEqual(tools.tool_read_file("a.py"), "x = 1\n")

    def test_refuses_parent_directory(self):
        self.write("a.py", "x = 1\n")
        self.assertIn("outside the working directory", tools.tool_read_file("../a.py"))

    def test_refuses_absolute_path_outside(self):
        self.assertIn("outside the working directory",
                      tools.tool_read_file("/etc/hosts"))

    def test_refuses_hidden_file_in_any_component(self):
        os.makedirs(".git")
        self.write(os.path.join(".git", "config"), "secret\n")
        self.assertIn("hidden", tools.tool_read_file(".git/config"))

    def test_list_dir_skips_hidden_entries(self):
        self.write("visible.py", "")
        self.write(".hidden", "")
        os.makedirs(".git")
        self.assertEqual(tools.tool_list_dir("."), "visible.py")


class TestLsFiltering(ToolTestCase):
    def test_long_listing_keeps_names_with_spaces(self):
        out = ("total 8\n"
               "drwxr-xr-x  4 u  g  128 Jan  1 00:00 .\n"
               "drwxr-xr-x  4 u  g  128 Jan  1 00:00 ..\n"
               "-rw-r--r--  1 u  g    0 Jan  1 00:00 my file.txt\n"
               "-rw-r--r--  1 u  g    0 Jan  1 00:00 .hidden\n")
        self.assertEqual(tools.filter_ls_output(out), "-rw-r--r--  1 u  g    0 Jan  1 00:00 my file.txt")

    def test_short_listing_drops_hidden_names(self):
        self.assertEqual(tools.filter_ls_output("a.py .hidden b.py\n"), "a.py b.py")

    def test_short_listing_all_hidden_is_dropped(self):
        self.assertEqual(tools.filter_ls_output(".a .b\n"), "")


class TestGrep(ToolTestCase):
    def test_finds_matches_as_path_line_text(self):
        self.write("mod.py", "alpha\nbeta\n")
        self.assertEqual(tools.tool_grep("beta", "."), "mod.py:2:beta")

    def test_skips_hidden_directories(self):
        os.makedirs(os.path.join(".git", "objects"))
        self.write(os.path.join(".git", "objects", "pack"), "needle\n")
        self.write("mod.py", "needle\n")
        self.assertEqual(tools.tool_grep("needle", "."), "mod.py:1:needle")

    def test_invalid_regex_is_reported_not_raised(self):
        self.assertIn("invalid regular expression", tools.tool_grep("(unclosed", "."))

    def test_output_is_capped(self):
        self.write("big.py", "hit\n" * (tools.GREP_MAX_MATCHES + 50))
        out = tools.tool_grep("hit", ".")
        self.assertIn("stopped after {} matches".format(tools.GREP_MAX_MATCHES), out)
        self.assertEqual(len(out.split("\n")), tools.GREP_MAX_MATCHES + 1)

    def test_pattern_starting_with_dash_is_not_a_flag(self):
        self.write("mod.py", "-x\n")
        self.assertEqual(tools.tool_grep("-x", "."), "mod.py:1:-x")


class TestRunShell(ToolTestCase):
    def test_refuses_non_whitelisted_command(self):
        self.assertIn("not whitelisted", tools.tool_run_shell("rm -rf ."))

    def test_refuses_path_outside_working_dir(self):
        self.assertIn("outside the working directory", tools.tool_run_shell("cat /etc/hosts"))

    def test_refuses_hidden_path(self):
        self.write(".env", "SECRET=1\n")
        self.assertIn("hidden", tools.tool_run_shell("cat .env"))

    def test_runs_whitelisted_command(self):
        self.write("note.txt", "hello\n")
        self.assertIn("hello", tools.tool_run_shell("cat note.txt"))

    def test_timeout_becomes_exit_code_not_exception(self):
        out, code = tools.run_external("sleep", ["5"], timeout=1)
        self.assertEqual(code, tools.TIMEOUT_EXIT_CODE)
        self.assertIn("timed out", out)


class TestEdits(ToolTestCase):
    def test_propose_edit_detects_stale_base(self):
        self.write("a.py", "new contents\n")
        result = tools.tool_propose_edit("a.py", "old contents\n", "other\n")
        self.assertIn("stale base", result)

    def test_dry_run_shows_diff_without_writing(self):
        self.write("a.py", "x = 1\n")
        tools.dry_run = True
        result = tools.tool_propose_edit("a.py", "x = 1\n", "x = 2\n")
        self.assertIn("dry-run", result)
        self.assertEqual(open("a.py").read(), "x = 1\n")

    def test_replace_in_file_rewrites_one_unique_snippet(self):
        self.write("a.py", "def normalize(s):\n    return s.strip()\n")
        tools.auto_approve = True
        result = tools.tool_replace_in_file("a.py", "return s.strip()", "return s.strip().lower()")
        self.assertIn("make check", result)
        self.assertIn("s.strip().lower()", open("a.py").read())

    def test_replace_in_file_rejects_ambiguous_snippet(self):
        self.write("a.py", "pass\npass\n")
        result = tools.tool_replace_in_file("a.py", "pass", "return")
        self.assertIn("appears 2 times", result)
        self.assertEqual(open("a.py").read(), "pass\npass\n")

    def test_replace_in_file_reports_missing_snippet(self):
        self.write("a.py", "x = 1\n")
        self.assertIn("not found", tools.tool_replace_in_file("a.py", "y = 2", "y = 3"))

    def test_failed_check_sets_the_module_flag(self):
        os.makedirs("proj")
        os.chdir("proj")
        self.write("Makefile", "check:\n\t@exit 1\n")
        tools.auto_approve = True
        result = tools.tool_propose_edit("Makefile", "check:\n\t@exit 1\n", "check:\n\t@exit 1\n# note\n")
        self.assertIn("make check FAILED", result)
        self.assertTrue(tools.make_check_failed)
        os.chdir(self.tmp.name)


class TestDispatch(ToolTestCase):
    def test_unknown_tool_is_feedback_not_an_exception(self):
        out = tools.call_tool("nope", {})
        self.assertIn("unknown tool", out)

    def test_missing_argument_is_reported(self):
        out = tools.call_tool("read_file", {})
        self.assertIn("missing required argument", out)

    def test_empty_string_is_a_valid_argument(self):
        # propose_edit uses "" as `old` to create a new file.
        result = tools.call_tool("propose_edit", {"path": "new.py", "old": "", "new": ""})
        self.assertIn("cannot create an empty file", result)

    def test_execute_tool_calls_replaces_a_missing_id(self):
        calls = [{"function": {"name": "list_dir", "arguments": '{"path": "."}'}}]
        results = tools.execute_tool_calls(calls)
        self.assertEqual(results[0][0], "call_0")

    def test_bad_json_reports_the_decode_error(self):
        calls = [{"id": "c1", "function": {"name": "read_file", "arguments": "{oops"}}]
        result = tools.execute_tool_calls(calls)[0][2]
        self.assertIn("invalid JSON", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
