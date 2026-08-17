"""The README's test count, checked against the suite it describes.

The README states how many tests this repo has in two places: the quickstart
block at the top, and the repo map near the bottom. Both are the first thing a
reader can verify for themselves, on a project whose pitch is that its numbers
are honest, and both had drifted independently (336 and 322 against an actual
348) because nothing was watching them.

A number in prose that no test reads is a number that goes stale. This module
makes the two claims enforceable: they are parsed out of the README and
asserted against a live discovery of the suite, so adding a test without
updating the README fails CI rather than quietly costing the project a little
credibility.

Deliberately anchored to the exact lines that make the claim, not to every
"N tests" in the file. The Phase 1 note further down reports 142 tests as
history, and history does not change when the suite grows.
"""

import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
README = os.path.join(REPO_ROOT, "README.md")

# The quickstart block: `... unittest discover -s tests   # 348 tests`
QUICKSTART_CLAIM = re.compile(r"unittest discover -s tests\s+#\s*(\d+) tests")
# The repo map: `tests/           348 tests; protocol invariants, ...`
REPO_MAP_CLAIM = re.compile(r"^tests/\s+(\d+) tests;", re.MULTILINE)


def read_readme():
    with open(README, encoding="utf-8") as handle:
        return handle.read()


def count_tests():
    """Number of test cases `unittest discover -s tests` would run.

    Counted by walking the discovered suite rather than by parsing the runner's
    output, so this stays correct if a module is added, split or renamed.
    """
    suite = unittest.defaultTestLoader.discover(TESTS_DIR, top_level_dir=TESTS_DIR)
    return sum(1 for _ in iter_cases(suite))


def iter_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_cases(item)
        else:
            yield item


class ReadmeTestCountTests(unittest.TestCase):
    def setUp(self):
        self.readme = read_readme()
        self.actual = count_tests()

    def test_discovery_loaded_every_module(self):
        """A module that fails to import is discovered as a failing placeholder.

        Without this, a broken import would still be counted and the two claims
        below would keep agreeing with a suite that no longer runs.
        """
        broken = [
            str(case)
            for case in iter_cases(
                unittest.defaultTestLoader.discover(TESTS_DIR, top_level_dir=TESTS_DIR)
            )
            if type(case).__name__ == "_FailedTest"
        ]
        self.assertEqual(broken, [], f"test modules failed to load: {broken}")

    def test_the_quickstart_block_states_the_real_count(self):
        match = QUICKSTART_CLAIM.search(self.readme)
        self.assertIsNotNone(
            match,
            "the README quickstart block no longer states a test count in the "
            "form this test reads; re-point the regex rather than dropping it",
        )
        self.assertEqual(
            int(match.group(1)),
            self.actual,
            "README quickstart block claims a stale test count",
        )

    def test_the_repo_map_states_the_real_count(self):
        match = REPO_MAP_CLAIM.search(self.readme)
        self.assertIsNotNone(
            match,
            "the README repo map no longer states a test count in the form this "
            "test reads; re-point the regex rather than dropping it",
        )
        self.assertEqual(
            int(match.group(1)),
            self.actual,
            "README repo map claims a stale test count",
        )

    def test_both_claims_agree_with_each_other(self):
        """They drifted apart once, which is how the staleness went unnoticed."""
        quickstart = QUICKSTART_CLAIM.search(self.readme)
        repo_map = REPO_MAP_CLAIM.search(self.readme)
        self.assertEqual(quickstart.group(1), repo_map.group(1))


if __name__ == "__main__":
    unittest.main()
