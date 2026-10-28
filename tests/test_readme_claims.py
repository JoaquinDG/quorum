"""The README's self-reported numbers, checked against what the code does.

The README states how many tests this repo has in two places: the quickstart
block at the top, and the repo map near the bottom. Both are the first thing a
reader can verify for themselves, on a project whose pitch is that its numbers
are honest, and both had drifted independently (336 and 322 against an actual
348) because nothing was watching them.

A number in prose that no test reads is a number that goes stale. This module
makes those claims enforceable: they are parsed out of the README and asserted
against a live discovery of the suite, so adding a test without updating the
README fails CI rather than quietly costing the project a little credibility.

Deliberately anchored to the exact lines that make the claim, not to every
"N tests" in the file. The Phase 1 note further down reports 142 tests as
history, and history does not change when the suite grows.

The same reasoning covers the quickstart's bill. That block sat at the 0.1.0
figures ($0.0379 and 6.7x) from August to October while the shield's fencing
pushed the real session to $0.0408 and 7.2x, so the README understated both
the cost and the multiple. Cost drift is worse than test-count drift, because
it runs in the reassuring direction by default: every change that adds tokens
makes the published multiple look better than the code delivers, and the
guardrail in that same block is what the number is being judged against.
"""

import importlib.util
import os
import re
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")
README = os.path.join(REPO_ROOT, "README.md")
QUICKSTART = os.path.join(REPO_ROOT, "examples", "quickstart.py")

# The quickstart block: `... unittest discover -s tests   # 348 tests`
QUICKSTART_CLAIM = re.compile(r"unittest discover -s tests\s+#\s*(\d+) tests")
# The repo map: `tests/           348 tests; protocol invariants, ...`
REPO_MAP_CLAIM = re.compile(r"^tests/\s+(\d+) tests;", re.MULTILINE)

# The three lines of the fenced bill block under "## The bill". Anchored to
# the start of the line, which is what distinguishes the README's own fenced
# copy from the quickstart's indented terminal output quoted elsewhere.
BILL_SESSION = re.compile(r"^this session:\s+\$(\d+\.\d+)\s*$", re.MULTILINE)
BILL_BASELINE = re.compile(
    r"^one model, once:\s+\$(\d+\.\d+)\s+\(([^)]+)\)\s*$", re.MULTILINE
)
BILL_MULTIPLE = re.compile(
    r"^multiple:\s+(\d+\.\d+)x\s+\((within|OVER) the 8x guardrail\)\s*$",
    re.MULTILINE,
)


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


def quickstart_bill():
    """Run the quickstart's own session offline and return its result.

    The example is imported rather than reproduced, so the task, the council
    and the mock pool behind the published number are the ones a reader gets
    when they run the command the README gives them. Restating the task here
    would let the example drift away from the claim, which is the whole bug
    this module exists to catch. Importing is side-effect free: everything
    that runs a session lives under ``main()``.

    No trace path is passed. The file the example writes is a demo artifact
    that the replay tests also read, and a test that rewrites it would make
    the suite order-dependent.
    """
    spec = importlib.util.spec_from_file_location("_quickstart_example", QUICKSTART)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    council = example.demo_council()
    return example.Session(council, example.mock_pool(council)).run(
        example.QUESTION.prompt
    )


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


class ReadmeBillBlockTests(unittest.TestCase):
    """The bill block quotes a real run, so these assert it is the real run.

    Run once for the class: the session is deterministic against mocks, and
    three assertions do not need three sessions.
    """

    @classmethod
    def setUpClass(cls):
        cls.readme = read_readme()
        cls.result = quickstart_bill()

    def test_the_session_cost_is_the_one_the_quickstart_prints(self):
        match = BILL_SESSION.search(self.readme)
        self.assertIsNotNone(
            match,
            "the README bill block no longer states a session cost in the "
            "form this test reads; re-point the regex rather than dropping it",
        )
        self.assertEqual(
            match.group(1),
            f"{self.result.cost_est:.4f}",
            "README bill block quotes a stale session cost; regenerate the "
            "block from a real run rather than editing the number by hand",
        )

    def test_the_baseline_is_the_one_the_quickstart_prints(self):
        match = BILL_BASELINE.search(self.readme)
        self.assertIsNotNone(
            match,
            "the README bill block no longer states a baseline in the form "
            "this test reads; re-point the regex rather than dropping it",
        )
        self.assertEqual(
            match.group(1),
            f"{self.result.baseline.cost_est:.4f}",
            "README bill block quotes a stale single-model baseline",
        )
        self.assertEqual(
            match.group(2),
            self.result.baseline.model_id,
            "README bill block names a model that is no longer the baseline "
            "seat; the printed model id is what lets a reader disagree with "
            "the comparison",
        )

    def test_the_multiple_and_its_guardrail_verdict_are_both_current(self):
        """The verdict matters as much as the number.

        A block reading "within the 8x guardrail" next to a multiple that has
        crossed 8 would be the one failure mode here that flatters the
        project twice over.
        """
        match = BILL_MULTIPLE.search(self.readme)
        self.assertIsNotNone(
            match,
            "the README bill block no longer states a cost multiple in the "
            "form this test reads; re-point the regex rather than dropping it",
        )
        self.assertIsNotNone(
            self.result.cost_multiple,
            "the quickstart session no longer reports a cost multiple, so the "
            "README is quoting a number the code cannot produce",
        )
        self.assertEqual(
            match.group(1),
            f"{self.result.cost_multiple:.1f}",
            "README bill block quotes a stale cost multiple",
        )
        self.assertEqual(
            match.group(2),
            "within" if self.result.cost_multiple <= 8 else "OVER",
            "README bill block states the wrong verdict against the 8x "
            "guardrail for the multiple it prints",
        )


if __name__ == "__main__":
    unittest.main()
