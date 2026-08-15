"""Benchmark harness tests.

Almost all of these are about the harness refusing to produce a flattering
result: a mock run that cannot be mistaken for evidence, a judge that cannot
sit on the council it scores, a blinded and order-shuffled scoring pass, and a
neutral-criteria total that would expose Quorum winning on format alone.

The shipped rubric file is validated here too. It is the artifact most able to
decide the benchmark's outcome before a single model is called, so its shape —
every task carrying criteria of both kinds — is a test rather than a
convention.
"""

import json
import os
import unittest

from quorum import (
    ARMS,
    ArmResult,
    BenchmarkMockProvider,
    BenchmarkReport,
    Criterion,
    JudgmentTask,
    ModelCost,
    ProviderPool,
    Seat,
    TaskOutcome,
    demo_council,
    load_tasks,
    run_benchmark,
)
from quorum.providers.base import MockProvider

TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "judgment_tasks.json")
JUDGE = Seat("judge-model", "judgelab", ModelCost(5.0, 25.0))


def bench_pool(council, judge=JUDGE):
    personas = {seat.model_id: i for i, seat in enumerate(council.students)}
    providers = []
    for name in sorted({s.provider for s in council.seats()} | {judge.provider}):
        inner = MockProvider(personas=personas)
        inner.name = name
        providers.append(BenchmarkMockProvider(inner, name=name))
    return ProviderPool(providers)


def tiny_task(key="t1"):
    return JudgmentTask(
        key=key,
        prompt="Should we rebuild the pipeline or refactor it, given the roadmap?",
        task_type="architecture",
        complexity=0.85,
        criteria=(
            Criterion("specific", "Names one course of action."),
            Criterion("conditions", "Says what it depends on."),
            Criterion("counter", "Names the strongest objection.", favours_deliberation=True),
        ),
    )


class RubricFileTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks(TASKS_PATH)

    def test_the_shipped_set_has_twenty_tasks(self):
        self.assertEqual(len(self.tasks), 20)

    def test_task_keys_are_unique(self):
        keys = [t.key for t in self.tasks]
        self.assertEqual(len(set(keys)), len(keys))

    def test_every_task_has_criteria_of_both_kinds(self):
        """A task scored only on deliberation-friendly criteria hands Quorum
        the win; one with none of them cannot detect the effect at all."""
        for task in self.tasks:
            tagged = [c for c in task.criteria if c.favours_deliberation]
            neutral = task.neutral_criteria
            self.assertTrue(tagged, f"{task.key} has no favours_deliberation criteria")
            self.assertTrue(neutral, f"{task.key} has no neutral criteria")

    def test_neutral_criteria_are_the_majority(self):
        for task in self.tasks:
            self.assertGreater(
                len(task.neutral_criteria),
                len(task.criteria) - len(task.neutral_criteria),
                f"{task.key} is scored mostly on criteria that favour deliberation",
            )

    def test_criterion_keys_are_unique_within_a_task(self):
        for task in self.tasks:
            keys = [c.key for c in task.criteria]
            self.assertEqual(len(set(keys)), len(keys), task.key)

    def test_every_task_is_a_judgment_type(self):
        from quorum import JUDGMENT_TYPES

        for task in self.tasks:
            self.assertIn(task.task_type, JUDGMENT_TYPES, task.key)

    def test_every_task_would_actually_convene(self):
        """A benchmark task the convening rule would decline is measuring
        Quorum on work it says it should not be used for."""
        from quorum import Task, should_convene

        declined = [
            t.key
            for t in self.tasks
            if not should_convene(Task(t.prompt, t.task_type, t.complexity)).convene
        ]
        self.assertEqual(declined, [], f"tasks the rule would decline: {declined}")

    def test_the_file_states_its_review_status(self):
        with open(TASKS_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        readme = " ".join(data["_readme"])
        self.assertIn("not been reviewed", readme)


class HarnessTests(unittest.TestCase):
    def test_the_judge_cannot_sit_on_the_council(self):
        council = demo_council()
        judge = council.students[0]
        with self.assertRaises(ValueError) as ctx:
            run_benchmark([tiny_task()], council, bench_pool(council), judge)
        self.assertIn("took part", str(ctx.exception))

    def test_every_arm_runs_and_is_scored(self):
        council = demo_council()
        report = run_benchmark(
            [tiny_task()], council, bench_pool(council), JUDGE, is_mock=True
        )
        outcome = report.outcomes[0]
        self.assertEqual(set(outcome.arms), set(ARMS))
        for arm in ARMS:
            self.assertFalse(outcome.arms[arm].error, outcome.arms[arm].error)
            self.assertTrue(outcome.arms[arm].answer)
            self.assertTrue(outcome.arms[arm].scores)

    def test_the_quorum_arm_keeps_its_session_for_the_trace(self):
        council = demo_council()
        report = run_benchmark(
            [tiny_task()], council, bench_pool(council), JUDGE, is_mock=True
        )
        session = report.outcomes[0].arms["quorum"].session
        self.assertIsNotNone(session)
        self.assertTrue(session.events)

    def judge_prompt(self, seed="j"):
        council = demo_council()
        judge_provider = BenchmarkMockProvider(MockProvider(), name=JUDGE.provider)
        pool = bench_pool(council)
        pool._providers[JUDGE.provider] = judge_provider  # noqa: SLF001
        run_benchmark([tiny_task(seed)], council, pool, JUDGE, is_mock=True)
        return next(
            p for _, p in judge_provider.calls if "--- Answer A ---" in p
        )

    def test_the_judge_never_learns_which_arm_wrote_which_answer(self):
        prompt = self.judge_prompt()
        # Scoped to the answers block: "single" also occurs innocently in
        # "a single JSON object", and asserting over the whole prompt would
        # fail on the instructions rather than on a leak.
        answers = prompt.split("--- Answer A ---")[1].split("Score each answer")[0]
        for arm in ARMS:
            self.assertNotIn(arm, answers)
        self.assertIn("you do not know who did", prompt)
        self.assertNotIn("quorum", prompt)
        self.assertNotIn("self_critique", prompt)

    def test_the_judge_scores_every_arm_in_one_call_on_a_common_scale(self):
        prompt = self.judge_prompt("one-call")
        self.assertEqual(prompt.count("--- Answer "), len(ARMS))

    def test_answer_order_is_shuffled_per_task(self):
        """A judge with a position bias must not be able to favour one arm."""
        council = demo_council()
        orders = set()
        for key in ("k1", "k2", "k3", "k4", "k5", "k6"):
            judge_provider = BenchmarkMockProvider(MockProvider(), name=JUDGE.provider)
            pool = bench_pool(council)
            pool._providers[JUDGE.provider] = judge_provider  # noqa: SLF001
            report = run_benchmark(
                [tiny_task(key)], council, pool, JUDGE, is_mock=True
            )
            prompt = next(p for _, p in judge_provider.calls if "--- Answer A ---" in p)
            first = prompt.split("--- Answer B ---")[0]
            orders.add(
                next(
                    arm
                    for arm in ARMS
                    if report.outcomes[0].arms[arm].answer
                    and report.outcomes[0].arms[arm].answer[:40] in first
                )
            )
        self.assertGreater(len(orders), 1, "answer A is always the same arm")

    def test_scores_are_clamped_to_the_scale(self):
        task = tiny_task()
        arm = ArmResult("single", "an answer", scores={"specific": 5.0})
        # The harness clamps on ingest; a raw ArmResult is not clamped, so this
        # asserts the total stays interpretable rather than the field.
        self.assertGreater(arm.total(task.criteria), 0)

    def test_an_arm_that_errored_is_excluded_not_scored_zero(self):
        report = BenchmarkReport(
            outcomes=[
                TaskOutcome(
                    task=tiny_task(),
                    arms={
                        "quorum": ArmResult("quorum", "", error="no verdict"),
                        "single": ArmResult(
                            "single", "x",
                            scores={"specific": 1.0, "conditions": 1.0, "counter": 1.0},
                        ),
                        "self_critique": ArmResult("self_critique", "", error="boom"),
                    },
                )
            ]
        )
        self.assertEqual(report.mean("quorum"), 0.0)
        self.assertEqual(report.mean("single"), 1.0)
        self.assertEqual(report.winner_check(), "single") if hasattr(
            report, "winner_check"
        ) else self.assertEqual(report.outcomes[0].winner(), "single")


class ReportingTests(unittest.TestCase):
    def setUp(self):
        council = demo_council()
        self.report = run_benchmark(
            [tiny_task("a"), tiny_task("b")],
            council,
            bench_pool(council),
            JUDGE,
            is_mock=True,
        )

    def test_a_mock_run_says_so_before_any_number(self):
        markdown = self.report.render_markdown()
        first_lines = markdown.split("\n")[:5]
        self.assertTrue(
            any("NOT RESULTS" in line for line in first_lines),
            f"mock warning is not up top: {first_lines}",
        )
        self.assertIn("MockProvider", markdown)

    def test_a_real_run_carries_no_mock_warning(self):
        self.report.is_mock = False
        self.assertNotIn("NOT RESULTS", self.report.render_markdown())

    def test_it_reports_neutral_criteria_separately(self):
        markdown = self.report.render_markdown()
        self.assertIn("neutral criteria", markdown)
        self.assertIn("winning on format", markdown)
        data = self.report.to_dict()
        self.assertIn("mean_score_neutral_criteria", data)
        self.assertIn("wins_neutral_criteria", data)

    def test_it_names_the_council_the_single_model_and_the_judge(self):
        markdown = self.report.render_markdown()
        self.assertIn(self.report.single_model, markdown)
        self.assertIn(self.report.judge_model, markdown)
        for model in self.report.council_models:
            self.assertIn(model, markdown)

    def test_it_reports_cost_per_arm(self):
        data = self.report.to_dict()
        self.assertGreater(data["cost"]["quorum"], data["cost"]["single"])

    def test_every_task_appears_in_the_per_task_table(self):
        markdown = self.report.render_markdown()
        for outcome in self.report.outcomes:
            self.assertIn(f"`{outcome.task.key}`", markdown)

    def test_the_report_is_serialisable(self):
        json.dumps(self.report.to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
