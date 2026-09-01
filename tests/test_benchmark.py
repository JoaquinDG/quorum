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
from quorum.benchmark import judge
from quorum.providers.base import Completion, MockProvider

TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "judgment_tasks.json")
JUDGE = Seat("judge-model", "judgelab", ModelCost(5.0, 25.0))


class FixedJudge:
    """A judge whose reply the test dictates, to exercise a partial scoring."""

    name = JUDGE.provider

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, model_id, prompt, max_tokens=2048):
        return Completion(text=self.reply, model_id=model_id)


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

    def test_every_task_keeps_at_least_one_neutral_criterion(self):
        """Weakened from "neutral criteria are the majority", on evidence.

        An independent reviewer called all 19 rubrics it saw
        deliberation-favouring and flagged 44 of 120 criteria as mistagged —
        including `conditions`, which sat in the neutral column of every task
        and which a critique round surfaces mechanically. Applying its verdict
        honestly leaves 7 of 20 tasks without a neutral majority.

        The majority could be restored by inventing neutral criteria to hit the
        old bar, which would be writing the rubric to pass its own test. The
        finding is more useful than the guarantee: **a rubric for judgement
        questions tilts toward deliberation close to structurally**, because
        the properties that make an answer good on an ambiguous question are
        largely the properties a debate produces. That is a real limit on what
        this benchmark can prove, and it is why the harness reports both
        columns rather than trusting either.
        """
        for task in self.tasks:
            self.assertTrue(
                task.neutral_criteria,
                f"{task.key} has no criterion a single answer can reach on merit",
            )

    def test_the_deliberation_tilt_is_recorded_rather_than_hidden(self):
        tilted = [
            t.key for t in self.tasks
            if len(t.neutral_criteria) <= len(t.criteria) - len(t.neutral_criteria)
        ]
        # Pinned so a future edit that quietly restores a comfortable majority
        # has to explain itself.
        self.assertEqual(len(tilted), 7, f"tilt changed: {tilted}")

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

    def test_the_file_records_its_adversarial_review(self):
        with open(TASKS_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        readme = " ".join(data["_readme"])
        self.assertIn("REVIEWED", readme)
        self.assertIn("hostile", readme)
        self.assertIn("44 of 120", readme)

    def test_the_unmeasurable_criterion_is_gone_everywhere(self):
        """`calibrated` was called unmeasurable in 18 of 19 reviews: a judge
        cannot apply it consistently, and it rewards the hedged tone
        synthesised text has more of."""
        for task in self.tasks:
            self.assertNotIn("calibrated", [c.key for c in task.criteria], task.key)

    def test_conditions_is_tagged_everywhere_it_appears(self):
        """It sat in the neutral column of every task. A critique round
        surfaces dependencies mechanically, so a council scored it for free."""
        for task in self.tasks:
            for c in task.criteria:
                if c.key == "conditions":
                    self.assertTrue(c.favours_deliberation, task.key)


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
        # The errored arms contribute no zeros. `single` contributes nothing
        # either: its one task is not comparable, so there is nothing to
        # average. It scored, but not against anyone.
        self.assertEqual(report.mean("quorum"), 0.0)
        self.assertEqual(report.mean("single"), 0.0)
        # Previously "single", which is the walkover this item exists to stop.
        self.assertEqual(report.outcomes[0].winner(), "none")


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


class TokenBudgetTests(unittest.TestCase):
    """A budget too small for a reasoning model favours the council silently.

    The first real run had `single` and `self_critique` cost $0.00 and produce
    nothing: 1024 tokens went entirely to reasoning, the adapter raised
    `ProviderTruncated`, and the harness recorded "no scored answer". A
    benchmark built to ask whether a council beats one model would have shown
    the one model failing to answer — favouring the council, in the exact
    measurement meant to check it.

    The council arm was unaffected because `SessionConfig.max_tokens` had
    already been raised for the same reason, which is what made the failure
    one-sided rather than obvious.
    """

    def test_the_arms_share_the_session_budget_order_of_magnitude(self):
        from quorum.benchmark import MAX_TOKENS
        from quorum import SessionConfig

        self.assertGreaterEqual(MAX_TOKENS, SessionConfig().max_tokens)

    def test_no_arm_still_hardcodes_a_small_budget(self):
        import inspect
        from quorum import benchmark

        source = inspect.getsource(benchmark)
        self.assertNotIn(", 1024)", source)
        self.assertNotIn(", 1024\n", source)


class IncompleteRunTests(unittest.TestCase):
    """A benchmark with a missing arm is not a close result. It is not a result.

    A real run had one provider out of credit. The single-model and
    self-critique arms produced nothing; the council arm degraded to two
    students and carried on, exactly as fail-closed intends, and scored 0.87
    against two zeros. The harness printed "wins: quorum 2" above the failure
    notice — the most flattering possible rendering of a walkover.
    """

    def report_with_a_missing_arm(self):
        task = tiny_task()
        return BenchmarkReport(outcomes=[TaskOutcome(task=task, arms={
            "quorum": ArmResult("quorum", "an answer",
                                scores={c.key: 0.9 for c in task.criteria}),
            "single": ArmResult("single", "", error="credit balance too low"),
            "self_critique": ArmResult("self_critique", "", error="credit balance too low"),
        })])

    def test_a_missing_arm_makes_the_run_incomparable(self):
        report = self.report_with_a_missing_arm()
        self.assertFalse(report.comparable)
        self.assertEqual(
            sorted({arm for _, arm in report.incomplete_arms}),
            ["self_critique", "single"],
        )

    def test_the_warning_precedes_any_score(self):
        markdown = self.report_with_a_missing_arm().render_markdown()
        self.assertIn("NOT A RESULT", markdown)
        self.assertLess(
            markdown.index("NOT A RESULT"),
            markdown.index("Mean rubric score"),
            "the leaderboard appears before the warning that it is meaningless",
        )

    def test_a_complete_run_carries_no_such_warning(self):
        council = demo_council()
        report = run_benchmark(
            [tiny_task()], council, bench_pool(council), JUDGE, is_mock=True
        )
        self.assertTrue(report.comparable)
        self.assertNotIn("NOT A RESULT", report.render_markdown())

    def test_the_surviving_arm_is_not_handed_the_task(self):
        """The walkover, at the level the tally actually reports."""
        report = self.report_with_a_missing_arm()
        self.assertEqual(report.outcomes[0].winner(), "none")
        self.assertEqual(report.wins()["quorum"], 0)
        self.assertEqual(report.wins()["none"], 1)


class UnscoredArmTests(unittest.TestCase):
    """An arm that answered but was never scored is absent, not zero.

    The 2026-08-19 run's dominant failure: 16 tasks showed `err` in all three
    arms at once, the signature of a judge call failing rather than of three
    independent provider failures. A judge that omits a label leaves the arm
    with an answer, no error, and an empty score dict, and `total()` reads a
    missing criterion as 0.0. So the arm was reported as missing by the banner
    and simultaneously averaged in as a real 0.00, printed in the per-task
    table as a score, and beaten by whichever arm the judge did reach.
    """

    def judged_by(self, reply: str, task=None):
        """Score three real answers with a judge whose reply we control."""
        task = task or tiny_task()
        arms = [ArmResult(arm, f"answer from {arm}") for arm in ARMS]
        judge(task, arms, JUDGE, ProviderPool([FixedJudge(reply)]),
              shuffle_seed=task.key)
        return arms

    def test_an_omitted_label_is_recorded_as_an_error(self):
        # The judge scores A and B and silently drops C.
        arms = self.judged_by(json.dumps({
            "A": {"specific": 0.9, "conditions": 0.9, "counter": 0.9},
            "B": {"specific": 0.8, "conditions": 0.8, "counter": 0.8},
        }))
        errored = [a for a in arms if a.error]
        self.assertEqual(len(errored), 1, "the dropped answer carries no error")
        self.assertIn("no usable scores", errored[0].error)
        self.assertFalse(errored[0].complete)

    def test_scores_matching_no_criterion_are_an_error_not_a_zero(self):
        arms = self.judged_by(json.dumps({
            label: {"not_a_criterion": 1.0} for label in ("A", "B", "C")
        }))
        for arm in arms:
            self.assertTrue(arm.error, "an unscorable arm passed as scored")
            self.assertEqual(arm.scores, {})

    def test_a_fully_scored_reply_still_passes(self):
        arms = self.judged_by(json.dumps({
            label: {"specific": 0.7, "conditions": 0.7, "counter": 0.7}
            for label in ("A", "B", "C")
        }))
        for arm in arms:
            self.assertEqual(arm.error, "")
            self.assertTrue(arm.complete)

    def test_an_unscored_arm_is_never_averaged_in_as_zero(self):
        task = tiny_task()
        full = {c.key: 0.8 for c in task.criteria}
        report = BenchmarkReport(outcomes=[TaskOutcome(task=task, arms={
            # answered, no error, and the judge never scored it
            "quorum": ArmResult("quorum", "a real answer", scores={}),
            "single": ArmResult("single", "x", scores=full),
            "self_critique": ArmResult("self_critique", "y", scores=full),
        })])
        self.assertFalse(report.outcomes[0].arms["quorum"].complete)
        self.assertFalse(report.comparable)
        self.assertEqual(report.mean("quorum"), 0.0)
        self.assertEqual(report.outcomes[0].winner(), "none")
        self.assertEqual(report.incomplete_arms, [(task.key, "quorum")])


class ExclusionRenderingTests(unittest.TestCase):
    """H1's closing condition: the per-task table cannot show a score next to an err."""

    def mixed_report(self):
        """One task every arm answered, one where `single` did not."""
        good, bad = tiny_task("good"), tiny_task("bad")
        full = {c.key: 0.8 for c in good.criteria}
        return BenchmarkReport(outcomes=[
            TaskOutcome(task=good, arms={
                "quorum": ArmResult("quorum", "a", scores=full, cost_est=0.10),
                "single": ArmResult("single", "b", scores=full, cost_est=0.01),
                "self_critique": ArmResult("self_critique", "c", scores=full,
                                           cost_est=0.02),
            }),
            TaskOutcome(task=bad, arms={
                "quorum": ArmResult("quorum", "a", scores=full, cost_est=0.10),
                "single": ArmResult("single", "", error="out of credit",
                                    cost_est=0.00),
                "self_critique": ArmResult("self_critique", "c", scores=full,
                                           cost_est=0.02),
            }),
        ])

    def per_task_rows(self, markdown):
        return [ln for ln in markdown.splitlines() if ln.startswith("| `")]

    def test_no_row_shows_a_score_beside_an_err(self):
        for row in self.per_task_rows(self.mixed_report().render_markdown()):
            if "err" not in row:
                continue
            cells = [c.strip() for c in row.split("|")[2:5]]
            offenders = [c for c in cells if c not in ("err", "n/c")]
            self.assertEqual(
                offenders, [],
                f"a score is printed next to an err: {row}",
            )

    def test_an_excluded_task_names_no_winner(self):
        rows = self.per_task_rows(self.mixed_report().render_markdown())
        bad = next(r for r in rows if "`bad`" in r)
        self.assertIn("excluded", bad)
        self.assertNotIn("quorum", bad)

    def test_excluded_tasks_are_counted_not_dropped(self):
        report = self.mixed_report()
        self.assertEqual(report.excluded, ["bad"])
        self.assertEqual(report.compared, 1)
        markdown = report.render_markdown()
        self.assertIn("`bad`", markdown, "an excluded task vanished from the report")
        self.assertIn("not compared", markdown)
        data = report.to_dict()
        self.assertEqual(data["tasks_excluded"], ["bad"])
        self.assertEqual(data["tasks_compared"], 1)
        self.assertEqual(data["tasks"], 2)

    def test_means_are_over_compared_tasks_only(self):
        report = self.mixed_report()
        # quorum answered both, but only the shared task counts. Averaging it
        # over two while `single` gets one is the comparison H1 forbids.
        self.assertAlmostEqual(report.mean("quorum"), 0.8)
        self.assertAlmostEqual(report.mean("single"), 0.8)

    def test_cost_still_covers_the_excluded_task(self):
        # The bill is the bill. Excluding a task from the score does not
        # refund the tokens it burned.
        self.assertAlmostEqual(self.mixed_report().cost("quorum"), 0.20)

    def test_a_clean_run_carries_no_exclusion_legend(self):
        council = demo_council()
        report = run_benchmark(
            [tiny_task()], council, bench_pool(council), JUDGE, is_mock=True
        )
        markdown = report.render_markdown()
        self.assertEqual(report.excluded, [])
        self.assertNotIn("not compared", markdown)
        self.assertNotIn("n/c", markdown)
