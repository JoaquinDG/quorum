"""Convening rule and cost-baseline tests.

The rule's job is to say *no* most of the time, so most of these tests are
about the gates. The ones that matter are the two traps:

- a genuinely hard task with a checkable answer must not convene (difficulty
  is not the signal — arguability is), and
- a task phrased like a judgement call but trivially decided must not convene
  (phrasing is not the signal either).

`evals/convening_eval.py` covers the rate across a whole workload, which is
what the acceptance criterion is actually about; a unit test cannot assert a
rate.
"""

import unittest

from quorum import (
    CONVENE_ALWAYS,
    CONVENE_CONSERVATIVE,
    CONVENE_DEFAULT,
    CONVENE_NEVER,
    CONVENING_PRESETS,
    Council,
    ModelCost,
    Seat,
    SessionConfig,
    Session,
    Task,
    convene,
    demo_council,
    mock_pool,
    pick_baseline_seat,
    should_convene,
    signals,
    single_model_baseline,
)

REBUILD = Task(
    "Our ingestion pipeline is three years old and increasingly slow. Should we rebuild "
    "it on a streaming architecture this year, or refactor it in place? The rebuild is a "
    "one-way door on the storage format.",
    "architecture",
    0.85,
)


class GateTests(unittest.TestCase):
    def test_a_genuine_judgment_call_convenes(self):
        decision = should_convene(REBUILD)
        self.assertTrue(decision.convene)
        self.assertGreater(decision.score, CONVENE_DEFAULT.min_score)

    def test_a_verifiable_task_never_convenes_however_hard(self):
        decision = should_convene(
            Task(
                "Find the race condition causing intermittent duplicate charges and write "
                "a failing test that reproduces it.",
                "code_generation",
                0.95,
            )
        )
        self.assertFalse(decision.convene)
        self.assertIn("verifiable", decision.gates)
        self.assertIn("checkable answer", decision.reason)

    def test_judgment_phrasing_on_a_trivial_question_does_not_convene(self):
        decision = should_convene(
            Task("Should we use tabs or spaces in the new repo?", "policy", 0.15)
        )
        self.assertFalse(decision.convene)

    def test_a_non_judgment_type_is_gated_before_scoring(self):
        decision = should_convene(Task("Should we summarise this?", "summarization", 0.9))
        self.assertFalse(decision.convene)
        self.assertEqual(decision.score, 0.0)

    def test_an_unclassifiable_task_fails_closed(self):
        decision = should_convene("Handle the thing we discussed.")
        self.assertFalse(decision.convene)
        self.assertIn("unclassified", decision.gates)
        self.assertIn("not convening on a guess", decision.reason)

    def test_low_complexity_is_gated(self):
        decision = should_convene(
            Task("Should we rename the pricing page?", "strategy", 0.2)
        )
        self.assertFalse(decision.convene)
        self.assertIn("complexity", decision.gates)

    def test_a_bare_string_is_accepted(self):
        self.assertFalse(should_convene("What is the p99 latency this week?").convene)

    def test_complexity_outside_the_range_is_refused(self):
        with self.assertRaises(ValueError):
            Task("x", "strategy", 1.5)


class PolicyTests(unittest.TestCase):
    def test_never_declines_everything(self):
        self.assertFalse(should_convene(REBUILD, CONVENE_NEVER).convene)

    def test_always_convenes_everything(self):
        decision = should_convene(
            Task("Extract the emails from this thread.", "extraction", 0.1),
            CONVENE_ALWAYS,
        )
        self.assertTrue(decision.convene)
        self.assertIn("force", decision.gates)

    def test_conservative_raises_the_bar(self):
        # Scores 0.69: clears the default 0.60, misses the conservative 0.72.
        borderline = Task(
            "Should we change our pricing model? It affects the roadmap and the budget.",
            "pricing",
            0.7,
        )
        self.assertTrue(should_convene(borderline, CONVENE_DEFAULT).convene)
        self.assertFalse(should_convene(borderline, CONVENE_CONSERVATIVE).convene)

    def test_presets_are_named(self):
        self.assertEqual(sorted(CONVENING_PRESETS), ["always", "conservative", "default", "never"])

    def test_a_decision_says_why(self):
        decision = should_convene(REBUILD)
        self.assertTrue(decision.reason)
        self.assertIn("signals", decision.to_dict())
        self.assertTrue(bool(decision))


class SignalTests(unittest.TestCase):
    def test_supplied_classification_is_authoritative(self):
        sig = signals(Task("anything at all", "strategy", 0.9))
        self.assertEqual(sig.task_type, "strategy")
        self.assertEqual(sig.complexity, 0.9)
        self.assertEqual(sig.inferred, ())

    def test_inference_is_flagged_as_inference(self):
        sig = signals(Task("Should we rebuild the platform or refactor the pricing model?"))
        self.assertIn("task_type", sig.inferred)
        self.assertIn("complexity", sig.inferred)

    def test_either_or_phrasing_registers_as_ambiguity(self):
        # Regression: the convening eval caught this as a false negative.
        sig = signals(
            Task(
                "We can either hire two senior engineers or four juniors on the same "
                "budget. Which is the better bet?",
                "strategy",
                0.8,
            )
        )
        self.assertGreater(sig.ambiguity, 0.0)
        self.assertTrue(should_convene(Task(
            "We can either hire two senior engineers or four juniors on the same budget. "
            "Which is the better bet given an 18-month runway?",
            "strategy", 0.8,
        )).convene)

    def test_markers_saturate(self):
        many = signals(Task("should we " * 10, "strategy", 0.8))
        two = signals(Task("should we x. should we y.", "strategy", 0.8))
        self.assertEqual(many.ambiguity, two.ambiguity)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.council = demo_council()
        self.result = convene(
            REBUILD.prompt, self.council, mock_pool(self.council), session_id="cost-1"
        )

    def test_the_baseline_is_the_priciest_seat_by_default(self):
        self.assertEqual(
            pick_baseline_seat(self.council).model_id, "delphi-frontier"
        )
        self.assertEqual(self.result.baseline.model_id, "delphi-frontier")

    def test_the_baseline_can_be_overridden(self):
        result = Session(
            self.council,
            mock_pool(self.council),
            config=SessionConfig(baseline_model="cinder-1"),
        ).run(REBUILD.prompt, session_id="cost-2")
        self.assertEqual(result.baseline.model_id, "cinder-1")
        # A cheaper baseline makes the council look more expensive, not less.
        self.assertGreater(result.cost_multiple, self.result.cost_multiple)

    def test_an_unknown_baseline_model_is_an_error(self):
        with self.assertRaises(KeyError):
            pick_baseline_seat(self.council, "not-a-model")

    def test_the_multiple_is_reported_and_plausible(self):
        self.assertIsNotNone(self.result.cost_multiple)
        self.assertGreater(self.result.cost_multiple, 1.0)
        self.assertLess(self.result.cost_multiple, 20.0)

    def test_the_demo_council_stays_inside_the_cost_guardrail(self):
        # The spec's guardrail is a median ≤ 8x a single answer.
        self.assertLessEqual(self.result.cost_multiple, 8.0)

    def test_a_partly_priced_council_still_prices_its_baseline(self):
        # The baseline is the priciest seat, so one unpriced seat does not
        # sink the multiple — but the session still reports the bill as
        # incomplete, which is the honest pair of statements.
        council = Council(
            students=(Seat("a", "l1", ModelCost(1, 2)), Seat("b", "l2", ModelCost(1, 2))),
            arbiter=Seat("z", "lz"),
        )
        result = convene(
            "Should we rebuild or refactor?", council, mock_pool(council), session_id="cost-4"
        )
        self.assertTrue(result.baseline.priced)
        self.assertIsNotNone(result.cost_multiple)
        self.assertFalse(result.cost_is_complete)
        self.assertEqual(result.unpriced_seats, ("z",))

    def test_an_unpriced_baseline_yields_no_multiple_rather_than_a_fake_one(self):
        council = Council(
            students=(Seat("a", "l1"), Seat("b", "l2")),
            arbiter=Seat("z", "lz"),
        )
        baseline = single_model_baseline(council, [])
        self.assertFalse(baseline.priced)
        result = convene("Should we rebuild or refactor?", council, mock_pool(council),
                         session_id="cost-3")
        self.assertIsNone(result.cost_multiple)

    def test_the_baseline_is_carried_in_the_trace(self):
        closed = [e for e in self.result.events if e.event_type == "session_closed"][0]
        self.assertEqual(closed.payload["baseline_model"], "delphi-frontier")
        self.assertAlmostEqual(
            closed.payload["baseline_cost_est"], self.result.baseline.cost_est
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
