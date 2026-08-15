"""Deanonymization probe tests.

The probe is the project's honesty mechanism, so the tests are mostly about
the ways a probe can quietly report a flattering number: a participant probing
itself, an abstention counted as a miss, a guess naming a model that never sat
down, or a chance baseline computed to make the excess look small.

The accuracy assertion is deliberately loose. `MockProvider` permutes the
roster from a hash, so it lands near chance with real sampling noise, and a
tight bound would make this test flaky rather than meaningful. The claim being
tested is "the harness measures something near chance when there is nothing to
find", not a specific number.
"""

import unittest

from quorum import (
    ModelCost,
    ProviderPool,
    ProviderUnavailable,
    ScriptedProvider,
    Seat,
    TraceWriter,
    convene,
    demo_council,
    mock_pool,
)
from quorum import trace as tr
from quorum.probe import ProbeError, ProbeReport, ProbeResult, probe_all, probe_session
from quorum.providers.base import MockProvider

QUESTION = "Should we rebuild the ingestion pipeline or refactor it in place?"
PROBER = Seat("probe-critic", "probelab", ModelCost(1.0, 5.0))


def pool_with_prober(council, prober_provider=None):
    pool = mock_pool(council)
    provider = prober_provider or MockProvider()
    provider.name = PROBER.provider
    pool._providers[PROBER.provider] = provider  # noqa: SLF001
    return pool


def a_session(session_id="probe-1", council=None):
    council = council or demo_council()
    return convene(QUESTION, council, mock_pool(council), session_id=session_id)


class ProbeIntegrityTests(unittest.TestCase):
    def test_a_participant_cannot_probe_its_own_session(self):
        council = demo_council()
        result = a_session(council=council)
        with self.assertRaises(ProbeError) as ctx:
            probe_session(result, council.students[0], pool_with_prober(council))
        self.assertIn("took part", str(ctx.exception))

    def test_the_arbiter_cannot_probe_either(self):
        council = demo_council()
        result = a_session(council=council)
        with self.assertRaises(ProbeError):
            probe_session(result, council.arbiter, pool_with_prober(council))

    def test_a_probe_needs_at_least_two_sheets(self):
        from test_session import default_script, scripted_council, critique_json

        script = default_script()
        for seat in (2, 3):
            script[f"model-{seat}"] = ["prose"] + script[f"model-{seat}"][1:]
        council, providers = scripted_council(script)
        from quorum import Session

        result = Session(council, providers).run(QUESTION, session_id="probe-lonely")
        pool = ProviderPool([ScriptedProvider({}, name=PROBER.provider, default="{}")])
        with self.assertRaises(ProbeError):
            probe_session(result, PROBER, pool)

    def test_the_probe_sees_only_anonymised_sheets(self):
        council = demo_council()
        result = a_session(council=council)
        provider = MockProvider()
        pool = pool_with_prober(council, provider)
        probe_session(result, PROBER, pool)

        prompt = provider.calls[0][1]
        # The roster is given on purpose — withholding it would understate the
        # leak — but nothing may say which sheet came from which model.
        for student in result.students:
            self.assertIn(student.model_id, prompt)  # roster
            self.assertNotIn(f"Student {student.seat}", prompt)
        # Each model name appears exactly once — in the roster — and never
        # adjacent to a sheet label, which is what would give the answer away.
        for student in result.students:
            self.assertEqual(prompt.count(student.model_id), 1)

    def test_probe_labels_are_independent_of_the_session_blinding(self):
        council = demo_council()
        result = a_session(council=council)
        provider = MockProvider()
        probe_session(result, PROBER, pool_with_prober(council, provider))
        prompt = provider.calls[0][1]
        for salt in ("r2-sheets", "r3-critics"):
            self.assertNotIn(salt, prompt)


class ScoringTests(unittest.TestCase):
    def scripted(self, reply):
        council = demo_council()
        result = a_session(council=council)
        pool = mock_pool(council)
        pool._providers[PROBER.provider] = ScriptedProvider(  # noqa: SLF001
            {PROBER.model_id: [reply]}, name=PROBER.provider
        )
        return result, probe_session(result, PROBER, pool)

    def test_a_correct_guess_scores(self):
        council = demo_council()
        result = a_session(council=council)
        # Build a reply that names the true author for every label by reading
        # the labelling the probe itself will produce.
        provider = MockProvider()
        probe = probe_session(result, PROBER, pool_with_prober(council, provider))
        perfect = {g.label: g.actual_model for g in probe.guesses}
        import json

        pool = mock_pool(council)
        pool._providers[PROBER.provider] = ScriptedProvider(  # noqa: SLF001
            {PROBER.model_id: [json.dumps({"guesses": perfect})]}, name=PROBER.provider
        )
        scored = probe_session(result, PROBER, pool)
        self.assertEqual(scored.accuracy, 1.0)
        self.assertAlmostEqual(scored.chance, 1 / 3)

    def test_a_guess_naming_a_model_that_did_not_take_part_is_dropped(self):
        _, probe = self.scripted('{"guesses": {"A": "gpt-nonexistent"}}')
        self.assertEqual(probe.attempts, 0)
        self.assertTrue(probe.abstained)

    def test_an_unparseable_reply_abstains_rather_than_scoring_zero(self):
        _, probe = self.scripted("I would rather not speculate.")
        self.assertTrue(probe.abstained)
        self.assertEqual(probe.attempts, 0)
        self.assertIn("unparseable", probe.detail)

    def test_a_provider_error_abstains(self):
        council = demo_council()
        result = a_session(council=council)
        pool = mock_pool(council)
        pool._providers[PROBER.provider] = ScriptedProvider(  # noqa: SLF001
            {PROBER.model_id: [ProviderUnavailable("503")]}, name=PROBER.provider
        )
        probe = probe_session(result, PROBER, pool)
        self.assertTrue(probe.abstained)
        self.assertIn("provider error", probe.detail)

    def test_a_partial_reply_scores_only_what_it_answered(self):
        council = demo_council()
        result = a_session(council=council)
        roster = sorted(s.model_id for s in result.students)
        import json

        pool = mock_pool(council)
        pool._providers[PROBER.provider] = ScriptedProvider(  # noqa: SLF001
            {PROBER.model_id: [json.dumps({"guesses": {"A": roster[0]}})]},
            name=PROBER.provider,
        )
        probe = probe_session(result, PROBER, pool)
        self.assertEqual(probe.attempts, 1)
        self.assertFalse(probe.abstained)


class ReportTests(unittest.TestCase):
    def test_abstentions_stay_out_of_the_denominator(self):
        report = ProbeReport(
            results=[
                ProbeResult("s1", "p", (), ("a", "b"), abstained=True),
                ProbeResult("s2", "p", (), ("a", "b"), abstained=True),
            ]
        )
        self.assertEqual(report.attempts, 0)
        self.assertEqual(report.accuracy, 0.0)
        self.assertEqual(report.abstentions, 2)
        self.assertIn("no scored guesses", report.summary())

    def test_chance_is_one_over_the_roster(self):
        council = demo_council()
        probe = probe_session(a_session(), PROBER, pool_with_prober(council))
        self.assertAlmostEqual(probe.chance, 1 / 3)

    def test_the_mock_prober_lands_near_chance(self):
        council = demo_council()
        pool = pool_with_prober(council)
        results = [
            convene(f"Should we do A{i} or B{i}, given cost and risk?", council,
                    mock_pool(council), session_id=f"probe-batch-{i}")
            for i in range(30)
        ]
        report = probe_all(results, PROBER, pool)

        self.assertEqual(report.scored_sessions, 30)
        self.assertGreater(report.attempts, 80)
        # Loose by design: the mock cannot detect authorship, so this asserts
        # "near chance with sampling noise", not a number.
        self.assertLess(
            abs(report.excess_over_chance),
            0.20,
            f"mock prober is not near chance: {report.summary()}",
        )

    def test_the_summary_reports_n_beside_the_rate(self):
        council = demo_council()
        report = probe_all([a_session()], PROBER, pool_with_prober(council))
        summary = report.summary()
        self.assertIn("chance", summary)
        self.assertIn("guesses", summary)
        self.assertIn("session", summary)


class TraceTests(unittest.TestCase):
    def test_a_probe_emits_a_probe_result_event(self):
        council = demo_council()
        result = a_session(council=council)
        writer = TraceWriter(clock=lambda: 1.0)
        probe = probe_session(result, PROBER, pool_with_prober(council), writer=writer)

        events = [e for e in writer.events if e.event_type == tr.PROBE_RESULT]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].session_id, result.session_id)
        self.assertEqual(events[0].actor, "prober")
        self.assertEqual(events[0].payload["accuracy"], round(probe.accuracy, 4))
        self.assertGreater(events[0].cost_est, 0, "the probe is a real call and costs")

    def test_probe_events_survive_replay(self):
        from quorum import replay

        council = demo_council()
        result = a_session(council=council)
        writer = TraceWriter(clock=lambda: 1.0)
        probe_session(result, PROBER, pool_with_prober(council), writer=writer)
        replayed = replay(list(result.events) + list(writer.events))
        self.assertEqual(len(replayed.probes), 1)
        self.assertIn("guesses", replayed.probes[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class FailedMeasurementIsNotAPassTests(unittest.TestCase):
    """Zero scored guesses must never read as "the blinding held".

    A real run had every prober call fail on a token budget. Accuracy and
    chance were both 0.0, so the excess was 0.0, and the verdict logic printed
    "at or below chance — no leak detected". The most reassuring possible
    output from a measurement that did not happen.
    """

    def test_a_report_with_no_guesses_knows_it_measured_nothing(self):
        report = ProbeReport(results=[
            ProbeResult("s1", "p", (), ("a", "b", "c"), abstained=True, detail="budget"),
            ProbeResult("s2", "p", (), ("a", "b", "c"), abstained=True, detail="budget"),
        ])
        self.assertFalse(report.measured)
        self.assertEqual(report.excess_over_chance, 0.0)  # the trap
        self.assertIn("no scored guesses", report.summary())
        self.assertFalse(report.to_dict()["measured"])

    def test_a_report_with_guesses_is_measured(self):
        from quorum.probe import Guess

        report = ProbeReport(results=[
            ProbeResult("s1", "p", (Guess("A", "a", "a", 1),), ("a", "b", "c")),
        ])
        self.assertTrue(report.measured)
        self.assertTrue(report.to_dict()["measured"])
