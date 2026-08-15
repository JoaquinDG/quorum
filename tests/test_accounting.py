"""Regression tests for the Phase 1 adversarial review.

Every test here corresponds to a bug that shipped in the first cut and was
found by probing rather than by reading. They are grouped by the property that
was violated, because the property is what has to keep holding:

1. A session that burned extra model calls must never report a *lower* cost
   than one that didn't. Repairs were invisible: the discarded call's tokens
   were dropped on the floor, so the cost guardrail erred downward on exactly
   the paths that cost extra money.
2. A cost that omits a participant must say so. Unpriced seats contributed
   `0.0` and the total looked precise.
3. Claim-compliance must measure compliance, not uptime. A 503 was counted as
   a schema failure — the same conflation Switchboard's README warns about.
4. `nuance` must be able to reach the answer. It was excluded from grading as
   well as critique, making it write-only.
5. Two sessions must never share an id. Under a frozen or coarse clock they
   did, and `replay` then refuses the merged file — so a collision doesn't
   corrupt one report, it makes both unreadable.
6. A round-1 sheet carrying `changed_position` must be an error, not a field
   quietly ignored.
"""

import json
import unittest

from quorum import (
    CRITIQUE_PROMPT_HEADER,
    ModelCost,
    ProviderUnavailable,
    Session,
    SheetSchemaError,
    convene,
    demo_council,
    mock_pool,
    parse_sheet,
    replay,
)
from quorum import trace as tr

from test_session import (
    TASK,
    critique_json,
    default_script,
    scripted_council,
    sheet_json,
)

VAGUE = json.dumps({"objections": [{"sheet": "A", "claim_n": 1, "argument": "Agreed"}]})


def run(script, session_id):
    council, providers = scripted_council(script)
    return Session(council, providers).run(TASK, session_id=session_id), providers


class RepairsAreBilledTests(unittest.TestCase):
    """A discarded model call is still a model call."""

    def setUp(self):
        self.clean, _ = run(default_script(), "acct-clean")

    def test_a_repaired_critique_costs_more_than_a_clean_session(self):
        script = default_script()
        script["model-1"] = [
            script["model-1"][0], VAGUE, critique_json(), script["model-1"][2]
        ]
        repaired, _ = run(script, "acct-repair")

        self.assertGreater(repaired.cost_est, self.clean.cost_est)
        self.assertEqual(len(repaired.discarded_calls), 1)
        self.assertGreater(repaired.repair_cost_est, 0)

    def test_a_critic_that_fails_twice_costs_more_than_a_clean_session(self):
        script = default_script()
        script["model-1"] = [script["model-1"][0], VAGUE, VAGUE, script["model-1"][2]]
        failed, _ = run(script, "acct-fail")

        self.assertGreater(failed.cost_est, self.clean.cost_est)
        # Both wasted calls are on the record: one discarded, one absent.
        self.assertEqual(len(failed.discarded_calls), 1)
        absent = [
            e
            for e in failed.events
            if e.event_type == tr.STUDENT_ABSENT and e.payload["seat"] == 1
        ]
        self.assertEqual(len(absent), 1)
        self.assertGreater(absent[0].tokens_out, 0, "the final failed call is unbilled")

    def test_a_repaired_verdict_is_billed(self):
        script = default_script()
        script["arbiter-model"] = ["prose", script["arbiter-model"][0]]
        repaired, _ = run(script, "acct-verdict")

        self.assertEqual(len(repaired.discarded_calls), 1)
        self.assertGreater(repaired.tokens_out, self.clean.tokens_out)

    def test_a_failed_arbiter_bills_its_last_attempt(self):
        script = default_script()
        script["arbiter-model"] = ["prose", "more prose"]
        failed, _ = run(script, "acct-noverdict")

        absent = [e for e in failed.events if e.event_type == tr.ARBITER_ABSENT]
        self.assertEqual(len(absent), 1)
        self.assertGreater(absent[0].tokens_out, 0)
        self.assertIn("more prose", absent[0].payload["raw"])

    def test_discarded_attempts_survive_replay(self):
        script = default_script()
        script["model-1"] = [
            script["model-1"][0], VAGUE, critique_json(), script["model-1"][2]
        ]
        live, _ = run(script, "acct-replay")
        replayed = replay(list(live.events))

        self.assertEqual(len(replayed.discarded), 1)
        self.assertAlmostEqual(replayed.cost_est, live.cost_est)
        self.assertEqual(replayed.discarded[0]["reason"], "non_compliant_critique")

    def test_a_clean_session_discards_nothing(self):
        self.assertEqual(self.clean.discarded_calls, ())
        self.assertEqual(self.clean.repair_cost_est, 0.0)


class UnpricedSeatTests(unittest.TestCase):
    def test_an_unpriced_seat_is_named_rather_than_costed_at_zero(self):
        result, _ = run(default_script(), "acct-unpriced")
        # The scripted arbiter carries no price.
        self.assertEqual(result.unpriced_seats, ("arbiter-model",))
        self.assertFalse(result.cost_is_complete)

    def test_a_fully_priced_council_reports_a_complete_cost(self):
        council = demo_council()
        result = convene(TASK, council, mock_pool(council), session_id="acct-priced")
        self.assertEqual(result.unpriced_seats, ())
        self.assertTrue(result.cost_is_complete)

    def test_the_incomplete_bill_is_stated_in_the_trace_and_the_replay(self):
        live, _ = run(default_script(), "acct-unpriced-trace")
        replayed = replay(list(live.events))
        self.assertFalse(replayed.cost_is_complete)
        self.assertIn("arbiter-model", replayed.unpriced_seats)

    def test_model_cost_knows_whether_it_is_a_price(self):
        self.assertFalse(ModelCost().priced)
        self.assertFalse(ModelCost(0.0, 0.0).priced)
        self.assertTrue(ModelCost(0.0, 5.0).priced)


class ComplianceIsNotUptimeTests(unittest.TestCase):
    def test_a_provider_outage_does_not_count_as_a_schema_failure(self):
        script = default_script()
        script["model-3"] = [ProviderUnavailable("503")]
        script["model-1"][1] = critique_json(("A",))
        script["model-2"][1] = critique_json(("A",))
        result, _ = run(script, "acct-outage")

        self.assertEqual(result.compliance_rate, 1.0)
        self.assertEqual(result.provider_errors, 1)
        self.assertEqual(result.parse_failures, 0)

    def test_a_schema_failure_still_lowers_compliance(self):
        script = default_script()
        script["model-2"] = ["prose"] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        result, _ = run(script, "acct-schema")

        self.assertLess(result.compliance_rate, 1.0)
        self.assertEqual(result.provider_errors, 0)
        self.assertGreaterEqual(result.parse_failures, 1)


class NuanceReachesTheArbiterTests(unittest.TestCase):
    SECRET = "THE_REAL_CONSTRAINT_IS_PROCUREMENT_LEAD_TIME"

    def setUp(self):
        script = default_script()
        script["model-1"][0] = sheet_json("position of model 1", nuance=self.SECRET)
        self.result, self.providers = run(script, "acct-nuance")

    def test_nuance_reaches_the_arbiter(self):
        arbiter_prompt = [p for _, p in self.providers.get("lab-x").calls][0]
        self.assertIn(self.SECRET, arbiter_prompt)
        self.assertIn("not critiqued by anyone", arbiter_prompt)

    def test_nuance_never_reaches_a_critic(self):
        for name in self.providers.names():
            for model, prompt in self.providers.get(name).calls:
                if CRITIQUE_PROMPT_HEADER in prompt:
                    self.assertNotIn(self.SECRET, prompt)

    def test_nuance_survives_on_the_sheet_and_in_the_trace(self):
        self.assertEqual(self.result.student(1).initial.nuance, self.SECRET)
        replayed = replay(list(self.result.events))
        self.assertEqual(replayed.students[1].initial.nuance, self.SECRET)


class SessionIdTests(unittest.TestCase):
    def test_ids_are_unique_under_a_frozen_clock(self):
        council = demo_council()
        session = Session(council, mock_pool(council), clock=lambda: 1.0)
        ids = [session.run(TASK).session_id for _ in range(3)]
        self.assertEqual(len(set(ids)), 3)

    def test_a_shared_writer_never_produces_an_unreplayable_file(self):
        council = demo_council()
        session = Session(council, mock_pool(council), clock=lambda: 1.0)
        session.run(TASK)
        session.run(TASK)
        groups = dict(tr.group_sessions(list(session.writer.events)))
        self.assertEqual(len(groups), 2)
        for events in groups.values():
            replay(events)  # would raise on a collision


class RoundOneStrictnessTests(unittest.TestCase):
    def test_a_round_one_sheet_cannot_declare_a_position_change(self):
        with self.assertRaises(SheetSchemaError) as ctx:
            parse_sheet(json.loads(sheet_json("p", changed_position=True)))
        self.assertIn("changed_position", str(ctx.exception))

    def test_a_revision_may_declare_one(self):
        parsed = parse_sheet(
            json.loads(sheet_json("p", changed_position=True, because=[])),
            allow_revision_fields=True,
        )
        self.assertEqual(parsed.position, "p")

    def test_a_student_declaring_a_change_in_round_one_is_marked_absent(self):
        script = default_script()
        script["model-2"] = [
            sheet_json("position of model 2", changed_position=False)
        ] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        result, _ = run(script, "acct-r1strict")

        self.assertIsNone(result.student(2).initial)
        self.assertTrue(result.reduced_council)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class PerModelComplianceTests(unittest.TestCase):
    """Session-wide compliance hides the seat you can actually act on.

    A real three-lab run scored 69% and every failure belonged to one model;
    the other two were perfect. Averaged, that reads as a council-wide
    struggle with the format. Broken out, it reads as one replaceable seat.
    """

    def test_failures_are_attributed_to_the_model_that_caused_them(self):
        vague = json.dumps(
            {"objections": [{"sheet": "A", "claim_n": 1, "argument": "Agreed"}]}
        )
        script = default_script()
        script["model-1"] = [script["model-1"][0], vague, vague, script["model-1"][2]]
        result, _ = run(script, "per-model")

        by_model = result.compliance_by_model
        self.assertGreater(by_model["model-1"]["failures"], 0)
        self.assertEqual(by_model["model-2"]["failures"], 0)
        self.assertEqual(by_model["model-3"]["failures"], 0)
        self.assertEqual(result.worst_complier, "model-1")

    def test_a_clean_session_names_no_worst_complier(self):
        result, _ = run(default_script(), "per-model-clean")
        self.assertIsNone(result.worst_complier)
        for entry in result.compliance_by_model.values():
            self.assertEqual(entry["failures"], 0)

    def test_it_is_serialisable_in_stats(self):
        result, _ = run(default_script(), "per-model-stats")
        json.dumps(result.stats())
