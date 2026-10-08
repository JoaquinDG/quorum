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

Point 7 was added later, and not by the Phase 1 review:

7. Malformation must measure the model's formatting, not our completion
   budget. The same conflation as point 3, one layer up: a response the
   budget cut off mid-object was repaired, then charged to the vendor by
   `malformation_by_model`. The repo's own canonical malformation fixture
   turned out to be a pure truncation, so the metric's headline example was
   its counterexample.
"""

import json
import os
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
    render_markdown,
    replay,
)
from quorum import trace as tr
from quorum.providers.recorded import (
    council_from_trace,
    recorded_session_id,
    replay_pool,
    task_of,
)

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


def _event(model, *, steps=(), truncated=False, repaired=True):
    """One scored attempt by `model`, carrying the repair record it would have
    written. Built by hand rather than provoked through a session because the
    unit under test is the derivation, and a mock cannot be made to truncate
    against a budget it never reads."""
    payload = {"model_id": model}
    if repaired:
        payload["repair"] = {
            "repaired": True,
            "truncated": truncated,
            "dropped_trailing": False,
            "steps": list(steps),
        }
    return tr.TraceEvent(
        ts=0.0, session_id="trunc", round=1, actor="Student 1",
        event_type=tr.SHEET_SUBMITTED, payload=payload,
        tokens_in=100, tokens_out=50,
    )


class TruncationIsNotMalformationTests(unittest.TestCase):
    """A budget we set is not a format the model missed.

    The same conflation as `ComplianceIsNotUptimeTests`, one layer up and
    found later. The adapter was fixed when a hard truncation came back empty
    and was recorded as a malformed sheet; a soft truncation, meaning partial
    JSON that repair closes, still reached `malformation_by_model` and was
    charged to the vendor. The metric exists to answer "can this model hold the
    format?", so an answer about our completion budget is not a smaller
    version of the right answer, it is a different question.
    """

    def test_a_repair_that_only_closed_a_truncation_is_not_malformation(self):
        rates = tr.malformation_by_model([
            _event("seat", steps=("closed_truncated_json",), truncated=True),
        ])
        rate = rates["seat"]
        self.assertEqual(rate.malformed, 0)
        self.assertEqual(rate.rate, 0.0)
        self.assertEqual(rate.truncated, 1)
        self.assertEqual(rate.truncation_rate, 1.0)

    def test_dropping_an_incomplete_trailing_value_is_also_ours(self):
        rates = tr.malformation_by_model([
            _event("seat", truncated=True, steps=(
                "closed_truncated_json", "dropped_incomplete_trailing_value",
            )),
        ])
        self.assertEqual(rates["seat"].malformed, 0)
        self.assertEqual(rates["seat"].truncated, 1)

    def test_a_fence_or_a_stray_quote_is_still_the_model_s(self):
        """The control. Without this the fix could be 'count nothing'."""
        for step in ("stripped_wrapper", "dropped_trailing_commas",
                     "escaped_inner_quotes"):
            with self.subTest(step=step):
                rates = tr.malformation_by_model([_event("seat", steps=(step,))])
                self.assertEqual(rates["seat"].malformed, 1, step)
                self.assertEqual(rates["seat"].truncated, 0, step)

    def test_a_truncation_that_also_needed_a_real_repair_counts_as_both(self):
        """Neither fact cancels the other: the model emitted a bad quote *and*
        we cut it off, so the seat is answerable for one and we for the other.
        """
        rates = tr.malformation_by_model([
            _event("seat", truncated=True, steps=(
                "escaped_inner_quotes", "closed_truncated_json",
            )),
        ])
        self.assertEqual(rates["seat"].malformed, 1)
        self.assertEqual(rates["seat"].truncated, 1)

    def test_repaired_never_exceeds_malformed(self):
        """`recovered_share` divides one by the other, so excluding truncation
        from only the denominator would push a share above 1.0."""
        rates = tr.malformation_by_model([
            _event("seat", steps=("closed_truncated_json",), truncated=True),
            _event("seat", steps=("escaped_inner_quotes",)),
        ])
        rate = rates["seat"]
        self.assertLessEqual(rate.repaired, rate.malformed)
        self.assertLessEqual(rate.recovered_share, 1.0)

    def test_a_seat_that_only_truncated_still_appears_in_the_table(self):
        """It must not vanish: a silent zero would be indistinguishable from a
        clean seat, which is the reassuring direction."""
        rates = tr.malformation_by_model([
            _event("seat", steps=("closed_truncated_json",), truncated=True),
        ])
        self.assertIn("seat", rates)

    def test_the_split_is_serialisable(self):
        rates = tr.malformation_by_model([
            _event("seat", steps=("closed_truncated_json",), truncated=True),
        ])
        payload = rates["seat"].to_dict()
        json.dumps(payload)
        self.assertEqual(payload["truncated"], 1)
        self.assertEqual(payload["malformed"], 0)


class TheRecordedIncidentIsReattributedTests(unittest.TestCase):
    """The 2026-08-19 incident, recounted.

    `malformed_seat.jsonl` is this repo's canonical malformation fixture and
    the reason the repair module exists. Replayed with repair on, it used to
    publish "Malformed responses (model-gamma): 2 of 3 (67%)", and both
    repairs were `closed_truncated_json` alone. The fixture that motivated the
    metric was the metric's own counterexample.

    Reclassified from the trace as recorded: nothing was re-run, because
    `repair.truncated` and `repair.steps` have been in the trace since repair
    shipped. Replay completeness is what made the correction free.
    """

    @staticmethod
    def _replayed():
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fixtures", "malformed_seat.jsonl",
        )
        events = tr.read_trace(path)
        council, providers = council_from_trace(events), replay_pool(events)
        result = Session(council, providers).run(
            task_of(events), session_id=recorded_session_id(events)
        )
        return replay(result.events)

    def test_the_cut_off_seat_is_no_longer_charged_with_malformation(self):
        session = self._replayed()
        gamma = session.malformation["model-gamma"]
        self.assertEqual(gamma.malformed, 0)
        self.assertEqual(gamma.truncated, 2)

    def test_the_other_seats_are_unaffected(self):
        """The fix must not have been 'stop counting'."""
        session = self._replayed()
        for model in ("model-alpha", "model-beta"):
            self.assertEqual(session.malformation[model].malformed, 0)
            self.assertEqual(session.malformation[model].truncated, 0)

    def test_the_report_calls_the_budget_ours(self):
        session = self._replayed()
        body = render_markdown(session)
        self.assertIn("Cut off by our completion budget (model-gamma)", body)
        self.assertNotIn("Malformed responses (model-gamma)", body)
