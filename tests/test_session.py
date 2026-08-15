"""Session engine tests: the happy path, and every way a participant can fail.

MockProvider always plays the protocol correctly, which makes it good for
acceptance criteria and useless for everything else. The failure cases here
run on ScriptedProvider, putting the exact bad output on the wire — a
six-claim sheet, a critique that is polite agreement, an arbiter that answers
in prose — because "the student is marked absent and the council is labelled
reduced" is a claim about behaviour under real model output, not about
exception plumbing.
"""

import json
import unittest

from quorum import (
    Council,
    CouncilError,
    ModelCost,
    ProviderPool,
    ProviderUnavailable,
    ScriptedProvider,
    Seat,
    Session,
    SessionConfig,
    convene,
    demo_council,
    mock_pool,
)
from quorum import trace as tr

TASK = "Should we rebuild the ingestion pipeline or refactor it in place?"
ARGUMENT = (
    "This claim rests on a stability record that predates the new ingest sources, "
    "so the base rate it cites no longer describes the system under discussion."
)


def sheet_json(position, *, claims=3, confidence=0.7, **extra):
    data = {
        "position": position,
        "claims": [
            {"n": i, "text": f"{position} — supporting consideration {i}"}
            for i in range(1, claims + 1)
        ],
        "assumptions": ["the current constraints hold for two quarters"],
        "would_change_my_mind": ["a measurement showing the opposite bottleneck"],
        "confidence": confidence,
    }
    data.update(extra)
    return json.dumps(data)


def critique_json(labels=("A", "B"), claim_n=1):
    return json.dumps(
        {
            "objections": [
                {"sheet": label, "claim_n": claim_n, "argument": ARGUMENT}
                for label in labels
            ]
        }
    )


def verdict_json(minority=("Student 1",)):
    return json.dumps(
        {
            "final_answer": "Refactor in place, staged over two releases.",
            "confidence_note": "Contested; it turns on the volume forecast.",
            "minority_report": [
                {"source": source, "kind": "objection", "substance": "cost curve risk"}
                for source in minority
            ],
        }
    )


def scripted_council(script, *, students=3):
    """A council of `students` scripted seats plus a scripted arbiter."""
    seats = [
        Seat(f"model-{i}", f"lab-{i}", ModelCost(1.0, 2.0))
        for i in range(1, students + 1)
    ]
    council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
    providers = ProviderPool(
        [ScriptedProvider(script, name=seat.provider) for seat in council.seats()]
    )
    return council, providers


def default_script(students=3, verdict_minority=("Student 1",)):
    script = {}
    for i in range(1, students + 1):
        script[f"model-{i}"] = [
            sheet_json(f"position of model {i}"),
            critique_json(("A", "B")[: students - 1]),
            sheet_json(f"position of model {i}", changed_position=False, because=[]),
        ]
    script["arbiter-model"] = [verdict_json(verdict_minority)]
    return script


class AcceptanceTests(unittest.TestCase):
    """The P0 acceptance criteria, run against mock providers."""

    def setUp(self):
        self.council = demo_council()
        self.result = convene(
            TASK, self.council, mock_pool(self.council), session_id="ac-1"
        )

    def test_three_answer_sheets(self):
        self.assertEqual(len(self.result.present_students), 3)
        self.assertTrue(all(s.initial is not None for s in self.result.students))

    def test_at_least_six_claim_level_objections(self):
        self.assertGreaterEqual(len(self.result.objections), 6)
        for objection in self.result.objections:
            target = self.result.student(objection.target_seat)
            self.assertIn(objection.claim_n, target.initial.claim_numbers)

    def test_three_revisions_with_structural_diffs_and_change_flags(self):
        for student in self.result.students:
            self.assertIsNotNone(student.diff)
            self.assertIsInstance(student.diff.position_changed, bool)
            self.assertTrue(student.diff.changed, "a revision that changed nothing at all")

    def test_one_verdict_with_a_minority_report_field(self):
        self.assertIsNotNone(self.result.verdict)
        self.assertIsInstance(self.result.verdict.minority_report, tuple)

    def test_council_is_not_reduced(self):
        self.assertFalse(self.result.reduced_council)
        self.assertTrue(self.result.ok)

    def test_position_change_rate_is_in_the_healthy_band(self):
        # 0% is theatre and 100% is herding; the offline demo should sit
        # between, or it proves nothing about the protocol.
        self.assertGreater(self.result.position_change_rate, 0.0)
        self.assertLess(self.result.position_change_rate, 1.0)

    def test_cost_is_accounted(self):
        self.assertGreater(self.result.cost_est, 0)
        self.assertGreater(self.result.tokens_out, 0)


class MalformedSheetTests(unittest.TestCase):
    def test_malformed_sheet_marks_the_student_absent_and_reduces_the_council(self):
        script = default_script()
        script["model-2"] = [sheet_json("six claims", claims=6)] + script["model-2"][1:]
        # Two remaining students each see exactly one other sheet.
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-malformed")

        self.assertEqual(result.council_size, 2)
        self.assertEqual(len(result.objections), 2, "the survivors still critique")
        self.assertTrue(result.reduced_council)
        self.assertIsNone(result.student(2).initial)
        self.assertIn(1, result.student(2).absent_rounds)
        reasons = [a.reason for a in result.absences]
        self.assertIn("malformed_sheet", reasons)

    def test_malformed_output_is_never_silently_coerced(self):
        script = default_script()
        script["model-2"] = ["I would rather answer in prose."] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-prose")

        self.assertIsNone(result.student(2).initial)
        self.assertIsNone(result.student(2).final)
        # The raw text survives in the trace: "six claims" and "an apology"
        # are different bugs.
        absent = [e for e in result.events if e.event_type == tr.STUDENT_ABSENT]
        self.assertIn("prose", absent[0].payload["raw"])

    def test_provider_outage_marks_the_student_absent(self):
        script = default_script()
        script["model-3"] = [ProviderUnavailable("503")]
        script["model-1"][1] = critique_json(("A",))
        script["model-2"][1] = critique_json(("A",))
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-outage")

        self.assertEqual(result.council_size, 2)
        self.assertEqual(
            [a.reason for a in result.absences if a.seat == 3], ["provider_error"]
        )
        self.assertTrue(result.ok, "a two-student session still produces a verdict")

    def test_council_below_the_minimum_closes_without_a_verdict(self):
        script = default_script()
        script["model-2"] = ["nonsense"] + script["model-2"][1:]
        script["model-3"] = ["also nonsense"] + script["model-3"][1:]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-lonely")

        self.assertFalse(result.ok)
        self.assertIsNone(result.verdict)
        self.assertIn("peer review", result.failed_reason)
        self.assertEqual(result.council_size, 1)


class NonCompliantCritiqueTests(unittest.TestCase):
    def test_a_repaired_critique_is_accepted_on_the_second_try(self):
        script = default_script()
        script["model-1"] = [
            script["model-1"][0],
            json.dumps({"objections": [{"sheet": "A", "argument": "I agree."}]}),
            critique_json(),  # the repair prompt gets a compliant answer
            script["model-1"][2],
        ]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-repair")

        self.assertEqual(len(result.objections_by(1)), 2)
        self.assertEqual([a for a in result.absences if a.seat == 1], [])
        self.assertLess(result.compliance_rate, 1.0, "the failed try is still counted")

    def test_a_critic_that_cannot_comply_twice_is_marked_non_compliant(self):
        vague = json.dumps(
            {"objections": [{"sheet": "A", "claim_n": 1, "argument": "Looks good"}]}
        )
        script = default_script()
        script["model-1"] = [
            script["model-1"][0],
            vague,
            vague,
            script["model-1"][2],
        ]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-vague")

        self.assertEqual(result.objections_by(1), ())
        self.assertEqual(
            [a.reason for a in result.absences if a.seat == 1],
            ["non_compliant_critique"],
        )

    def test_a_non_compliant_critic_still_answers_objections_against_it(self):
        vague = json.dumps(
            {"objections": [{"sheet": "A", "claim_n": 1, "argument": "Agreed"}]}
        )
        script = default_script()
        script["model-1"] = [script["model-1"][0], vague, vague, script["model-1"][2]]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-vague-2")

        # Absence is scoped to the round it happened in.
        self.assertEqual(result.student(1).absent_rounds, (2,))
        self.assertIsNotNone(result.student(1).diff)
        self.assertGreater(len(result.objections_against(1)), 0)

    def test_no_repairs_configured_means_one_shot(self):
        vague = json.dumps(
            {"objections": [{"sheet": "A", "claim_n": 1, "argument": "Agreed"}]}
        )
        script = default_script()
        script["model-1"] = [script["model-1"][0], vague, script["model-1"][2]]
        council, providers = scripted_council(script)

        result = Session(
            council, providers, config=SessionConfig(critique_repairs=0)
        ).run(TASK, session_id="s-oneshot")

        self.assertEqual(result.objections_by(1), ())


class RevisionTests(unittest.TestCase):
    def test_a_declared_change_with_an_identical_sheet_is_flagged(self):
        script = default_script()
        script["model-1"][2] = sheet_json(
            "position of model 1", changed_position=True, because=[]
        )
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-declared")

        diff = result.student(1).diff
        self.assertTrue(result.student(1).declared_change)
        self.assertFalse(diff.position_changed)
        self.assertFalse(diff.declaration_matches_diff)
        self.assertEqual(result.position_change_rate, 0.0)

    def test_a_failed_revision_leaves_the_opening_sheet_standing(self):
        script = default_script()
        script["model-2"][2] = "no thanks"
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-norevision")

        student = result.student(2)
        self.assertEqual(student.final, student.initial)
        self.assertIsNone(student.diff)
        self.assertEqual(student.absent_rounds, (3,))
        self.assertTrue(result.ok)

    def test_a_real_position_change_is_recorded_with_its_citation(self):
        council, providers = scripted_council(default_script())
        # Rewrite model-1's revision to genuinely move, citing whichever critic
        # actually objected to it.
        result = Session(council, providers).run(TASK, session_id="s-move-probe")
        critic_label = result.blinding[3].label_for(
            1, result.objections_against(1)[0].critic_seat
        )
        claim_n = result.objections_against(1)[0].claim_n

        script = default_script()
        script["model-1"][2] = sheet_json(
            "a genuinely different position after argument",
            changed_position=True,
            because=[{"critic": critic_label, "claim_n": claim_n}],
        )
        council, providers = scripted_council(script)
        result = Session(council, providers).run(TASK, session_id="s-move-probe")

        student = result.student(1)
        self.assertTrue(student.changed_position)
        self.assertTrue(student.diff.declaration_matches_diff)
        self.assertEqual(student.because[0].critic, critic_label)
        changes = [e for e in result.events if e.event_type == tr.POSITION_CHANGED]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].payload["because"][0]["critic_seat"],
                         result.objections_against(1)[0].critic_seat)


class ArbiterTests(unittest.TestCase):
    def test_an_unparseable_verdict_is_retried_once_then_the_session_closes(self):
        script = default_script()
        script["arbiter-model"] = ["I'd rather summarise in prose.", "still prose"]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-noverdict")

        self.assertIsNone(result.verdict)
        self.assertFalse(result.ok)
        self.assertIn("arbiter produced no valid verdict", result.failed_reason)
        self.assertEqual(
            len([e for e in result.events if e.event_type == tr.ARBITER_ABSENT]), 1
        )

    def test_a_repaired_verdict_is_accepted(self):
        script = default_script()
        script["arbiter-model"] = ["prose", verdict_json()]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-verdict-repair")

        self.assertIsNotNone(result.verdict)
        self.assertTrue(result.ok)

    def test_dissent_attributed_to_a_nonexistent_student_is_rejected(self):
        script = default_script(verdict_minority=("Student 9",))
        script["arbiter-model"] = [script["arbiter-model"][0], "prose"]
        council, providers = scripted_council(script)

        result = Session(council, providers).run(TASK, session_id="s-badsource")

        self.assertIsNone(result.verdict)

    def test_minority_items_are_traced_and_resolved_to_a_model(self):
        council, providers = scripted_council(default_script())
        result = Session(council, providers).run(TASK, session_id="s-minority")

        recorded = [e for e in result.events if e.event_type == tr.MINORITY_RECORDED]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].payload["source_seat"], 1)
        self.assertEqual(recorded[0].payload["source_model"], "model-1")
        self.assertTrue(result.dissent_preserved)

    def test_an_empty_minority_report_is_legal(self):
        council, providers = scripted_council(default_script(verdict_minority=()))
        result = Session(council, providers).run(TASK, session_id="s-consensus")

        self.assertTrue(result.ok)
        self.assertFalse(result.dissent_preserved)


class CouncilValidationTests(unittest.TestCase):
    def test_the_arbiter_cannot_hold_a_student_seat(self):
        with self.assertRaises(CouncilError) as ctx:
            Council(
                students=(Seat("a", "lab-a"), Seat("b", "lab-b")),
                arbiter=Seat("a", "lab-a"),
            )
        self.assertIn("took part in", str(ctx.exception))

    def test_duplicate_students_are_rejected(self):
        with self.assertRaises(CouncilError):
            Council(
                students=(Seat("a", "lab-a"), Seat("a", "lab-a")),
                arbiter=Seat("z", "lab-z"),
            )

    def test_a_fourth_student_is_rejected(self):
        with self.assertRaises(CouncilError):
            Council(
                students=tuple(Seat(f"m{i}", f"l{i}") for i in range(4)),
                arbiter=Seat("z", "lab-z"),
            )

    def test_a_lone_student_is_rejected(self):
        with self.assertRaises(CouncilError):
            Council(students=(Seat("a", "lab-a"),), arbiter=Seat("z", "lab-z"))

    def test_a_missing_provider_fails_before_the_session_starts(self):
        council = demo_council()
        with self.assertRaises(KeyError):
            Session(council, ProviderPool([]))

    def test_two_student_council_runs(self):
        script = default_script(students=2)
        for i in (1, 2):
            script[f"model-{i}"][1] = critique_json(("A",))
        council, providers = scripted_council(script, students=2)

        result = Session(council, providers).run(TASK, session_id="s-two")

        self.assertTrue(result.ok)
        self.assertFalse(result.reduced_council, "two seated, two answered")
        self.assertEqual(len(result.objections), 2)


class SessionShapeTests(unittest.TestCase):
    def test_an_empty_question_is_refused(self):
        council = demo_council()
        with self.assertRaises(ValueError):
            Session(council, mock_pool(council)).run("   ")

    def test_session_ids_are_stable_when_supplied(self):
        council = demo_council()
        first = convene(TASK, council, mock_pool(council), session_id="fixed")
        second = convene(TASK, council, mock_pool(council), session_id="fixed")
        self.assertEqual(first.session_id, second.session_id)
        self.assertEqual(
            [o.argument for o in first.objections],
            [o.argument for o in second.objections],
        )

    def test_stats_are_serialisable(self):
        council = demo_council()
        result = convene(TASK, council, mock_pool(council), session_id="stats")
        json.dumps(result.stats())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
