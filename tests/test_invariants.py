"""Protocol invariants — one test per structural guarantee.

Everything Quorum claims about trustworthiness reduces to five properties. If
any of them can be broken by a refactor without a test going red, the claim is
marketing:

1. **Independence** — round 1 shows a student nothing but the question.
2. **Schema blinding** — round 2 carries no identity signal: no model names,
   no provider names, no seat numbers, and no `nuance` field.
3. **No self-grading** — the arbiter never debated, and is not called until
   the debate is over. No student critiques its own sheet.
4. **Fail-closed** — a participant that errors is reported absent, and its
   absence is visible everywhere the verdict is.
5. **Fixed rounds** — no "keep going until they agree" loop.

Plus the trace's own invariant: it must carry everything a renderer needs,
which `test_trace_replay` proves by rebuilding the session from it.
"""

import unittest

import json

from quorum import (
    CRITIQUE_PROMPT_HEADER,
    REVISION_PROMPT_HEADER,
    SHEET_PROMPT_HEADER,
    VERDICT_PROMPT_HEADER,
    Council,
    ModelCost,
    ProviderPool,
    ScriptedProvider,
    Seat,
    Session,
    SessionConfig,
    build_sheet_prompt,
    convene,
    demo_council,
    mock_pool,
)
from quorum import trace as tr

from test_session import TASK, critique_json, default_script, sheet_json


def run_with_recording_mocks():
    """Run the demo council and hand back the provider objects, which record
    every (model_id, prompt) pair they were asked to complete."""
    council = demo_council()
    pool = mock_pool(council)
    session = Session(council, pool)
    result = session.run(TASK, session_id="inv-1")
    providers = {name: pool.get(name) for name in pool.names()}
    return council, result, providers


def prompts_for(providers, seat):
    return [prompt for model, prompt in providers[seat.provider].calls if model == seat.model_id]


class IndependenceTests(unittest.TestCase):
    """Anti-sycophancy mechanism #1: nobody sees anybody in round 1."""

    def setUp(self):
        self.council, self.result, self.providers = run_with_recording_mocks()

    def test_round_one_prompt_is_exactly_the_question(self):
        expected = build_sheet_prompt(TASK)
        for seat in self.council.students:
            self.assertEqual(prompts_for(self.providers, seat)[0], expected)

    def test_round_one_prompt_contains_no_peer_content(self):
        positions = [s.initial.position for s in self.result.students]
        for seat in self.council.students:
            first = prompts_for(self.providers, seat)[0]
            for position in positions:
                self.assertNotIn(position, first)

    def test_every_student_answers_before_any_critique_is_issued(self):
        kinds = [
            e.event_type
            for e in self.result.events
            if e.event_type in (tr.SHEET_SUBMITTED, tr.OBJECTION_RAISED)
        ]
        first_objection = kinds.index(tr.OBJECTION_RAISED)
        self.assertEqual(
            kinds[:first_objection].count(tr.SHEET_SUBMITTED), len(self.council.students)
        )


class BlindingInvariantTests(unittest.TestCase):
    """Anti-sycophancy mechanism #2: critique the claim, not the author."""

    def setUp(self):
        self.council, self.result, self.providers = run_with_recording_mocks()
        self.critique_prompts = [
            prompt
            for seat in self.council.students
            for prompt in prompts_for(self.providers, seat)
            if CRITIQUE_PROMPT_HEADER in prompt
        ]
        self.assertEqual(len(self.critique_prompts), 3)

    def test_no_model_or_provider_names_appear_in_a_critique_prompt(self):
        names = [s.model_id for s in self.council.seats()]
        names += [s.provider for s in self.council.seats()]
        for prompt in self.critique_prompts:
            for name in names:
                self.assertNotIn(name, prompt, f"{name!r} leaked into a critique prompt")

    def test_no_seat_numbers_appear_in_a_critique_prompt(self):
        for prompt in self.critique_prompts:
            for seat in self.council.student_seats():
                self.assertNotIn(f"Student {seat}", prompt)

    def test_a_critic_never_receives_its_own_sheet(self):
        for seat_no in self.council.student_seats():
            own = self.result.student(seat_no).initial.position
            seat = self.council.student(seat_no)
            for prompt in prompts_for(self.providers, seat):
                if CRITIQUE_PROMPT_HEADER in prompt:
                    self.assertNotIn(own, prompt)

    def test_a_critic_never_objects_to_its_own_claims(self):
        for objection in self.result.objections:
            self.assertNotEqual(objection.critic_seat, objection.target_seat)

    def test_nuance_is_never_shown_to_another_student(self):
        """The blinding invariant covers *students*, not the arbiter.

        `nuance` holds the prose the schema could not carry, which makes it the
        likeliest fingerprint and the one thing nobody is allowed to object to
        — so no critic may see it. The arbiter does (see
        `test_accounting.NuanceReachesTheArbiterTests`): excluding it there
        too made the field write-only, unable to influence the answer it
        exists to inform. That is a deliberate, stated exposure, not an
        oversight — nuance is the one unblinded channel to the grader.
        """
        secret = "PROCUREMENT_TIMELINE_IS_THE_REAL_CONSTRAINT"
        script = default_script()
        script["model-1"][0] = sheet_json("position of model 1", nuance=secret)
        seats = [Seat(f"model-{i}", f"lab-{i}", ModelCost(1, 2)) for i in (1, 2, 3)]
        council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
        pool = ProviderPool(
            [ScriptedProvider(script, name=s.provider) for s in council.seats()]
        )
        session = Session(council, pool)
        result = session.run(TASK, session_id="inv-nuance")

        self.assertEqual(result.student(1).initial.nuance, secret)
        student_models = {s.model_id for s in council.students}
        for name in pool.names():
            for model, prompt in pool.get(name).calls:
                if model not in student_models:
                    continue
                if model == "model-1" and REVISION_PROMPT_HEADER in prompt:
                    continue  # its own sheet, echoed back to it
                self.assertNotIn(secret, prompt, f"nuance leaked to {model}")

    def test_round_three_relabels_the_critics(self):
        self.assertNotEqual(self.result.blinding[2].salt, self.result.blinding[3].salt)

    def test_revision_prompts_name_critics_not_seats_or_models(self):
        for seat in self.council.students:
            for prompt in prompts_for(self.providers, seat):
                if REVISION_PROMPT_HEADER not in prompt:
                    continue
                for other in self.council.seats():
                    if other.model_id == seat.model_id:
                        continue
                    self.assertNotIn(other.model_id, prompt)
                    self.assertNotIn(other.provider, prompt)


class NoSelfGradingTests(unittest.TestCase):
    def setUp(self):
        self.council, self.result, self.providers = run_with_recording_mocks()

    def test_the_arbiter_holds_no_student_seat(self):
        self.assertNotIn(
            self.council.arbiter.model_id, [s.model_id for s in self.council.students]
        )

    def test_the_arbiter_is_called_exactly_once_and_only_to_grade(self):
        calls = prompts_for(self.providers, self.council.arbiter)
        self.assertEqual(len(calls), 1)
        self.assertIn(VERDICT_PROMPT_HEADER, calls[0])

    def test_the_arbiter_never_sees_model_identities(self):
        prompt = prompts_for(self.providers, self.council.arbiter)[0]
        for seat in self.council.students:
            self.assertNotIn(seat.model_id, prompt)
            self.assertNotIn(seat.provider, prompt)

    def test_the_arbiter_sees_every_participant_and_both_of_their_sheets(self):
        prompt = prompts_for(self.providers, self.council.arbiter)[0]
        for student in self.result.students:
            self.assertIn(student.label, prompt)
            self.assertIn(student.initial.position, prompt)
            self.assertIn(student.final.position, prompt)


class FailClosedTests(unittest.TestCase):
    def test_an_absent_student_contributes_nothing_to_the_verdict_briefing(self):
        script = default_script()
        script["model-2"] = ["prose, not a sheet"] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        seats = [Seat(f"model-{i}", f"lab-{i}", ModelCost(1, 2)) for i in (1, 2, 3)]
        council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
        pool = ProviderPool(
            [ScriptedProvider(script, name=s.provider) for s in council.seats()]
        )
        result = Session(council, pool).run(TASK, session_id="inv-failclosed")

        verdict_prompt = [
            p for _, p in pool.get("lab-x").calls if VERDICT_PROMPT_HEADER in p
        ][0]
        self.assertNotIn("Student 2", verdict_prompt)
        self.assertTrue(result.reduced_council)

    def test_the_reduced_council_is_stated_on_the_verdict_event(self):
        script = default_script()
        script["model-2"] = ["prose"] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        seats = [Seat(f"model-{i}", f"lab-{i}", ModelCost(1, 2)) for i in (1, 2, 3)]
        council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
        pool = ProviderPool(
            [ScriptedProvider(script, name=s.provider) for s in council.seats()]
        )
        result = Session(council, pool).run(TASK, session_id="inv-reduced")

        verdicts = [e for e in result.events if e.event_type == tr.VERDICT_DELIVERED]
        self.assertTrue(verdicts[0].payload["reduced_council"])
        self.assertEqual(verdicts[0].payload["council_size"], 2)
        closed = [e for e in result.events if e.event_type == tr.SESSION_CLOSED][0]
        self.assertTrue(closed.payload["reduced_council"])


class FixedRoundTests(unittest.TestCase):
    def test_a_clean_session_makes_exactly_one_call_per_student_per_round(self):
        council, result, providers = run_with_recording_mocks()
        total = sum(len(providers[name].calls) for name in providers)
        self.assertEqual(total, 3 * len(council.students) + 1)

    def test_repairs_are_bounded_by_config(self):
        vague = '{"objections": [{"sheet": "A", "claim_n": 1, "argument": "Agreed"}]}'
        script = default_script()
        script["model-1"] = [script["model-1"][0]] + [vague] * 8 + [script["model-1"][2]]
        seats = [Seat(f"model-{i}", f"lab-{i}", ModelCost(1, 2)) for i in (1, 2, 3)]
        council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
        pool = ProviderPool(
            [ScriptedProvider(script, name=s.provider) for s in council.seats()]
        )
        Session(council, pool).run(TASK, session_id="inv-bounded")

        # The full per-round budget, spelled out so a change to any of them
        # shows up here rather than as a mystery call count:
        #   round 1  sheet                        1 call
        #   round 2  critique + 1 repair          2 calls
        #   round 3  revision + 1 repair          2 calls
        # The script feeds non-compliant text throughout, so every repair the
        # config allows is taken and none beyond it.
        config = SessionConfig()
        expected = 1 + (1 + config.critique_repairs) + (1 + config.revision_repairs)
        self.assertEqual(len(pool.get("lab-1").calls), expected)

    def test_a_revision_repair_is_offered_exactly_once(self):
        """Round 3 had no repair budget until a real model's revision was
        rejected for carrying six claims. One repair, then absent."""
        six_claims = json.dumps({
            "position": "a position",
            "claims": [{"n": i, "text": f"claim {i} text"} for i in range(1, 7)],
            "assumptions": ["a"], "would_change_my_mind": ["b"], "confidence": 0.6,
            "changed_position": False, "because": [],
        })
        script = default_script()
        script["model-1"] = [script["model-1"][0], critique_json(), six_claims, six_claims]
        seats = [Seat(f"model-{i}", f"lab-{i}", ModelCost(1, 2)) for i in (1, 2, 3)]
        council = Council(students=tuple(seats), arbiter=Seat("arbiter-model", "lab-x"))
        pool = ProviderPool(
            [ScriptedProvider(script, name=s.provider) for s in council.seats()]
        )
        result = Session(council, pool).run(TASK, session_id="inv-revrepair")

        self.assertEqual(len(pool.get("lab-1").calls), 4)  # sheet, critique, revision, repair
        self.assertEqual(result.student(1).absent_rounds, (3,))
        discarded = [
            e for e in result.events
            if e.event_type == tr.ATTEMPT_DISCARDED and e.payload["seat"] == 1
        ]
        self.assertEqual(len(discarded), 1)
        self.assertIn("exceeds the cap", discarded[0].payload["detail"])

    def test_no_round_beyond_grading_is_ever_traced(self):
        _, result, _ = run_with_recording_mocks()
        self.assertLessEqual(max(e.round for e in result.events), 4)


class ObjectionIntegrityTests(unittest.TestCase):
    def test_every_objection_targets_a_claim_that_exists(self):
        _, result, _ = run_with_recording_mocks()
        for objection in result.objections:
            target = result.student(objection.target_seat)
            self.assertIn(objection.claim_n, target.initial.claim_numbers)

    def test_every_objection_engages_a_claim_at_length(self):
        _, result, _ = run_with_recording_mocks()
        for objection in result.objections:
            self.assertGreaterEqual(len(objection.argument), 40)

    def test_every_student_objects_to_every_other_student(self):
        council, result, _ = run_with_recording_mocks()
        for critic in council.student_seats():
            targets = {o.target_seat for o in result.objections_by(critic)}
            self.assertEqual(targets, set(council.student_seats()) - {critic})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
