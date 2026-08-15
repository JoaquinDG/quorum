"""Rounds 2 and 3 of a real session, as regression fixtures.

`test_real_output.py` covers round 1. This covers what happened when the same
three models were run through the critique and revision rounds — which found
three engine defects that mocks could not, because a mock emits what its
author decided it would emit.

Each test below pins one of those findings. If one starts failing, the
behaviour changed and the README claims need re-checking.
"""

import json
import os
import unittest

from quorum import (
    MAX_CLAIMS,
    SessionConfig,
    SheetError,
    diff_sheets,
    parse_critique,
    parse_revision,
    parse_sheet,
)

HERE = os.path.dirname(__file__)
SHEETS = os.path.join(HERE, "fixtures", "real_sheets")
SESSION = os.path.join(HERE, "fixtures", "real_session")
MODELS = ("opus", "sonnet", "haiku")


def load(directory, name):
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        return handle.read()


def blinding(round_no):
    return json.loads(load(SESSION, f"blinding_r{round_no}.json"))


class CritiqueScheamHeldTests(unittest.TestCase):
    """The hardest test available: three models that already agreed."""

    def critiques(self):
        meta = blinding(2)
        for name in MODELS:
            allowed = {k: tuple(v) for k, v in meta[name]["allowed"].items()}
            yield name, parse_critique(
                load(SESSION, f"{name}_critique.json"), allowed=allowed, actor=name
            )

    def test_every_real_critique_parses(self):
        for name, objections in self.critiques():
            with self.subTest(model=name):
                self.assertTrue(objections)

    def test_the_council_produced_far_more_than_the_minimum(self):
        total = sum(len(o) for _, o in self.critiques())
        self.assertGreaterEqual(total, 6, "the AC minimum for three students")
        self.assertGreaterEqual(total, 20, "real models were much more prolific")

    def test_every_critic_engaged_every_foreign_sheet(self):
        meta = blinding(2)
        for name, objections in self.critiques():
            with self.subTest(model=name):
                self.assertEqual(
                    {o.sheet for o in objections}, set(meta[name]["allowed"])
                )

    def test_no_objection_is_agreement_in_disguise(self):
        """All three concluded the same thing in round 1. The format gives
        them nowhere to say so, and none of them found a way around it."""
        banned = ("i agree", "agreed,", "no objection", "this is correct")
        for name, objections in self.critiques():
            for objection in objections:
                for phrase in banned:
                    self.assertNotIn(phrase, objection.argument.lower(), name)

    def test_arguments_are_far_above_the_minimum_length(self):
        for name, objections in self.critiques():
            with self.subTest(model=name):
                self.assertGreater(min(len(o.argument) for o in objections), 300)


class RevisionRepairTests(unittest.TestCase):
    """Round 3 had no repair budget until this session."""

    def test_a_six_claim_revision_is_still_rejected(self):
        six = {
            "position": "a position",
            "claims": [{"n": i, "text": f"claim {i}"} for i in range(1, 7)],
            "assumptions": ["a"], "would_change_my_mind": ["b"], "confidence": 0.6,
            "changed_position": True, "because": [],
        }
        with self.assertRaises(SheetError) as ctx:
            parse_revision(json.dumps(six), allowed={"A": (1,)}, actor="opus")
        self.assertIn(f"exceeds the cap of {MAX_CLAIMS}", str(ctx.exception))

    def test_the_round_now_has_a_repair_budget(self):
        self.assertGreaterEqual(SessionConfig().revision_repairs, 1)

    def test_the_repaired_revision_complies_and_still_changed_position(self):
        """Opus's first attempt had six claims. The repair prompt asked it to
        merge or drop one; it came back at five and still moved."""
        meta = blinding(3)
        allowed = {k: tuple(v) for k, v in meta["opus"]["allowed"].items()}
        revision = parse_revision(
            load(SESSION, "opus_revision.json"), allowed=allowed, actor="opus"
        )
        self.assertEqual(len(revision.sheet.claims), MAX_CLAIMS)
        self.assertTrue(revision.changed_position)
        self.assertGreaterEqual(len(revision.because), 1)


class RevisionDiffTests(unittest.TestCase):
    def revisions(self):
        meta = blinding(3)
        for name in MODELS:
            allowed = {k: tuple(v) for k, v in meta[name]["allowed"].items()}
            initial = parse_sheet(load(SHEETS, f"{name}.txt"))
            revision = parse_revision(
                load(SESSION, f"{name}_revision.json"), allowed=allowed, actor=name
            )
            yield name, initial, revision, diff_sheets(
                initial, revision.sheet, declared_change=revision.changed_position
            )

    def test_every_revision_parses_and_cites_objections(self):
        for name, _, revision, _ in self.revisions():
            with self.subTest(model=name):
                self.assertTrue(revision.because, "cited nothing")

    def test_holding_a_position_while_revising_claims_is_not_a_discrepancy(self):
        """Haiku held its position word for word, declared `false`, and rewrote
        four of five claims. The flag used to call that a mismatch."""
        for name, _, revision, diff in self.revisions():
            if name != "haiku":
                continue
            self.assertFalse(revision.changed_position)
            self.assertFalse(diff.position_changed)
            self.assertTrue(
                diff.declaration_matches_diff,
                "declaration_matches_diff is comparing against the wrong thing",
            )
            self.assertTrue(diff.claims_dropped or diff.claims_added)

    def test_every_model_lowered_its_confidence_under_objection(self):
        for name, _, _, diff in self.revisions():
            with self.subTest(model=name):
                self.assertLess(diff.confidence_delta, 0)

    def test_the_claim_matcher_separates_rewrites_from_untouched_claims(self):
        """Real revisions rewrite rather than reword: similarities came in at
        0.06-0.33, while an untouched claim matched at 1.00."""
        for name, _, _, diff in self.revisions():
            if name != "haiku":
                continue
            # One claim survived verbatim, so it is neither dropped nor added.
            self.assertEqual(len(diff.claims_dropped), 4)
            self.assertEqual(len(diff.claims_added), 4)


class PositionChangeRateGranularityTests(unittest.TestCase):
    def test_a_three_student_council_can_barely_land_in_the_healthy_band(self):
        """The 15-60% band is a rate across sessions. Applied to one session
        with three students, three of the four reachable values are outside it
        by arithmetic — which is why no surface presents it as a verdict."""
        reachable = [k / 3 for k in range(4)]
        inside = [r for r in reachable if 0.15 <= r <= 0.60]
        self.assertEqual(len(inside), 1)
        self.assertAlmostEqual(inside[0], 1 / 3)

    def test_this_session_scored_outside_the_band(self):
        meta = blinding(3)
        moved = 0
        for name in MODELS:
            allowed = {k: tuple(v) for k, v in meta[name]["allowed"].items()}
            initial = parse_sheet(load(SHEETS, f"{name}.txt"))
            revision = parse_revision(
                load(SESSION, f"{name}_revision.json"), allowed=allowed, actor=name
            )
            moved += diff_sheets(initial, revision.sheet).position_changed
        self.assertEqual(moved, 2)  # 67%, reported as found rather than massaged


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
