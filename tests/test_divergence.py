"""Wording-spread score and skeptic seat.

This metric was shipped as a *disagreement* score and demoted after real model
output showed it calling a unanimous council "sharply contested" (see
`test_real_output.py`). What survives is a lexical measure of vocabulary
spread, and these tests pin both of its failure directions on purpose — a
heuristic whose blind spots live only in prose is a heuristic whose blind
spots get forgotten.
"""

import json
import unittest

from quorum import (
    CRITIQUE_PROMPT_HEADER,
    SKEPTIC_INSTRUCTION,
    Session,
    SessionConfig,
    convene,
    demo_council,
    disagreement,
    leading_sheet_label,
    mock_pool,
    parse_sheet,
    replay,
)

from test_session import TASK, default_script, scripted_council, sheet_json


def sheet(position, claims, confidence=0.7):
    return parse_sheet(
        {
            "position": position,
            "claims": claims,
            "assumptions": ["something holds"],
            "would_change_my_mind": ["evidence otherwise"],
            "confidence": confidence,
        }
    )


AGREE_A = sheet("We should refactor the pipeline in place", ["Rebuilds cost two quarters"])
AGREE_B = sheet("We should refactor the pipeline in place", ["Rebuilds cost two quarters"])
DIFFERENT = sheet(
    "Instrumentation first; the bottleneck is unmeasured",
    ["Nobody has profiled where the queue forms"],
    confidence=0.4,
)


class DisagreementTests(unittest.TestCase):
    def test_identical_sheets_score_near_zero(self):
        result = disagreement([AGREE_A, AGREE_B])
        self.assertLess(result.score, 0.05)
        self.assertEqual(result.label, "near-identical wording")

    def test_different_wording_scores_high(self):
        result = disagreement([AGREE_A, DIFFERENT])
        self.assertGreater(result.score, 0.4)
        self.assertIn("lexical variety", result.label)

    def test_no_label_claims_anything_about_agreement(self):
        """The demotion, enforced. Labels describe wording; the moment one
        says "contested" again, it is asserting something the method cannot
        see."""
        for sheets in ([AGREE_A, AGREE_B], [AGREE_A, DIFFERENT]):
            label = disagreement(sheets).label
            for banned in ("contested", "unanimous", "aligned", "agree", "dissent"):
                self.assertNotIn(banned, label)

    def test_a_single_sheet_has_no_pairs_to_compare(self):
        result = disagreement([AGREE_A])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.pairs, 0)

    def test_pairs_are_counted_so_zero_is_readable(self):
        # 0.0 from "everyone agreed" and 0.0 from "there was nobody to
        # disagree with" are different facts.
        self.assertEqual(disagreement([AGREE_A, AGREE_B, DIFFERENT]).pairs, 3)

    def test_confidence_spread_moves_the_score(self):
        low = disagreement([AGREE_A, sheet(AGREE_A.position, ["Rebuilds cost two quarters"], 0.7)])
        high = disagreement([AGREE_A, sheet(AGREE_A.position, ["Rebuilds cost two quarters"], 0.1)])
        self.assertGreater(high.score, low.score)
        self.assertGreater(high.confidence_spread, 0)

    def test_components_are_exposed_for_inspection(self):
        data = disagreement([AGREE_A, DIFFERENT]).to_dict()
        for key in ("position_divergence", "claim_divergence", "confidence_spread"):
            self.assertIn(key, data)
        self.assertEqual(data["method"], "lexical")

    def test_opposite_positions_in_similar_words_score_low(self):
        """Blind spot one, constructed. Blind spot two — same conclusion,
        different words, scored as high variety — is the one that showed up
        in real output and is pinned in `test_real_output.py`."""
        yes = sheet("We should rebuild the pipeline this year", ["It is worth the cost"])
        no = sheet("We should not rebuild the pipeline this year", ["It is worth the cost"])
        result = disagreement([yes, no])
        self.assertLess(
            result.position_divergence,
            0.2,
            "if this ever rises, the score became semantic and the docs are stale",
        )

    def test_the_score_is_recorded_on_the_session_and_in_the_trace(self):
        council = demo_council()
        result = convene(TASK, council, mock_pool(council), session_id="div-1")
        self.assertIsNotNone(result.disagreement)
        self.assertGreater(result.disagreement.score, 0)
        replayed = replay(list(result.events))
        self.assertAlmostEqual(
            replayed.disagreement["score"], round(result.disagreement.score, 4)
        )

    def test_it_reaches_the_report(self):
        from quorum import render_html, render_markdown

        council = demo_council()
        result = convene(TASK, council, mock_pool(council), session_id="div-2")
        session = replay(list(result.events))
        self.assertIn("Opening wording spread", render_html(session))
        self.assertIn("Opening wording spread", render_markdown(session))
        self.assertIn("does **not** measure agreement", render_markdown(session))


class SkepticSeatTests(unittest.TestCase):
    def confidences(self, script, *, c1, c2, c3):
        for seat, confidence in ((1, c1), (2, c2), (3, c3)):
            script[f"model-{seat}"][0] = sheet_json(
                f"position of model {seat}", confidence=confidence
            )
        return script

    def run_with_skeptic(self, skeptic, *, c1=0.9, c2=0.5, c3=0.4):
        script = self.confidences(default_script(), c1=c1, c2=c2, c3=c3)
        council, providers = scripted_council(script)
        result = Session(
            council, providers, config=SessionConfig(skeptic_seat=skeptic)
        ).run(TASK, session_id=f"skeptic-{skeptic}")
        return result, providers

    def critique_prompt(self, providers, seat):
        return [
            p
            for model, p in providers.get(f"lab-{seat}").calls
            if model == f"model-{seat}" and CRITIQUE_PROMPT_HEADER in p
        ][0]

    def test_the_skeptic_is_told_to_attack_the_leading_sheet(self):
        result, providers = self.run_with_skeptic(2)
        prompt = self.critique_prompt(providers, 2)
        self.assertIn("YOU HOLD THE SKEPTIC SEAT", prompt)
        # Seat 1 is the most confident, so seat 2 must be pointed at whichever
        # label seat 1 was blinded to for it.
        label = result.blinding[2].label_for(2, 1)
        self.assertIn(f"Sheet {label} is the most confident", prompt)

    def test_only_the_skeptic_gets_the_instruction(self):
        _, providers = self.run_with_skeptic(2)
        for seat in (1, 3):
            self.assertNotIn(
                "SKEPTIC SEAT", self.critique_prompt(providers, seat)
            )

    def test_no_skeptic_by_default(self):
        council, providers = scripted_council(default_script())
        Session(council, providers).run(TASK, session_id="skeptic-off")
        for seat in (1, 2, 3):
            self.assertNotIn("SKEPTIC SEAT", self.critique_prompt(providers, seat))

    def test_the_instruction_carries_no_identity(self):
        result, providers = self.run_with_skeptic(2)
        prompt = self.critique_prompt(providers, 2)
        for model in ("model-1", "model-3", "lab-1", "lab-3", "Student 1"):
            self.assertNotIn(model, prompt)

    def test_the_skeptic_still_owes_every_sheet_an_objection(self):
        result, _ = self.run_with_skeptic(2)
        self.assertEqual(
            {o.target_seat for o in result.objections_by(2)}, {1, 3}
        )
        self.assertIn("Every sheet still needs", SKEPTIC_INSTRUCTION)

    def test_the_seat_is_recorded_in_the_trace(self):
        result, _ = self.run_with_skeptic(3)
        self.assertEqual(replay(list(result.events)).skeptic_seat, 3)

    def test_the_leading_sheet_is_the_most_confident(self):
        blinded = {
            "A": sheet("low", ["x claim"], 0.2),
            "B": sheet("high", ["y claim"], 0.9),
        }
        self.assertEqual(leading_sheet_label(blinded), "B")

    def test_ties_break_deterministically(self):
        blinded = {
            "A": sheet("one", ["x claim"], 0.5),
            "B": sheet("two", ["y claim"], 0.5),
        }
        self.assertEqual(leading_sheet_label(blinded), "A")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
