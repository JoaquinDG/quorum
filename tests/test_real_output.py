"""The schema, met by real models for the first time.

`MockProvider` emits whatever its author decided it would emit, so a parser
that has only ever seen mock output has never been tested — it has been shown
its own reflection. These fixtures are three round-1 sheets from Claude Opus,
Sonnet and Haiku answering the real `build_sheet_prompt`, captured by hand.

They earn their place twice.

**They confirmed the tolerant/strict split works.** Three sheets arrived with
three different transport quirks — a fenced block, a stray blank line inside
the JSON object, escaped quotes inside claim text — and all three parsed with
zero compliance warnings, hitting the five-claim cap exactly. The strictness
that rejects six claims and a confidence of 1.4 did not reject anything a real
model actually produced.

**They demoted the disagreement score.** All three reached the same conclusion
and the lexical metric called the question "sharply contested". That test is
below, asserting the failure, because a metric whose known limitation is only
described in prose is a metric whose limitation gets forgotten.

What these are not: a benchmark, or a blinding measurement. One lab, three
tiers, one question, orchestrated by hand.
"""

import os
import unittest

from quorum import disagreement, parse_sheet

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "real_sheets")
MODELS = ("opus", "sonnet", "haiku")


def load(name):
    with open(os.path.join(FIXTURES, f"{name}.txt"), encoding="utf-8") as handle:
        return handle.read()


class RealOutputParsesTests(unittest.TestCase):
    def test_every_real_sheet_parses(self):
        for name in MODELS:
            with self.subTest(model=name):
                sheet = parse_sheet(load(name), actor=name)
                self.assertTrue(sheet.position)
                self.assertTrue(sheet.claims)

    def test_real_models_respect_the_five_claim_cap(self):
        """The cap is the schema's core discipline and the likeliest thing for
        a real model to blow through. All three landed exactly on it."""
        for name in MODELS:
            with self.subTest(model=name):
                self.assertEqual(len(parse_sheet(load(name)).claims), 5)

    def test_no_compliance_warnings_on_any_real_sheet(self):
        """One-sentence claims are the rule enforced as a warning rather than
        an error, on the argument that hard-failing real content over a regex
        is worse than recording a rate. The rate was zero."""
        for name in MODELS:
            with self.subTest(model=name):
                self.assertEqual(parse_sheet(load(name)).compliance_warnings, ())

    def test_confidence_is_in_range_and_actually_varied(self):
        values = [parse_sheet(load(n)).confidence for n in MODELS]
        for value in values:
            self.assertTrue(0.0 <= value <= 1.0)
        self.assertGreater(len(set(values)), 1, "all three models gave the same number")

    # -- the transport quirks, each named -------------------------------

    def test_a_fenced_reply_is_tolerated(self):
        self.assertTrue(load("haiku").lstrip().startswith("```"))
        self.assertTrue(parse_sheet(load("haiku")).position)

    def test_a_stray_newline_inside_the_object_is_tolerated(self):
        self.assertIn("]\n\n, ", load("sonnet"))
        self.assertTrue(parse_sheet(load("sonnet")).position)

    def test_escaped_quotes_inside_claim_text_survive_the_brace_scanner(self):
        raw = load("opus")
        self.assertIn('\\"', raw)
        sheet = parse_sheet(raw)
        self.assertTrue(any('"' in c.text for c in sheet.claims))

    def test_every_real_sheet_used_the_nuance_field(self):
        """All three reached for the escape valve, which is evidence the
        five-claim cap does flatten something — the open question the field
        exists to answer."""
        for name in MODELS:
            with self.subTest(model=name):
                self.assertTrue(parse_sheet(load(name)).nuance)


class DisagreementScoreFailsOnRealOutputTests(unittest.TestCase):
    """Pins the finding that demoted the metric.

    If any of these start failing, the score became semantic and
    `divergence.py` and the README are stale — which is a good problem, but
    the docs must move with it.
    """

    def setUp(self):
        self.sheets = [parse_sheet(load(n)) for n in MODELS]

    def test_all_three_models_reached_the_same_conclusion(self):
        for sheet in self.sheets:
            self.assertIn("refactor in place", sheet.position.lower())

    def test_the_lexical_score_calls_that_unanimity_high_variety(self):
        result = disagreement(self.sheets)
        self.assertGreater(
            result.score,
            0.6,
            "the metric no longer misreads this; update divergence.py and the README",
        )
        self.assertIn("lexical", result.label)

    def test_claim_divergence_is_the_component_driving_it(self):
        result = disagreement(self.sheets)
        self.assertGreater(result.claim_divergence, 0.7)

    def test_the_label_no_longer_asserts_anything_about_agreement(self):
        result = disagreement(self.sheets)
        for banned in ("contested", "unanimous", "aligned", "agree"):
            self.assertNotIn(banned, result.label)
        self.assertFalse(result.measures_agreement)

    def test_the_caveat_travels_with_the_number(self):
        data = disagreement(self.sheets).to_dict()
        self.assertFalse(data["measures_agreement"])
        self.assertIn("unanimous council", data["caveat"])


class SingleLabCorrelationTests(unittest.TestCase):
    def test_three_tiers_of_one_lab_converged_on_one_answer(self):
        """Empirical support for `Council.single_lab`.

        Three models from one family, answering independently with no
        knowledge of each other, produced the same conclusion *and* similar
        reasoning shapes. That is what correlated priors look like, and it is
        why a single-lab council's agreement is weak evidence.
        """
        sheets = [parse_sheet(load(n)) for n in MODELS]
        conclusions = {"refactor" in s.position.lower() for s in sheets}
        self.assertEqual(conclusions, {True})
        # Same load-bearing consideration reached independently by all three.
        for sheet in sheets:
            blob = (sheet.position + " ".join(c.text for c in sheet.claims)).lower()
            self.assertIn("one-way door", blob)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
