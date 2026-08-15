"""Answer-sheet schema tests.

The split under test is "tolerant of format, strict about meaning". These
cases are written to hold that line from both sides: a fenced, prose-wrapped
sheet with a stringly-typed confidence must parse, and a sheet with six claims
or a confidence of 1.4 must not — no clamping, no truncation, no silent
repair, because every one of those turns a schema violation into a number the
final report will quote.
"""

import json
import unittest

from quorum import (
    MAX_CLAIMS,
    NonCompliantCritique,
    SheetParseError,
    SheetSchemaError,
    diff_sheets,
    extract_json,
    parse_critique,
    parse_revision,
    parse_sheet,
    parse_verdict,
)

VALID = {
    "position": "We should refactor the pipeline in place rather than rebuild it",
    "claims": [
        {"n": 1, "text": "A rebuild costs two quarters the roadmap cannot absorb"},
        {"n": 2, "text": "The current schema has been stable for eighteen months"},
        {"n": 3, "text": "Refactoring keeps the migration reversible at every step"},
    ],
    "assumptions": ["The schema stays stable through the next two quarters"],
    "would_change_my_mind": ["Evidence that the schema changes more than monthly"],
    "confidence": 0.7,
}

ARGUMENT = (
    "This claim rests on an eighteen-month stability record that predates the new "
    "ingest sources, so the base rate it cites no longer describes the system."
)


def sheet(**overrides):
    data = json.loads(json.dumps(VALID))
    data.update(overrides)
    return data


class ExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_wrapped_json(self):
        text = 'Here is my answer sheet:\n{"a": 1}\nHope that helps!'
        self.assertEqual(extract_json(text), {"a": 1})

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        text = 'preamble {"a": "a } brace", "b": 2} trailing'
        self.assertEqual(extract_json(text), {"a": "a } brace", "b": 2})

    def test_no_json_at_all(self):
        with self.assertRaises(SheetParseError):
            extract_json("I would rather answer in prose.")

    def test_empty_response(self):
        with self.assertRaises(SheetParseError):
            extract_json("   ")


class SheetParsingTests(unittest.TestCase):
    def test_valid_sheet(self):
        parsed = parse_sheet(VALID)
        self.assertEqual(len(parsed.claims), 3)
        self.assertEqual(parsed.claim_numbers, (1, 2, 3))
        self.assertAlmostEqual(parsed.confidence, 0.7)
        self.assertEqual(parsed.compliance_warnings, ())

    def test_claims_as_bare_strings_are_numbered_in_order(self):
        parsed = parse_sheet(sheet(claims=["first claim", "second claim"]))
        self.assertEqual(parsed.claim_numbers, (1, 2))
        self.assertEqual(parsed.claims[1].text, "second claim")

    def test_numeric_prefix_in_claim_text_is_honoured_and_stripped(self):
        parsed = parse_sheet(sheet(claims=["1. first claim", "2. second claim"]))
        self.assertEqual(parsed.claims[0].text, "first claim")
        self.assertEqual(parsed.claim_numbers, (1, 2))

    def test_confidence_as_string(self):
        self.assertAlmostEqual(parse_sheet(sheet(confidence="0.42")).confidence, 0.42)

    def test_confidence_as_percentage(self):
        self.assertAlmostEqual(parse_sheet(sheet(confidence="65%")).confidence, 0.65)

    def test_single_falsifier_as_bare_string(self):
        parsed = parse_sheet(sheet(would_change_my_mind="a single falsifier"))
        self.assertEqual(parsed.would_change_my_mind, ("a single falsifier",))

    def test_nuance_is_kept(self):
        parsed = parse_sheet(sheet(nuance="the real answer turns on procurement"))
        self.assertIn("procurement", parsed.nuance)

    # -- strictness --------------------------------------------------------

    def test_too_many_claims_is_rejected(self):
        claims = [f"claim number {i}" for i in range(MAX_CLAIMS + 1)]
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(claims=claims))

    def test_zero_claims_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(claims=[]))

    def test_non_contiguous_numbering_is_rejected(self):
        claims = [
            {"n": 1, "text": "first"},
            {"n": 2, "text": "second"},
            {"n": 4, "text": "fourth"},
        ]
        with self.assertRaises(SheetSchemaError) as ctx:
            parse_sheet(sheet(claims=claims))
        self.assertIn("no gaps", str(ctx.exception))

    def test_duplicate_numbering_is_rejected(self):
        claims = [{"n": 1, "text": "first"}, {"n": 1, "text": "also first"}]
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(claims=claims))

    def test_confidence_above_one_is_rejected_not_clamped(self):
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(confidence=1.4))

    def test_confidence_below_zero_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(confidence=-0.1))

    def test_boolean_confidence_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(confidence=True))

    def test_missing_falsifiers_is_rejected(self):
        data = sheet()
        del data["would_change_my_mind"]
        with self.assertRaises(SheetSchemaError):
            parse_sheet(data)

    def test_empty_position_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_sheet(sheet(position="   "))

    def test_unexpected_field_is_rejected(self):
        with self.assertRaises(SheetSchemaError) as ctx:
            parse_sheet(sheet(recommendation="just do it"))
        self.assertIn("recommendation", str(ctx.exception))

    def test_actor_is_attached_to_the_error(self):
        with self.assertRaises(SheetSchemaError) as ctx:
            parse_sheet(sheet(confidence=3), actor="student:2")
        self.assertEqual(ctx.exception.actor, "student:2")

    # -- compliance warnings are not errors --------------------------------

    def test_multi_sentence_claim_warns_but_parses(self):
        claims = ["The rebuild is cheap. The refactor is not.", "Second claim"]
        parsed = parse_sheet(sheet(claims=claims))
        self.assertEqual(len(parsed.claims), 2)
        self.assertTrue(
            any("claim 1 is more than one sentence" in w for w in parsed.compliance_warnings)
        )

    def test_multi_sentence_position_warns_but_parses(self):
        parsed = parse_sheet(sheet(position="Refactor it. Do not rebuild it."))
        self.assertTrue(
            any("position is more than one sentence" in w for w in parsed.compliance_warnings)
        )


class CritiqueParsingTests(unittest.TestCase):
    allowed = {"A": (1, 2, 3), "B": (1, 2)}

    def critique(self, objections):
        return json.dumps({"objections": objections})

    def test_valid_critique(self):
        raw = parse_critique(
            self.critique(
                [
                    {"sheet": "A", "claim_n": 2, "argument": ARGUMENT},
                    {"sheet": "B", "claim_n": 1, "argument": ARGUMENT},
                ]
            ),
            allowed=self.allowed,
        )
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0].sheet, "A")

    def test_sheet_label_prefix_is_normalised(self):
        raw = parse_critique(
            self.critique(
                [
                    {"sheet": "Sheet A", "claim_n": 1, "argument": ARGUMENT},
                    {"sheet": "b", "claim_n": 1, "argument": ARGUMENT},
                ]
            ),
            allowed=self.allowed,
        )
        self.assertEqual({o.sheet for o in raw}, {"A", "B"})

    def test_skipping_a_sheet_is_non_compliant(self):
        with self.assertRaises(NonCompliantCritique) as ctx:
            parse_critique(
                self.critique([{"sheet": "A", "claim_n": 1, "argument": ARGUMENT}]),
                allowed=self.allowed,
            )
        self.assertIn("B", str(ctx.exception))

    def test_vague_agreement_is_non_compliant(self):
        with self.assertRaises(NonCompliantCritique):
            parse_critique(
                self.critique(
                    [
                        {"sheet": "A", "claim_n": 1, "argument": "I agree."},
                        {"sheet": "B", "claim_n": 1, "argument": ARGUMENT},
                    ]
                ),
                allowed=self.allowed,
            )

    def test_too_short_an_argument_is_non_compliant(self):
        with self.assertRaises(NonCompliantCritique):
            parse_critique(
                self.critique(
                    [
                        {"sheet": "A", "claim_n": 1, "argument": "Weak reasoning here"},
                        {"sheet": "B", "claim_n": 1, "argument": ARGUMENT},
                    ]
                ),
                allowed=self.allowed,
            )

    def test_objection_without_a_claim_number_is_non_compliant(self):
        with self.assertRaises(NonCompliantCritique):
            parse_critique(
                self.critique([{"sheet": "A", "argument": ARGUMENT}]),
                allowed=self.allowed,
            )

    def test_objection_against_a_claim_that_does_not_exist(self):
        with self.assertRaises(SheetSchemaError) as ctx:
            parse_critique(
                self.critique([{"sheet": "B", "claim_n": 3, "argument": ARGUMENT}]),
                allowed=self.allowed,
            )
        self.assertIn("claim 3", str(ctx.exception))

    def test_objection_against_an_unseen_sheet(self):
        with self.assertRaises(SheetSchemaError):
            parse_critique(
                self.critique([{"sheet": "C", "claim_n": 1, "argument": ARGUMENT}]),
                allowed=self.allowed,
            )

    def test_empty_objection_list_is_non_compliant(self):
        with self.assertRaises(NonCompliantCritique):
            parse_critique(self.critique([]), allowed=self.allowed)


class RevisionParsingTests(unittest.TestCase):
    allowed = {"A": (1, 2), "B": (3,)}

    def test_valid_revision_with_citation(self):
        data = sheet(changed_position=True, because=[{"critic": "A", "claim_n": 2}])
        revision = parse_revision(json.dumps(data), allowed=self.allowed)
        self.assertTrue(revision.changed_position)
        self.assertEqual(revision.because[0].critic, "A")

    def test_changed_position_as_string(self):
        data = sheet(changed_position="true", because=[])
        self.assertTrue(parse_revision(json.dumps(data), allowed=self.allowed).changed_position)

    def test_citing_an_objection_nobody_raised_is_rejected(self):
        data = sheet(changed_position=True, because=[{"critic": "B", "claim_n": 1}])
        with self.assertRaises(SheetSchemaError):
            parse_revision(json.dumps(data), allowed=self.allowed)

    def test_no_citation_is_fine(self):
        revision = parse_revision(json.dumps(sheet()), allowed=self.allowed)
        self.assertFalse(revision.changed_position)
        self.assertEqual(revision.because, ())


class DiffTests(unittest.TestCase):
    def test_identical_sheets_do_not_diff(self):
        before = parse_sheet(VALID)
        diff = diff_sheets(before, parse_sheet(VALID))
        self.assertFalse(diff.changed)
        self.assertFalse(diff.position_changed)

    def test_rewritten_position_is_detected(self):
        before = parse_sheet(VALID)
        after = parse_sheet(sheet(position="We should rebuild after all"))
        diff = diff_sheets(before, after)
        self.assertTrue(diff.position_changed)

    def test_dropped_claim_survives_renumbering(self):
        before = parse_sheet(VALID)
        after = parse_sheet(
            sheet(
                claims=[
                    {"n": 1, "text": "The current schema has been stable for eighteen months"},
                    {"n": 2, "text": "Refactoring keeps the migration reversible at every step"},
                ]
            )
        )
        diff = diff_sheets(before, after)
        self.assertEqual(len(diff.claims_dropped), 1)
        self.assertIn("two quarters", diff.claims_dropped[0].text)
        self.assertEqual(diff.claims_added, ())

    def test_reworded_claim_is_an_edit_not_a_drop_and_add(self):
        before = parse_sheet(VALID)
        claims = json.loads(json.dumps(VALID["claims"]))
        claims[1]["text"] = "The current schema has been stable for eighteen months so far"
        diff = diff_sheets(before, parse_sheet(sheet(claims=claims)))
        self.assertEqual(len(diff.claims_edited), 1)
        self.assertEqual(diff.claims_dropped, ())
        self.assertEqual(diff.claims_added, ())

    def test_confidence_delta(self):
        before = parse_sheet(VALID)
        diff = diff_sheets(before, parse_sheet(sheet(confidence=0.5)))
        self.assertAlmostEqual(diff.confidence_delta, -0.2)

    def test_declared_change_without_a_real_one_is_flagged(self):
        before = parse_sheet(VALID)
        diff = diff_sheets(before, parse_sheet(VALID), declared_change=True)
        self.assertFalse(diff.declaration_matches_diff)
        self.assertFalse(diff.position_changed)

    def test_real_change_without_declaring_it_is_flagged(self):
        before = parse_sheet(VALID)
        after = parse_sheet(sheet(position="We should rebuild after all"))
        diff = diff_sheets(before, after, declared_change=False)
        self.assertFalse(diff.declaration_matches_diff)


class VerdictParsingTests(unittest.TestCase):
    sources = ("Student 1", "Student 2")

    def verdict(self, **overrides):
        data = {
            "final_answer": "Refactor in place, staged over two releases.",
            "confidence_note": "Contested; the disagreement is about volume forecasts.",
            "minority_report": [],
        }
        data.update(overrides)
        return json.dumps(data)

    def test_valid_verdict(self):
        parsed = parse_verdict(self.verdict(), allowed_sources=self.sources)
        self.assertEqual(parsed.minority_report, ())

    def test_minority_report_is_attributed(self):
        parsed = parse_verdict(
            self.verdict(
                minority_report=[
                    {"source": "Student 2", "kind": "objection", "substance": "cost curve"}
                ]
            ),
            allowed_sources=self.sources,
        )
        self.assertEqual(parsed.minority_report[0].source, "Student 2")

    def test_unattributable_dissent_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_verdict(
                self.verdict(
                    minority_report=[
                        {"source": "Student 9", "kind": "claim", "substance": "x"}
                    ]
                ),
                allowed_sources=self.sources,
            )

    def test_missing_confidence_note_is_rejected(self):
        with self.assertRaises(SheetSchemaError):
            parse_verdict(
                json.dumps({"final_answer": "yes", "minority_report": []}),
                allowed_sources=self.sources,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
