"""Syntactic repair of malformed model JSON.

The tests are split by intent. The first group is recovery: responses that
used to cost a seat its round now parse. The second is the one that matters
more — the guarantees about what repair will *not* do. A repair module that
quietly completes a half-written argument would manufacture a position and
attribute it to a model that never took it, and no amount of recovery rate
would be worth that.
"""

import json
import unittest

from quorum.repair import RepairReport, recover
from quorum.sheets import SheetParseError


def strings_in(value):
    """Every string that appears anywhere in a decoded JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings_in(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings_in(v)]
    return []


class CleanResponsesAreUntouched(unittest.TestCase):
    def test_valid_json_reports_no_repair(self):
        value, report = recover('{"position": "rebuild"}')
        self.assertEqual(value, {"position": "rebuild"})
        self.assertFalse(report.repaired)
        self.assertEqual(report.steps, ())

    def test_fenced_json_was_already_handled_and_is_not_a_repair(self):
        """`extract_json` stripped fences before this module existed."""
        _, report = recover('```json\n{"a": 1}\n```')
        self.assertFalse(report.repaired)

    def test_preamble_was_already_handled(self):
        _, report = recover('Here is my answer sheet:\n{"a": 1}')
        self.assertFalse(report.repaired)

    def test_a_dict_passes_straight_through(self):
        value, report = recover({"a": 1})
        self.assertEqual(value, {"a": 1})
        self.assertFalse(report.repaired)


class TruncationIsRecovered(unittest.TestCase):
    """The incident class: the model did the work, the closing brace is missing."""

    def test_unterminated_string_is_closed_and_kept(self):
        value, report = recover('{"position": "we should run as-i')
        self.assertEqual(value["position"], "we should run as-i")
        self.assertTrue(report.truncated)

    def test_nested_containers_are_closed_in_order(self):
        value, _ = recover('{"claims": [{"n": 1, "text": "partial')
        self.assertEqual(value, {"claims": [{"n": 1, "text": "partial"}]})

    def test_dangling_separator_is_dropped_not_completed(self):
        value, report = recover('{"a": 1, "b": ')
        self.assertEqual(value, {"a": 1})
        self.assertTrue(report.dropped_trailing)
        self.assertIn("dropped_incomplete_trailing_value", report.steps)

    def test_the_incident_shape(self):
        """A round-2 critique cut mid-argument, as on 2026-08-19."""
        full = json.dumps({"objections": [
            {"sheet": "A", "claim_n": 1, "argument": "The base rate cited no longer holds."},
            {"sheet": "B", "claim_n": 2, "argument": "This assumes traffic stays flat."},
        ]})
        value, report = recover(full[: int(len(full) * 0.8)])
        self.assertTrue(report.repaired)
        self.assertTrue(report.truncated)
        self.assertGreaterEqual(len(value["objections"]), 1)
        self.assertEqual(value["objections"][0]["argument"],
                         "The base rate cited no longer holds.")


class StructuralSlipsAreRepaired(unittest.TestCase):
    def test_trailing_comma_in_object(self):
        value, report = recover('{"a": 1,}')
        self.assertEqual(value, {"a": 1})
        self.assertIn("dropped_trailing_commas", report.steps)

    def test_trailing_comma_in_array(self):
        value, _ = recover('{"xs": [1, 2, ]}')
        self.assertEqual(value, {"xs": [1, 2]})

    def test_unescaped_inner_quotes(self):
        value, report = recover('{"argument": "the model said "run as-is" out loud"}')
        self.assertEqual(value["argument"], 'the model said "run as-is" out loud')
        self.assertIn("escaped_inner_quotes", report.steps)

    def test_braces_inside_prose_are_not_structure(self):
        """Claim text routinely contains braces; repair must not read them."""
        value, _ = recover('{"argument": "the {config} block is wrong')
        self.assertEqual(value["argument"], "the {config} block is wrong")


class RepairNeverInventsContent(unittest.TestCase):
    """The guarantees. These are the reason the module is allowed to exist."""

    def test_every_recovered_string_came_from_the_response(self):
        """Nothing in the output that was not in the input.

        The honesty invariant, checked structurally rather than by inspection:
        if repair ever completed a sentence, the completed text would not be
        found in the original bytes.
        """
        cases = [
            '{"position": "we should run as-i',
            '{"claims": [{"n": 1, "text": "three years of edge cases liv',
            '{"objections": [{"sheet": "A", "claim_n": 1, "argument": "the base rate',
            '{"a": 1, "b": ',
            '{"xs": [1, 2, ]}',
        ]
        for raw in cases:
            with self.subTest(raw=raw[:40]):
                value, _ = recover(raw)
                for text in strings_in(value):
                    self.assertIn(text, raw,
                                  f"repair produced text absent from the response: {text!r}")

    def test_a_truncated_field_stays_truncated(self):
        """No completion, no paraphrase — the fragment is left as a fragment."""
        value, report = recover('{"position": "we should refac')
        self.assertEqual(value["position"], "we should refac")
        self.assertTrue(report.truncated)

    def test_missing_fields_are_not_filled_in(self):
        value, _ = recover('{"position": "x"')
        self.assertEqual(set(value), {"position"})

    def test_unrecoverable_input_raises_rather_than_guessing(self):
        for raw in ("", "   ", "I cannot help with that request."):
            with self.subTest(raw=raw):
                with self.assertRaises(SheetParseError):
                    recover(raw)

    def test_repair_is_deterministic(self):
        raw = '{"claims": [{"n": 1, "text": "partial'
        first, r1 = recover(raw)
        second, r2 = recover(raw)
        self.assertEqual(first, second)
        self.assertEqual(r1, r2)


class ReportIsDisclosable(unittest.TestCase):
    def test_clean_report_adds_nothing_to_a_payload(self):
        self.assertEqual(RepairReport().as_payload(), {})

    def test_repaired_report_is_visible_in_a_payload(self):
        payload = RepairReport(repaired=True, truncated=True,
                               steps=("closed_truncated_json",)).as_payload()
        self.assertTrue(payload["repair"]["repaired"])
        self.assertTrue(payload["repair"]["truncated"])

    def test_report_round_trips(self):
        report = RepairReport(repaired=True, truncated=True, dropped_trailing=True,
                              steps=("a", "b"))
        self.assertEqual(RepairReport.from_dict(report.to_dict()), report)


if __name__ == "__main__":
    unittest.main()
