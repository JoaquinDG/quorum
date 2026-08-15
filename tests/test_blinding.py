"""Blinding tests.

The interesting property is not "labels exist" — it is that the labels do not
encode seat order. Naive blinding ("the other sheets, in seat order, A then
B") is reproducible, looks anonymous, and leaks the whole seating chart after
one session: a critic that recognises one sheet knows the other by
elimination, and anyone comparing two sessions can align every label. These
tests pin the permutation as deterministic *and* non-positional, which are the
two things that have to hold at once.
"""

import unittest

from quorum import BlindingRound, blind_map, build_blinding, invert


class BlindMapTests(unittest.TestCase):
    seats = (1, 2, 3)

    def test_recipient_never_sees_its_own_sheet(self):
        for recipient in self.seats:
            mapping = blind_map("s1", recipient, self.seats, salt="r2")
            self.assertNotIn(recipient, mapping.values())
            self.assertEqual(len(mapping), len(self.seats) - 1)

    def test_labels_are_contiguous_from_a(self):
        mapping = blind_map("s1", 1, self.seats, salt="r2")
        self.assertEqual(sorted(mapping), ["A", "B"])

    def test_mapping_is_deterministic(self):
        first = blind_map("s1", 2, self.seats, salt="r2")
        second = blind_map("s1", 2, self.seats, salt="r2")
        self.assertEqual(first, second)

    def test_a_different_session_can_permute_differently(self):
        # Not every session id flips the order — with two others there is a
        # 50% chance either way — so this asserts the population, not a pair.
        orders = {
            tuple(blind_map(f"session-{i}", 1, self.seats, salt="r2").values())
            for i in range(40)
        }
        self.assertEqual(len(orders), 2, "labels never permute across sessions")

    def test_labels_are_not_seat_order(self):
        """The failure this guards against: 'A is always the lower seat'."""
        positional = 0
        for i in range(40):
            mapping = blind_map(f"session-{i}", 3, self.seats, salt="r2")
            if mapping["A"] < mapping["B"]:
                positional += 1
        self.assertNotIn(positional, (0, 40), "label order tracks seat order")

    def test_salt_changes_the_permutation(self):
        # Round 3 relabels critics so a student cannot align "the sheet I
        # critiqued" with "the critic who attacked me". Same reasoning as
        # above: assert over the population.
        differing = sum(
            blind_map(f"s{i}", 1, self.seats, salt="r2")
            != blind_map(f"s{i}", 1, self.seats, salt="r3")
            for i in range(40)
        )
        self.assertGreater(differing, 0)

    def test_two_seat_council_still_blinds(self):
        mapping = blind_map("s1", 1, (1, 2), salt="r2")
        self.assertEqual(mapping, {"A": 2})


class BlindingRoundTests(unittest.TestCase):
    def setUp(self):
        self.blinding = build_blinding("s1", (1, 2, 3), salt="r2")

    def test_label_and_seat_lookups_are_inverses(self):
        for recipient in (1, 2, 3):
            for label, seat in self.blinding.by_recipient[recipient].items():
                self.assertEqual(self.blinding.label_for(recipient, seat), label)
                self.assertEqual(self.blinding.seat_for(recipient, label), seat)

    def test_round_trips_through_a_dict(self):
        restored = BlindingRound.from_dict(self.blinding.to_dict())
        self.assertEqual(restored.salt, self.blinding.salt)
        self.assertEqual(restored.by_recipient, self.blinding.by_recipient)

    def test_json_keys_survive_stringification(self):
        # The trace stores recipients as JSON object keys, which are strings.
        as_dict = self.blinding.to_dict()
        self.assertEqual(sorted(as_dict["by_recipient"]), ["1", "2", "3"])
        restored = BlindingRound.from_dict(as_dict)
        self.assertEqual(restored.by_recipient[1], self.blinding.by_recipient[1])

    def test_invert(self):
        self.assertEqual(
            invert({"A": 3, "B": 2}),
            {3: "A", 2: "B"},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
