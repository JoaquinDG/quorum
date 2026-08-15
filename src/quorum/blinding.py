"""Who sees what, under which name.

Blinding in Quorum is two mechanisms stacked. The schema does the heavy
lifting — an answer sheet has almost no stylistic surface to fingerprint (see
`sheets`). This module does the rest: it decides which anonymous label each
participant sees each other participant under, and it makes that decision
*unlearnable from position*.

The naive implementation — "the other sheets, in seat order, labelled A then
B" — leaks the entire seating chart after one session. Student 3 always sees
seat 1 as A, so a critic that recognises one sheet has deduced the other by
elimination, and a reader comparing two sessions can align every label. Labels
are therefore permuted per (session, recipient, round) from a hash, which is
deterministic (tests and replay reproduce it exactly) without being
positional.

Blinding runs in round 3 as well, under a fresh mapping. A student revising
its sheet is told what was argued against claim 2, not who argued it, and a
round-2 label reused in round 3 would let a student align "the sheet I
critiqued" with "the critic who attacked me". Different salt, different
permutation.

What is *not* blinded: the trace. It records seats, models, and every mapping,
because the trace is the auditor's view rather than a participant's, and a
report that cannot say who changed their mind has no product in it. No
participant is ever shown the trace.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass

LABELS = string.ascii_uppercase


def _rank_key(session_id: str, salt: str, recipient: int, seat: int) -> bytes:
    """A stable pseudo-random sort key for one (viewer, viewed) pair."""
    material = f"{session_id}|{salt}|{recipient}|{seat}".encode("utf-8")
    return hashlib.blake2b(material, digest_size=16).digest()


def blind_map(
    session_id: str, recipient: int, seats: list[int] | tuple[int, ...], *, salt: str
) -> dict[str, int]:
    """Map anonymous labels to seats, from the point of view of `recipient`.

    The recipient's own seat is excluded — a student never critiques its own
    sheet, and offering it the chance is an invitation to self-endorse.
    """
    others = sorted(s for s in seats if s != recipient)
    if len(others) > len(LABELS):  # pragma: no cover - councils are capped at 3
        raise ValueError(f"cannot blind {len(others)} participants with {len(LABELS)} labels")
    ordered = sorted(others, key=lambda s: _rank_key(session_id, salt, recipient, s))
    return {LABELS[i]: seat for i, seat in enumerate(ordered)}


def invert(mapping: dict[str, int]) -> dict[int, str]:
    return {seat: label for label, seat in mapping.items()}


@dataclass(frozen=True)
class BlindingRound:
    """One round's complete label assignment, as recorded in the trace.

    Kept as a first-class object because replay needs it: an objection is
    stored canonically against a seat, but the report wants to show the label
    the critic actually wrote, and reconstructing that from scratch would mean
    re-deriving hashes in the renderer. The chess-PGN rule applies — if the
    renderer needs it, the trace carries it.
    """

    salt: str
    by_recipient: dict[int, dict[str, int]]

    def label_for(self, recipient: int, seat: int) -> str:
        return invert(self.by_recipient[recipient])[seat]

    def seat_for(self, recipient: int, label: str) -> int:
        return self.by_recipient[recipient][label]

    def to_dict(self) -> dict[str, object]:
        return {
            "salt": self.salt,
            "by_recipient": {
                str(recipient): dict(mapping)
                for recipient, mapping in sorted(self.by_recipient.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> BlindingRound:
        return cls(
            salt=data["salt"],
            by_recipient={
                int(recipient): {label: int(seat) for label, seat in mapping.items()}
                for recipient, mapping in data["by_recipient"].items()
            },
        )


def build_blinding(
    session_id: str, seats: list[int] | tuple[int, ...], *, salt: str
) -> BlindingRound:
    return BlindingRound(
        salt=salt,
        by_recipient={r: blind_map(session_id, r, seats, salt=salt) for r in seats},
    )
