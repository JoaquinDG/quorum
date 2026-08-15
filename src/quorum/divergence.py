"""How much the opening sheets differ in *wording*.

Computed on the round-1 sheets alone — before critique, before revision —
because that is the only moment the positions are genuinely independent. After
round 2 the sheets have influenced each other and convergence stops being
evidence of anything.

**DEMOTED: this does not measure disagreement.** It was shipped as a
disagreement score with its lexical blind spot documented. Then it met real
model output and failed in the common case, not the exotic one.

Three Claude tiers answered one question. All three said *refactor in place*
— the same conclusion, reached by similar reasoning. This metric scored the
question **0.68, "sharply contested"**, because the three models used
different vocabulary to say the same thing. Claim divergence was 0.79 between
sheets that agreed.

That is not a blind spot at the edge of the method. It is the method: word
overlap tracks vocabulary, and vocabulary varies most exactly where models are
articulate. The failure runs both ways —

- same conclusion, different words → scored as sharply contested (observed, in
  `tests/test_real_output.py`)
- opposite conclusions, near-identical words ("we should rebuild" / "we should
  **not** rebuild") → scored as agreeing (constructed, in
  `tests/test_divergence.py`)

— and a number that can say "contested" about unanimity is worse than no
number, because a reader who trusts it is misled in the direction the whole
project exists to prevent.

**REMOVED FROM THE REPORT** by the council's own decision. Quorum was run on
this question (`evals/DOGFOOD.md`, `wording_metric`): all three students opened
arguing to keep it as a relabelled heuristic, and all three reversed. Their
finding was that renaming does not help, because a precise-looking number gets
read as substantive however it is captioned — a unanimous council using varied
phrasing still *looks* contested. Keeping an inert metric on the page is not
free: it costs clutter, cognitive overhead, and standing risk of downstream
misuse.

The computation stays here, and the trace still records it, on the strength of
the dissent the arbiter refused to drop: deleting the *display* sacrifices no
optionality while the raw data survives, so any real use case can recompute it
later. The council also rejected paying for a semantic replacement before a
concrete consumer exists.

Nothing in the protocol reads it, and no reader-facing surface shows it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from .sheets import AnswerSheet

W_POSITION = 0.50
W_CLAIMS = 0.35
W_CONFIDENCE = 0.15

_MAX_CONFIDENCE_SPREAD = 0.5
"""The largest standard deviation two values in [0, 1] can have, used to
normalise the spread onto the same 0-1 scale as the other components."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _claim_overlap(a: AnswerSheet, b: AnswerSheet) -> float:
    """Best-match mean similarity between two claim sets.

    Symmetric by construction: matching a's claims against b's and b's against
    a's and averaging, because a sheet with two claims compared against one
    with five would otherwise score differently depending on which came first.
    """
    if not a.claims or not b.claims:
        return 0.0

    def directed(left: AnswerSheet, right: AnswerSheet) -> float:
        return sum(
            max(_similarity(c.text, other.text) for other in right.claims)
            for c in left.claims
        ) / len(left.claims)

    return (directed(a, b) + directed(b, a)) / 2


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


@dataclass(frozen=True)
class Disagreement:
    score: float
    position_divergence: float
    claim_divergence: float
    confidence_spread: float
    pairs: int

    @property
    def label(self) -> str:
        """Describes *wording*, never agreement.

        The old labels ("sharply contested", "near-unanimous") asserted
        something about the positions that the method cannot see, and were
        observed calling a unanimous council sharply contested.
        """
        if self.score >= 0.66:
            return "high lexical variety"
        if self.score >= 0.4:
            return "moderate lexical variety"
        if self.score >= 0.2:
            return "low lexical variety"
        return "near-identical wording"

    @property
    def measures_agreement(self) -> bool:
        """Always False. Present so any surface that shows this number has to
        acknowledge what it is, rather than quietly implying otherwise."""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "position_divergence": round(self.position_divergence, 4),
            "claim_divergence": round(self.claim_divergence, 4),
            "confidence_spread": round(self.confidence_spread, 4),
            "pairs": self.pairs,
            "method": "lexical",
            "measures_agreement": False,
            "caveat": "word overlap only; observed calling a unanimous council contested",
        }


def disagreement(sheets: list[AnswerSheet] | tuple[AnswerSheet, ...]) -> Disagreement:
    """Score how far apart the opening sheets are, pairwise.

    A council of one has nothing to disagree with and scores 0 — not because
    the question was uncontested, but because the session cannot say. The
    `pairs` count is carried so a reader can tell those two cases apart.
    """
    sheets = [s for s in sheets if s is not None]
    pairs = list(combinations(sheets, 2))
    if not pairs:
        return Disagreement(0.0, 0.0, 0.0, 0.0, 0)

    position = sum(1 - _similarity(a.position, b.position) for a, b in pairs) / len(pairs)
    claims = sum(1 - _claim_overlap(a, b) for a, b in pairs) / len(pairs)
    spread = min(
        1.0, _stdev([s.confidence for s in sheets]) / _MAX_CONFIDENCE_SPREAD
    )

    score = W_POSITION * position + W_CLAIMS * claims + W_CONFIDENCE * spread
    return Disagreement(
        score=score,
        position_divergence=position,
        claim_divergence=claims,
        confidence_spread=spread,
        pairs=len(pairs),
    )
