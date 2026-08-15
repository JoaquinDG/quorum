"""What the council cost, against what one model would have cost.

A 5–10x multiplier invites exactly one question — "why not just ask the best
model twice?" — and the only honest answer is a number. So every session
computes its own baseline and carries it in the trace, which means the report
can print it without re-deriving prices the record does not contain.

The baseline is deliberately conservative: *the same question, answered once,
by the priciest seat in the room*. It is not "the cheapest thing that would
have worked" (unknowable) and not "three sheets from one model" (a different
protocol). Picking the priciest seat uses price as a proxy for capability,
which is rough — a catalog can price a weak model highly — so
`SessionConfig.baseline_model` overrides it, and the chosen model is recorded
alongside the number so a reader can disagree with the choice rather than
having to guess it.

The multiple is reported even when it is unflattering. A session that came in
at 11x should say 11x; the spec's ≤8x guardrail is a target to be measured
against, not a claim to be protected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .council import Council
from .trace import SHEET_SUBMITTED, STUDENT_ABSENT, TraceEvent


@dataclass(frozen=True)
class Baseline:
    """What one model, answering once, would have cost."""

    model_id: str
    cost_est: float
    tokens_in: int
    tokens_out: int
    priced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_model": self.model_id,
            "baseline_cost_est": self.cost_est,
            "baseline_tokens_in": self.tokens_in,
            "baseline_tokens_out": self.tokens_out,
            "baseline_priced": self.priced,
        }


def pick_baseline_seat(council: Council, override: str | None = None, seat=None):
    """The seat the baseline is priced against.

    Defaults to the most expensive seat, on the reasoning that if you were
    going to ask one model a hard question you would ask your best one, and
    that comparing a council against a *cheap* single model would flatter the
    council by construction.

    `seat` supplies a yardstick that need not be *in* the council, which is
    required to compare lineups honestly. Naming a seated model by string
    cannot express "price every arm against one model" when the arms differ:
    the moment a lineup drops that model, the comparison either crashes or
    silently reverts to a different ruler. Pinning a `Seat` decouples "what
    one model would have cost" from "who happened to be in this room".
    """
    if seat is not None:
        return seat
    if override:
        for seat in council.seats():
            if seat.model_id == override:
                return seat
        raise KeyError(
            f"baseline_model {override!r} holds no seat; seated: "
            f"{[s.model_id for s in council.seats()]}"
        )
    return max(
        council.seats(),
        key=lambda s: (s.cost.input_per_mtok + s.cost.output_per_mtok, s.model_id),
    )


def single_model_baseline(
    council: Council, events: list[TraceEvent] | tuple[TraceEvent, ...], *,
    override: str | None = None, seat=None,
) -> Baseline:
    """Cost of answering the question once, from the session's own round 1.

    Round 1 is the right sample: every student saw an identical prompt and
    produced one answer, so it *is* a single-model call, measured rather than
    guessed. Output tokens are averaged across the students who answered —
    using the largest would price the baseline against the most verbose
    participant and quietly shrink the multiple.
    """
    seat = pick_baseline_seat(council, override, seat)
    round_one = [
        e
        for e in events
        if e.round == 1 and e.event_type in (SHEET_SUBMITTED, STUDENT_ABSENT)
    ]
    answered = [e for e in round_one if e.tokens_out > 0]
    tokens_in = max((e.tokens_in for e in round_one), default=0)
    tokens_out = (
        round(sum(e.tokens_out for e in answered) / len(answered)) if answered else 0
    )
    return Baseline(
        model_id=seat.model_id,
        cost_est=seat.cost.estimate(tokens_in, tokens_out),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        priced=seat.cost.priced,
    )


def cost_multiple(session_cost: float, baseline: Baseline) -> float | None:
    """How many single-model answers this session cost. `None` if unpriceable.

    **Within-session only. Do not compare this number across councils.**

    The baseline defaults to the priciest seat *in that council*, so changing
    the lineup moves the yardstick as well as the measurement. A lineup
    experiment made the trap concrete: swapping the arbiter from Opus to
    Sonnet reported 13.3x -> 27.4x, reading as a doubling in cost, while the
    session actually got *cheaper* in absolute terms. The multiple only moved
    because the most expensive seat — and therefore the denominator — had.

    Pinning `SessionConfig.baseline_seat` fixes the *price* half of the
    problem but not the *token* half: the baseline's token counts are measured
    from each session's own round 1, so a council with a verbose student
    inflates its own yardstick and reports a flatteringly small multiple. Two
    lineups measured this directly — one reported 6.0x and another 27.4x while
    their absolute costs differed by only 48%.

    **Compare `cost_est`. The multiple answers "was this session worth it?",
    not "which lineup is cheaper?"**
    """
    if not baseline.priced or baseline.cost_est <= 0:
        return None
    return session_cost / baseline.cost_est
