"""The deanonymization probe: measuring the blinding instead of assuming it.

Every anonymization scheme claims to work. This one publishes a number.

A critic model is shown the round-1 sheets under their blinded labels, given
the roster of model families that took part, and asked to say which wrote
which. Guess accuracy is tracked against chance. Near chance means the schema
did its job. Above chance is a known, quantified limitation — stated in the
README with the number attached, not argued away.

Three design commitments make the number worth reading:

**The probe never runs on a participant.** A student would recognise its own
sheet instantly and score a free hit, which would inflate the leak without
measuring one. `probe_session` refuses a prober that holds a seat, in the same
way `Council` refuses an arbiter that debated.

**The prober is given every advantage.** It is told the roster, told that each
model wrote exactly one sheet, and free to reason about argument style,
hedging, formatting and vocabulary. A probe that has to *also* guess the
candidate list would understate the leak, and understating the leak is the one
direction this measurement must not err in.

**A refusal is not a pass.** A prober that declines to guess, or returns
something unparseable, is recorded as `abstained` and excluded from the
accuracy denominator rather than counted as a miss. Counting abstentions as
wrong answers would let a cautious prober manufacture a reassuring number.

What the probe cannot tell you: it measures whether *this* prober, on *these*
sheets, can identify authorship. A stronger prober may do better, and one
run's accuracy on a handful of sessions carries wide error bars — which is why
`ProbeReport` reports `n` beside the rate and the README is expected to quote
both.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from .council import Council, Seat
from .prompts import PROBE_PROMPT_HEADER, build_probe_prompt, render_sheet
from .providers.base import ProviderError, ProviderPool
from .sheets import SheetError, extract_json
from . import trace as tr

PROBER = "prober"


class ProbeError(ValueError):
    """The probe cannot be run as configured."""


@dataclass(frozen=True)
class Guess:
    label: str
    guessed_model: str
    actual_model: str
    seat: int

    @property
    def correct(self) -> bool:
        return self.guessed_model == self.actual_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "guessed_model": self.guessed_model,
            "actual_model": self.actual_model,
            "seat": self.seat,
            "correct": self.correct,
        }


@dataclass(frozen=True)
class ProbeResult:
    session_id: str
    prober_model: str
    guesses: tuple[Guess, ...]
    roster: tuple[str, ...]
    abstained: bool = False
    detail: str = ""

    @property
    def hits(self) -> int:
        return sum(1 for g in self.guesses if g.correct)

    @property
    def attempts(self) -> int:
        return len(self.guesses)

    @property
    def accuracy(self) -> float:
        return self.hits / self.attempts if self.attempts else 0.0

    @property
    def chance(self) -> float:
        """Probability of a correct guess at random.

        `1/len(roster)` per sheet: the prober is told each model wrote exactly
        one sheet, but is not required to produce a permutation, and scoring
        it as one would understate chance for a prober that repeats a guess.
        """
        return 1 / len(self.roster) if self.roster else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prober_model": self.prober_model,
            "roster": list(self.roster),
            "guesses": [g.to_dict() for g in self.guesses],
            "hits": self.hits,
            "attempts": self.attempts,
            "accuracy": round(self.accuracy, 4),
            "chance": round(self.chance, 4),
            "abstained": self.abstained,
            "detail": self.detail,
        }


@dataclass
class ProbeReport:
    """Aggregate across sessions. The published blinding metric."""

    results: list[ProbeResult] = field(default_factory=list)

    @property
    def sessions(self) -> int:
        return len(self.results)

    @property
    def scored_sessions(self) -> int:
        return sum(1 for r in self.results if not r.abstained and r.attempts)

    @property
    def attempts(self) -> int:
        return sum(r.attempts for r in self.results if not r.abstained)

    @property
    def hits(self) -> int:
        return sum(r.hits for r in self.results if not r.abstained)

    @property
    def accuracy(self) -> float:
        return self.hits / self.attempts if self.attempts else 0.0

    @property
    def chance(self) -> float:
        scored = [r for r in self.results if not r.abstained and r.attempts]
        if not scored:
            return 0.0
        return sum(r.chance * r.attempts for r in scored) / sum(r.attempts for r in scored)

    @property
    def excess_over_chance(self) -> float:
        """Percentage points above random. The number that actually matters.

        Zero or below means the schema blinding held for this prober. Positive
        is a leak, and its size is the finding.
        """
        return self.accuracy - self.chance

    @property
    def abstentions(self) -> int:
        return sum(1 for r in self.results if r.abstained)

    def summary(self) -> str:
        if not self.attempts:
            return f"no scored guesses across {self.sessions} session(s)"
        return (
            f"{self.accuracy:.1%} accuracy vs {self.chance:.1%} chance "
            f"({self.excess_over_chance:+.1%} points) over {self.attempts} guesses "
            f"in {self.scored_sessions} session(s)"
            + (f", {self.abstentions} abstained" if self.abstentions else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "scored_sessions": self.scored_sessions,
            "abstentions": self.abstentions,
            "attempts": self.attempts,
            "hits": self.hits,
            "accuracy": round(self.accuracy, 4),
            "chance": round(self.chance, 4),
            "excess_over_chance": round(self.excess_over_chance, 4),
            "summary": self.summary(),
        }


def _parse_guesses(text: str, roster: tuple[str, ...], labels: tuple[str, ...]) -> dict[str, str]:
    """Pull `{label: model}` out of the prober's reply.

    Strict about the roster: a guess naming a model that did not take part is
    dropped rather than counted as a miss, because it is a formatting failure
    rather than a failed identification, and folding the two together would
    make the leak look smaller than it is.
    """
    data = extract_json(text, actor=PROBER)
    if isinstance(data, dict):
        data = data.get("guesses", data)
    if isinstance(data, dict):
        pairs = data.items()
    elif isinstance(data, list):
        pairs = [
            (item.get("sheet", item.get("label")), item.get("model", item.get("guess")))
            for item in data
            if isinstance(item, dict)
        ]
    else:
        raise SheetError(f"unreadable probe reply: {text[:120]!r}", actor=PROBER)

    guesses: dict[str, str] = {}
    lookup = {m.lower(): m for m in roster}
    for label, model in pairs:
        if not isinstance(label, str) or not isinstance(model, str):
            continue
        label = label.strip().upper().removeprefix("SHEET").strip()
        if label not in labels:
            continue
        canonical = lookup.get(model.strip().lower())
        if canonical:
            guesses[label] = canonical
    return guesses


def probe_session(
    result: Any,
    prober: Seat,
    providers: ProviderPool,
    *,
    council: Council | None = None,
    writer: tr.TraceWriter | None = None,
    max_tokens: int = 1024,
    shuffle_seed: int | None = 0,
) -> ProbeResult:
    """Ask `prober` to identify the authors of a session's round-1 sheets.

    `result` is a live `SessionResult`. The probe reads the sheets and the true
    seat→model mapping from it, which is legitimate precisely because the probe
    is the auditor's instrument, not a participant: it is scoring the record,
    and only the anonymised half is ever put on the wire.
    """
    council = council or result.council
    if prober.model_id in [s.model_id for s in council.seats()]:
        raise ProbeError(
            f"prober {prober.model_id!r} took part in this session; a participant "
            "recognises its own sheet and scores a hit that measures nothing"
        )
    if not providers.has(prober.provider):
        raise KeyError(f"no provider registered for {prober.provider!r}")

    present = [s for s in result.students if s.present]
    if len(present) < 2:
        raise ProbeError("a probe needs at least two sheets to confuse")

    # Labels for the probe are its own, unrelated to any round's blinding —
    # reusing a round-2 mapping would leak that mapping into the audit record.
    order = list(range(len(present)))
    if shuffle_seed is not None:
        random.Random(f"{result.session_id}|probe|{shuffle_seed}").shuffle(order)
    labelled = [(chr(ord("A") + i), present[seat]) for i, seat in enumerate(order)]
    labels = tuple(label for label, _ in labelled)

    roster = tuple(sorted(s.model_id for s in present))
    rendered = "\n\n".join(
        render_sheet(student.initial, label) for label, student in labelled
    )
    prompt = build_probe_prompt(result.task, rendered, roster)

    try:
        completion = providers.get(prober.provider).complete(
            prober.model_id, prompt, max_tokens
        )
    except ProviderError as exc:
        probe = ProbeResult(
            result.session_id, prober.model_id, (), roster, abstained=True,
            detail=f"provider error: {exc}",
        )
        _emit(writer, result, prober, probe, completion=None)
        return probe

    try:
        guessed = _parse_guesses(completion.text, roster, labels)
    except (SheetError, ValueError) as exc:
        probe = ProbeResult(
            result.session_id, prober.model_id, (), roster, abstained=True,
            detail=f"unparseable reply: {exc}",
        )
        _emit(writer, result, prober, probe, completion)
        return probe

    guesses = tuple(
        Guess(
            label=label,
            guessed_model=guessed[label],
            actual_model=student.model_id,
            seat=student.seat,
        )
        for label, student in labelled
        if label in guessed
    )
    probe = ProbeResult(
        result.session_id,
        prober.model_id,
        guesses,
        roster,
        abstained=not guesses,
        detail="" if guesses else "no usable guesses in reply",
    )
    _emit(writer, result, prober, probe, completion)
    return probe


def _emit(writer, result, prober: Seat, probe: ProbeResult, completion) -> None:
    if writer is None:
        return
    tokens_in = completion.input_tokens if completion else 0
    tokens_out = completion.output_tokens if completion else 0
    writer.emit(
        session_id=result.session_id,
        round=0,
        actor=PROBER,
        event_type=tr.PROBE_RESULT,
        payload=probe.to_dict(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_est=prober.cost.estimate(tokens_in, tokens_out),
    )


def probe_all(
    results: Iterable[Any],
    prober: Seat,
    providers: ProviderPool,
    *,
    writer: tr.TraceWriter | None = None,
) -> ProbeReport:
    """Probe a batch of sessions and aggregate. This is the published metric."""
    report = ProbeReport()
    for result in results:
        report.results.append(
            probe_session(result, prober, providers, writer=writer)
        )
    return report


__all__ = [
    "Guess",
    "PROBER",
    "PROBE_PROMPT_HEADER",
    "ProbeError",
    "ProbeReport",
    "ProbeResult",
    "probe_all",
    "probe_session",
]
