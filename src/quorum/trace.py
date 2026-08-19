"""The trace: append-only JSONL, and the only artifact that matters.

Quorum's product is not the final answer, it is the argument that produced it.
That means the transcript has to survive the process that made it — outliving
the objects in memory, the report that renders it, and any future UI.

The design rule is the chess-PGN principle: **any renderer is a player for
this file.** A PGN does not ship with a board; it carries enough that any
board can be reconstructed. If a Session Report, a benchmark harness, or the
v2 replay world would need something the trace does not carry, the trace
format is wrong — not the renderer. `replay.py` exists to keep that honest: it
reconstructs a full session from the file alone, and the test suite compares
the result against the live session object field by field.

Two consequences fall out of that rule:

- Events carry *full* content, not references. A `sheet_submitted` event holds
  the whole sheet. A trace of pointers into a database is a log, not a record.
- The trace is not blinded. It records seats, model identities, and every
  label mapping, because the auditor's view and the participant's view are
  different views on purpose. No participant is ever shown the trace.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

TASK_POSED = "task_posed"
SHEET_SUBMITTED = "sheet_submitted"
SHEETS_BLINDED = "sheets_blinded"
OBJECTION_RAISED = "objection_raised"
SHEET_REVISED = "sheet_revised"
POSITION_CHANGED = "position_changed"
VERDICT_DELIVERED = "verdict_delivered"
MINORITY_RECORDED = "minority_recorded"
STUDENT_ABSENT = "student_absent"
PROBE_RESULT = "probe_result"
ARBITER_ABSENT = "arbiter_absent"
SESSION_CLOSED = "session_closed"
ATTEMPT_DISCARDED = "attempt_discarded"
INJECTION_FLAGGED = "injection_flagged"
ROUND_COMPLETED = "round_completed"
SESSION_RESUMED = "session_resumed"

EVENT_TYPES = frozenset(
    {
        TASK_POSED,
        SHEET_SUBMITTED,
        SHEETS_BLINDED,
        OBJECTION_RAISED,
        SHEET_REVISED,
        POSITION_CHANGED,
        VERDICT_DELIVERED,
        MINORITY_RECORDED,
        STUDENT_ABSENT,
        PROBE_RESULT,
        # Three additions to the vocabulary in the spec.
        #
        # ARBITER_ABSENT: fail-closed has to cover the grader too, and folding
        # it into STUDENT_ABSENT would make "how many students dropped out?" —
        # a published protocol-health number — quietly wrong.
        #
        # SESSION_CLOSED: replay needs a terminator to distinguish a session
        # that ended from a file that was truncated mid-write.
        #
        # ATTEMPT_DISCARDED: a re-prompted critique costs a real model call
        # whose output is thrown away. Without an event, that call is invisible
        # and the session reports a *lower* cost than a clean one — the exact
        # direction of error a cost guardrail must not have. It also answers
        # "what are repairs costing us?", which is worth knowing.
        ARBITER_ABSENT,
        SESSION_CLOSED,
        ATTEMPT_DISCARDED,
        # INJECTION_FLAGGED: the shield noticed something in text that one
        # participant wrote and another was about to read. It is an event
        # rather than a field on the sheet for the reason everything else here
        # is an event: the trace is the auditor's record, and "was this
        # session's verdict written after somebody tried to instruct the
        # arbiter?" is a question the file has to be able to answer on its
        # own. Flagging is not blocking, so a session can carry these events
        # and still be perfectly sound; what it cannot do is carry them
        # silently.
        INJECTION_FLAGGED,
        # ROUND_COMPLETED: the checkpoint. Everything needed to rebuild state
        # was already in the trace — replay proves that — but "where did this
        # session get to?" had to be inferred from which events happened to be
        # present, and inference is the wrong basis for deciding what to
        # re-bill. An explicit marker makes the resume point a fact in the
        # record rather than a deduction about it.
        #
        # SESSION_RESUMED: a session that was interrupted and continued is not
        # the same artifact as one that ran straight through, even when the
        # final trace is identical in schema. Anyone auditing a verdict is
        # entitled to know the debate stopped and started again, and where.
        ROUND_COMPLETED,
        SESSION_RESUMED,
    }
)

SYSTEM = "system"
ARBITER = "arbiter"


def student_actor(seat: int) -> str:
    return f"student:{seat}"


@dataclass(frozen=True)
class TraceEvent:
    ts: float
    session_id: str
    round: int
    actor: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_est: float = 0.0
    # Cache accounting. Added after the schema was in use, so every field here
    # is optional on read and defaults to zero: a trace written before these
    # existed still parses, and replays of it report no caching rather than
    # failing. `tokens_in` above is the uncached remainder, so the gross input
    # for an event is tokens_in + cache_read + cache_write.
    cache_read: int = 0
    cache_write: int = 0
    # What this call would have cost with no caching. Carried rather than
    # re-derived, for the same reason the baseline is: a reader comparing
    # effective against uncached cost should not have to hold prices the
    # record does not contain.
    uncached_cost_est: float = 0.0

    @property
    def gross_tokens_in(self) -> int:
        return self.tokens_in + self.cache_read + self.cache_write

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "session_id": self.session_id,
            "round": self.round,
            "actor": self.actor,
            "event_type": self.event_type,
            "payload": self.payload,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_est": self.cost_est,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "uncached_cost_est": self.uncached_cost_est,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        missing = {
            "ts",
            "session_id",
            "round",
            "actor",
            "event_type",
            "payload",
        } - set(data)
        if missing:
            raise ValueError(f"trace event is missing fields: {sorted(missing)}")
        if data["event_type"] not in EVENT_TYPES:
            raise ValueError(f"unknown event_type: {data['event_type']!r}")
        return cls(
            ts=float(data["ts"]),
            session_id=str(data["session_id"]),
            round=int(data["round"]),
            actor=str(data["actor"]),
            event_type=str(data["event_type"]),
            payload=dict(data["payload"]),
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            cost_est=float(data.get("cost_est", 0.0)),
            cache_read=int(data.get("cache_read", 0)),
            cache_write=int(data.get("cache_write", 0)),
            uncached_cost_est=float(
                # Falls back to the effective cost, so a pre-cache trace
                # reports "no saving" rather than a free session.
                data.get("uncached_cost_est", data.get("cost_est", 0.0))
            ),
        )


@dataclass(frozen=True)
class CacheSummary:
    """What a session read, what of it was cached, and what that was worth.

    The honest-invariant carrier for prompt caching: it reports the effective
    bill beside the bill the same session would have run up with no cache, so
    a saving has to be visible as the difference between two numbers rather
    than asserted. When nothing caches, the two are equal and `saved` is 0.0 —
    which is a finding, not a gap in the measurement.
    """

    tokens_in: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_est: float = 0.0
    uncached_cost_est: float = 0.0

    @property
    def gross_tokens_in(self) -> int:
        """Every input token the models read, cached or not."""
        return self.tokens_in + self.cache_read + self.cache_write

    @property
    def cache_hit_share(self) -> float:
        """Fraction of gross input served from cache, 0.0 when nothing ran.

        Counts reads only. A cache *write* is not a hit — it is the full-price
        token that paid to create the entry, and folding writes in here would
        let a session that only ever wrote report a high hit rate.
        """
        gross = self.gross_tokens_in
        return self.cache_read / gross if gross else 0.0

    @property
    def saved(self) -> float:
        """USD the cache actually saved. Negative when writes never paid off.

        Not clamped. A session that writes cache entries and never reads them
        pays the write premium for nothing, and that is exactly the case a
        reader needs to see rather than have rounded up to zero.
        """
        return self.uncached_cost_est - self.cost_est

    @classmethod
    def from_events(cls, events: Iterable[TraceEvent]) -> CacheSummary:
        events = list(events)
        return cls(
            tokens_in=sum(e.tokens_in for e in events),
            cache_read=sum(e.cache_read for e in events),
            cache_write=sum(e.cache_write for e in events),
            cost_est=sum(e.cost_est for e in events),
            uncached_cost_est=sum(e.uncached_cost_est for e in events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross_tokens_in": self.gross_tokens_in,
            "tokens_in_uncached": self.tokens_in,
            "cache_read_tokens": self.cache_read,
            "cache_write_tokens": self.cache_write,
            "cache_hit_share": round(self.cache_hit_share, 4),
            "cost_est": round(self.cost_est, 6),
            "uncached_cost_est": round(self.uncached_cost_est, 6),
            "saved_est": round(self.saved, 6),
        }


@dataclass(frozen=True)
class MalformationRate:
    """How often a seat's responses failed to parse, and how often that was
    recovered without paying for another generation.

    The number workstream 2 is measured against. Kept per model rather than
    per seat because the question it answers is about a vendor — "is the
    cheapest seat in the lineup also the one that cannot hold the format?" —
    and that has been true often enough to be worth watching rather than
    remembering.
    """

    attempts: int = 0
    malformed: int = 0
    repaired: int = 0
    discarded: int = 0

    @property
    def rate(self) -> float:
        return self.malformed / self.attempts if self.attempts else 0.0

    @property
    def recovered_share(self) -> float:
        """Of the responses that arrived malformed, how many were saved."""
        return self.repaired / self.malformed if self.malformed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "malformed": self.malformed,
            "repaired": self.repaired,
            "discarded": self.discarded,
            "rate": round(self.rate, 4),
            "recovered_share": round(self.recovered_share, 4),
        }


def malformation_by_model(events: Iterable[TraceEvent]) -> dict[str, MalformationRate]:
    """Per-model malformation, counted from the trace alone.

    Derived rather than tallied during the run, so it can be recomputed for
    any recorded session — including the ones captured before this existed,
    which is what makes a before/after comparison possible at all.
    """
    events = list(events)
    attempts: dict[str, int] = {}
    malformed: dict[str, int] = {}
    repaired: dict[str, int] = {}
    discarded: dict[str, int] = {}

    def model_of(event: TraceEvent) -> str:
        payload = event.payload
        return str(payload.get("model_id") or payload.get("critic_model") or "")

    for event in events:
        model = model_of(event)
        if not model:
            continue
        if event.tokens_in or event.tokens_out:
            attempts[model] = attempts.get(model, 0) + 1
        if event.event_type == ATTEMPT_DISCARDED:
            malformed[model] = malformed.get(model, 0) + 1
            discarded[model] = discarded.get(model, 0) + 1
        elif event.event_type == STUDENT_ABSENT and event.payload.get("raw"):
            malformed[model] = malformed.get(model, 0) + 1
            discarded[model] = discarded.get(model, 0) + 1
        elif event.payload.get("repair", {}).get("repaired"):
            # Repaired responses arrived malformed too — counting only the
            # discards would make repair look like the malformation going
            # away, when what it did was stop the malformation costing
            # anything. Those are different claims.
            malformed[model] = malformed.get(model, 0) + 1
            repaired[model] = repaired.get(model, 0) + 1

    models = set(attempts) | set(malformed)
    return {
        m: MalformationRate(
            attempts=attempts.get(m, 0),
            malformed=malformed.get(m, 0),
            repaired=repaired.get(m, 0),
            discarded=discarded.get(m, 0),
        )
        for m in sorted(models)
    }


class TraceWriter:
    """Records events in memory and, optionally, appends them to a JSONL file.

    In-memory is not a convenience for tests: the session needs its own events
    back to build a result, and reading them off disk to do that would make
    every session depend on a writable filesystem. The file is the durable
    copy of the same list, so a session with `path=None` is still fully
    replayable in-process.

    `clock` is injectable because a deterministic suite cannot assert on
    wall-clock timestamps, and a trace whose ordering depends on clock
    resolution is a trace that reorders itself on a fast machine.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self._clock = clock
        self._events: list[TraceEvent] = []
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def emit(
        self,
        *,
        session_id: str,
        round: int,
        actor: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_est: float = 0.0,
        cache_read: int = 0,
        cache_write: int = 0,
        uncached_cost_est: float | None = None,
    ) -> TraceEvent:
        if event_type not in EVENT_TYPES:
            # A typo'd event type is silently invisible to every consumer that
            # filters by name, which is the worst possible failure for an
            # audit record.
            raise ValueError(
                f"unknown event_type {event_type!r}; known types: {sorted(EVENT_TYPES)}"
            )
        event = TraceEvent(
            ts=self._clock(),
            session_id=session_id,
            round=round,
            actor=actor,
            event_type=event_type,
            payload=payload or {},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_est=cost_est,
            cache_read=cache_read,
            cache_write=cache_write,
            # Defaulting to the effective cost keeps the invariant that
            # uncached >= effective for callers that do not price the
            # counterfactual: they report no saving, never a negative one.
            uncached_cost_est=cost_est if uncached_cost_est is None else uncached_cost_est,
        )
        self._events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def adopt(self, events: Iterable[TraceEvent]) -> None:
        """Take prior events into memory without rewriting them to disk.

        What resume needs: the file already holds these lines, so re-emitting
        them would duplicate the record, but the in-memory list is what builds
        the final result, so leaving them out would produce a session that
        reports only the rounds it happened to run live.
        """
        self._events.extend(events)

    def of_type(self, event_type: str) -> tuple[TraceEvent, ...]:
        return tuple(e for e in self._events if e.event_type == event_type)

    def totals(self) -> tuple[int, int, float]:
        """(tokens_in, tokens_out, cost_est) across the whole session."""
        return (
            sum(e.tokens_in for e in self._events),
            sum(e.tokens_out for e in self._events),
            sum(e.cost_est for e in self._events),
        )

    def cache_totals(self) -> CacheSummary:
        """Cache accounting across the whole session.

        Separate from `totals()` rather than widening its tuple, because that
        tuple is part of the reading surface and a fourth element would break
        every caller that unpacks it.
        """
        return CacheSummary.from_events(self._events)


def read_trace(path: str) -> list[TraceEvent]:
    """Load a trace file. Blank lines are skipped; malformed ones are not.

    A trace with an unreadable line has lost part of the record, and a reader
    that shrugs past it produces a transcript that looks complete and isn't.
    """
    events: list[TraceEvent] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed trace line: {exc}") from None
            try:
                events.append(TraceEvent.from_dict(data))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from None
    return events


def group_sessions(events: list[TraceEvent]) -> Iterator[tuple[str, list[TraceEvent]]]:
    """Split a multi-session trace file by session_id, preserving order."""
    by_session: dict[str, list[TraceEvent]] = {}
    for event in events:
        by_session.setdefault(event.session_id, []).append(event)
    for session_id, session_events in by_session.items():
        yield session_id, session_events
