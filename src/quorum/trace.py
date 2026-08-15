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
from typing import Any, Callable, Iterator

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
        )


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
        )
        self._events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def of_type(self, event_type: str) -> tuple[TraceEvent, ...]:
        return tuple(e for e in self._events if e.event_type == event_type)

    def totals(self) -> tuple[int, int, float]:
        """(tokens_in, tokens_out, cost_est) across the whole session."""
        return (
            sum(e.tokens_in for e in self._events),
            sum(e.tokens_out for e in self._events),
            sum(e.cost_est for e in self._events),
        )


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
