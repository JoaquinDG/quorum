"""Continue an interrupted debate instead of re-running and re-billing it.

A session that dies in round 3 has already paid for rounds 1 and 2 — nine
model calls, most of a debate's input tokens — and re-running it from the
question pays for all of them a second time to obtain answers already on
disk. Nothing about that spend buys anything: the sheets were valid, the
objections were recorded, and the trace has them.

Resume reuses the canonical trace as the checkpoint store rather than adding a
parallel one. That is not only tidier, it is the only version that stays
correct: a second store would have to be kept in step with the trace, and the
first time it drifted, a resumed session would continue from a state the
record does not describe.

**The session id is the load-bearing detail.** Blinding labels and shield
fence nonces both derive from it, so resuming under the recorded id is what
makes round 3's prompts byte-identical to the ones the interrupted run would
have sent — which keeps the debate coherent and lets the caching work apply
across the interruption rather than starting cold.

What resume will not do is hide that it happened. A resumed session emits a
`session_resumed` event naming the round it restarted at, and the final trace
is schema-identical to an uninterrupted one in every other respect.
"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from typing import Any, Iterable

from . import trace as tr
from .sheets import AnswerSheet, parse_sheet


class ResumeError(RuntimeError):
    """The session cannot be resumed from what is on disk."""


class SessionLocked(ResumeError):
    """Another process is already resuming this session.

    Raised rather than waited on. Two workers continuing the same debate would
    each re-run the remaining rounds, double-billing exactly the spend this
    module exists to avoid, and would interleave their events into one trace
    that then describes a session that never happened.
    """


@dataclass(frozen=True)
class Checkpoint:
    """Where an interrupted session got to."""

    session_id: str
    task: str
    last_round: int
    complete: bool

    @property
    def resumable(self) -> bool:
        return not self.complete and self.last_round >= 1


def inspect(events: Iterable[tr.TraceEvent]) -> Checkpoint:
    """Read a trace and say where it stopped.

    `last_round` is the highest round with a `round_completed` marker, so it
    reflects rounds that finished rather than rounds that were started. A run
    killed halfway through round 2 resumes *at* round 2 and re-asks the seats
    that never answered; it does not credit itself with a round that was
    interrupted mid-flight.
    """
    events = list(events)
    if not events:
        raise ResumeError("empty trace: nothing to resume")
    session_id = events[0].session_id
    if any(e.session_id != session_id for e in events):
        raise ResumeError("trace holds more than one session; split it first")
    task = ""
    last_round = 0
    complete = False
    for event in events:
        if event.event_type == tr.TASK_POSED:
            task = str(event.payload.get("task", ""))
        elif event.event_type == tr.ROUND_COMPLETED:
            last_round = max(last_round, int(event.payload.get("round", 0)))
        elif event.event_type == tr.SESSION_CLOSED:
            complete = True
    if not task:
        raise ResumeError("trace has no task_posed event; cannot resume")
    return Checkpoint(
        session_id=session_id, task=task, last_round=last_round, complete=complete
    )


class TraceLock:
    """An advisory lock beside the trace file.

    Deliberately the simplest thing that prevents the failure that matters:
    two resumes of the same session running the same rounds at once. Created
    with O_EXCL so the check and the claim are one operation — a check
    followed by a create is a race, and this lock exists to close a race.

    It is not a distributed lock and does not pretend to be. On a shared
    filesystem without working O_EXCL, or after a hard kill that leaves the
    file behind, `break_stale` is the documented escape hatch rather than a
    silent timeout that would reintroduce the double-run.
    """

    def __init__(self, path: str) -> None:
        self.path = path + ".lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise SessionLocked(
                    f"{self.path} exists, so another resume is in progress; "
                    "if that process died, remove the file to release it"
                ) from None
            raise
        os.write(self._fd, str(os.getpid()).encode())

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> TraceLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _sheet(payload: dict[str, Any], actor: str) -> AnswerSheet | None:
    raw = payload.get("sheet")
    if not isinstance(raw, dict):
        return None
    try:
        return parse_sheet(raw, actor=actor, allow_revision_fields=True)
    except Exception:  # noqa: BLE001 - a sheet the record cannot re-parse
        return None


def rebuild(events: Iterable[tr.TraceEvent], council: Any) -> dict[str, Any]:
    """Reconstruct the engine's own state from the recorded events.

    Returns the arguments the round functions take. Blinding is deliberately
    absent: it is derived from the session id and the present seats, so it is
    recomputed rather than restored — a value that can be recalculated exactly
    should never be carried, because a carried copy can be wrong and a derived
    one cannot.
    """
    from .session import Absence, Objection, StudentRecord

    events = list(events)
    records: dict[int, StudentRecord] = {
        seat: StudentRecord(
            seat=seat,
            model_id=council.student(seat).model_id,
            provider=council.student(seat).provider,
        )
        for seat in council.student_seats()
    }
    absences: list[Absence] = []
    objections: list[Objection] = []
    counters = {"attempts": 0, "failures": 0, "provider_errors": 0}

    for event in events:
        payload = event.payload
        kind = event.event_type
        if kind == tr.SHEET_SUBMITTED:
            seat = int(payload.get("seat", 0))
            if seat in records:
                sheet = _sheet(payload, event.actor)
                records[seat].initial = sheet
                records[seat].final = sheet
        elif kind == tr.SHEET_REVISED:
            seat = int(payload.get("seat", 0))
            if seat in records:
                revised = _sheet(payload, event.actor)
                if revised is not None:
                    records[seat].final = revised
                    records[seat].declared_change = bool(
                        payload.get("declared_change", False)
                    )
        elif kind == tr.OBJECTION_RAISED:
            objections.append(
                Objection(
                    critic_seat=int(payload.get("critic_seat", 0)),
                    target_seat=int(payload.get("target_seat", 0)),
                    claim_n=int(payload.get("claim_n", 0)),
                    argument=str(payload.get("argument", "")),
                    sheet_label=str(payload.get("sheet_label", "")),
                )
            )
        elif kind == tr.STUDENT_ABSENT:
            seat_no = payload.get("seat")
            absences.append(
                Absence(
                    seat=int(seat_no) if seat_no is not None else None,
                    actor=event.actor,
                    round=event.round,
                    reason=str(payload.get("reason", "")),
                    detail=str(payload.get("detail", "")),
                )
            )
            if isinstance(seat_no, int) and seat_no in records:
                record = records[seat_no]
                record.absent_rounds = tuple(
                    sorted(set(record.absent_rounds) | {event.round})
                )
        elif kind == tr.ATTEMPT_DISCARDED:
            counters["failures"] += 1
        if event.tokens_in or event.tokens_out:
            counters["attempts"] += 1

    return {
        "records": records,
        "absences": absences,
        "objections": objections,
        "counters": counters,
    }
