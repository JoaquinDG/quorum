"""Reconstruct a session from its trace, and nothing else.

This module is the proof obligation behind the trace format. It imports no
session state, takes no arguments beyond a list of events, and rebuilds every
position, objection, diff, absence and verdict from the file. If something a
reader needs cannot be rebuilt here, the trace is incomplete — and the fix
belongs in `trace`/`session`, never in a renderer that quietly re-derives what
the record failed to keep.

The test suite runs a live session and a replay of its trace side by side and
compares them field by field. That test is the format's spec; this module is
just the first client of it, and the Session Report and the v2 replay world
will be the next two.

    python3 replay.py traces/session.jsonl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

from . import trace as tr
from .blinding import BlindingRound
from .sheets import AnswerSheet, Verdict, parse_sheet, parse_verdict


@dataclass
class ReplayedObjection:
    critic_seat: int
    target_seat: int
    claim_n: int
    argument: str
    sheet_label: str
    critic_model: str = ""
    target_model: str = ""
    claim_text: str = ""


@dataclass
class ReplayedStudent:
    seat: int
    model_id: str
    provider: str
    initial: AnswerSheet | None = None
    final: AnswerSheet | None = None
    diff: dict[str, Any] | None = None
    declared_change: bool = False
    because: tuple[dict[str, Any], ...] = ()
    position_change: dict[str, Any] | None = None
    absences: tuple[dict[str, Any], ...] = ()

    @property
    def label(self) -> str:
        return f"Student {self.seat}"

    @property
    def present(self) -> bool:
        return self.initial is not None

    @property
    def changed_position(self) -> bool:
        return bool(self.diff and self.diff.get("position_changed"))


@dataclass
class ReplayedSession:
    session_id: str = ""
    task: str = ""
    students: dict[int, ReplayedStudent] = field(default_factory=dict)
    arbiter: dict[str, Any] = field(default_factory=dict)
    objections: list[ReplayedObjection] = field(default_factory=list)
    blinding: dict[int, BlindingRound] = field(default_factory=dict)
    verdict: Verdict | None = None
    minority: list[dict[str, Any]] = field(default_factory=list)
    absences: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[dict[str, Any]] = field(default_factory=list)
    closed: dict[str, Any] = field(default_factory=dict)
    probes: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_est: float = 0.0

    @property
    def present_students(self) -> list[ReplayedStudent]:
        return [s for s in self.students.values() if s.present]

    @property
    def council_size(self) -> int:
        return len(self.present_students)

    @property
    def reduced_council(self) -> bool:
        return bool(self.closed.get("reduced_council"))

    @property
    def single_lab(self) -> bool:
        return bool(self.closed.get("single_lab"))

    @property
    def council_warnings(self) -> list[str]:
        return list(self.closed.get("council_warnings", []))

    @property
    def disagreement(self) -> dict[str, Any]:
        """Round-1 divergence, as recorded at session close. Lexical — see
        `divergence` for what that does and does not mean."""
        return dict(self.closed.get("disagreement", {}))

    @property
    def skeptic_seat(self) -> int | None:
        return self.closed.get("skeptic_seat")

    @property
    def baseline_model(self) -> str:
        return self.closed.get("baseline_model", "")

    @property
    def baseline_cost_est(self) -> float:
        return float(self.closed.get("baseline_cost_est") or 0.0)

    @property
    def cost_multiple(self) -> float | None:
        value = self.closed.get("cost_multiple")
        return None if value is None else float(value)

    @property
    def compliance_rate(self) -> float:
        attempts = self.closed.get("parse_attempts", 0)
        if not attempts:
            return 0.0
        return (attempts - self.closed.get("parse_failures", 0)) / attempts

    @property
    def dissent_preserved(self) -> bool:
        return bool(self.verdict and self.verdict.minority_report)

    @property
    def repair_cost_est(self) -> float:
        return sum(a.get("cost_est", 0.0) for a in self.discarded)

    @property
    def cost_is_complete(self) -> bool:
        return bool(self.closed.get("cost_is_complete", True))

    @property
    def unpriced_seats(self) -> list[str]:
        return list(self.closed.get("unpriced_seats", []))

    @property
    def failed_reason(self) -> str | None:
        return self.closed.get("failed_reason")

    @property
    def position_change_rate(self) -> float:
        revisers = [s for s in self.present_students if s.diff is not None]
        if not revisers:
            return 0.0
        return sum(1 for s in revisers if s.changed_position) / len(revisers)


def replay(events: list[tr.TraceEvent]) -> ReplayedSession:
    """Rebuild one session. Events must all share a `session_id`."""
    if not events:
        raise ValueError("cannot replay an empty trace")
    ids = {e.session_id for e in events}
    if len(ids) > 1:
        raise ValueError(
            f"trace holds {len(ids)} sessions; split with trace.group_sessions first"
        )

    session = ReplayedSession(session_id=events[0].session_id)
    for event in events:
        session.tokens_in += event.tokens_in
        session.tokens_out += event.tokens_out
        session.cost_est += event.cost_est
        payload = event.payload

        if event.event_type == tr.TASK_POSED:
            session.task = payload["task"]
            session.arbiter = dict(payload.get("arbiter", {}))
            for entry in payload.get("students", []):
                seat = int(entry["seat"])
                session.students[seat] = ReplayedStudent(
                    seat=seat,
                    model_id=entry["model_id"],
                    provider=entry["provider"],
                )

        elif event.event_type == tr.SHEET_SUBMITTED:
            student = session.students[int(payload["seat"])]
            sheet = parse_sheet(payload["sheet"], actor=event.actor)
            student.initial = sheet
            student.final = sheet

        elif event.event_type == tr.SHEETS_BLINDED:
            session.blinding[event.round] = BlindingRound.from_dict(
                {"salt": payload["salt"], "by_recipient": payload["mapping"]}
            )

        elif event.event_type == tr.OBJECTION_RAISED:
            session.objections.append(
                ReplayedObjection(
                    critic_seat=int(payload["critic_seat"]),
                    target_seat=int(payload["target_seat"]),
                    claim_n=int(payload["claim_n"]),
                    argument=payload["argument"],
                    sheet_label=payload["sheet_label"],
                    critic_model=payload.get("critic_model", ""),
                    target_model=payload.get("target_model", ""),
                    claim_text=payload.get("claim_text", ""),
                )
            )

        elif event.event_type == tr.SHEET_REVISED:
            student = session.students[int(payload["seat"])]
            student.final = parse_sheet(payload["sheet"], actor=event.actor)
            student.diff = dict(payload["diff"])
            student.declared_change = bool(payload.get("declared_change", False))
            student.because = tuple(payload.get("because", ()))

        elif event.event_type == tr.POSITION_CHANGED:
            session.students[int(payload["seat"])].position_change = dict(payload)

        elif event.event_type == tr.VERDICT_DELIVERED:
            sources = tuple(s.label for s in session.students.values())
            session.verdict = parse_verdict(
                payload["verdict"], allowed_sources=sources, actor=event.actor
            )

        elif event.event_type == tr.MINORITY_RECORDED:
            session.minority.append(dict(payload))

        elif event.event_type in (tr.STUDENT_ABSENT, tr.ARBITER_ABSENT):
            entry = {"round": event.round, "actor": event.actor, **payload}
            session.absences.append(entry)
            seat = payload.get("seat")
            if seat is not None:
                student = session.students[int(seat)]
                student.absences = student.absences + (entry,)

        elif event.event_type == tr.ATTEMPT_DISCARDED:
            session.discarded.append(
                {"round": event.round, "actor": event.actor, "cost_est": event.cost_est,
                 **payload}
            )

        elif event.event_type == tr.PROBE_RESULT:
            session.probes.append(dict(payload))

        elif event.event_type == tr.SESSION_CLOSED:
            session.closed = dict(payload)

    return session


def replay_file(path: str) -> list[ReplayedSession]:
    """Replay every session in a (possibly multi-session) trace file."""
    events = tr.read_trace(path)
    return [replay(group) for _, group in tr.group_sessions(events)]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _sheet_lines(sheet: AnswerSheet, indent: str = "    ") -> list[str]:
    lines = [f"{indent}position: {sheet.position}"]
    lines += [f"{indent}  {c.number}. {c.text}" for c in sheet.claims]
    lines.append(f"{indent}confidence: {sheet.confidence:.2f}")
    return lines


def render(session: ReplayedSession) -> str:
    """A plain-text transcript. Deliberately unstyled — this is the proof that
    the trace is complete, not the Session Report (that lands in Phase 2)."""
    out: list[str] = [
        f"session {session.session_id}",
        f"question: {session.task}",
        "",
        f"council: {session.council_size} of {len(session.students)} students answered"
        + (" (REDUCED COUNCIL)" if session.reduced_council else ""),
        f"arbiter: {session.arbiter.get('model_id', '?')}",
        "",
        "=== ROUND 1 — silent exam ===",
    ]
    for student in session.students.values():
        if not student.present:
            out.append(f"  {student.label} ({student.model_id}): ABSENT")
            continue
        out.append(f"  {student.label} ({student.model_id}):")
        out += _sheet_lines(student.initial)  # type: ignore[arg-type]

    out += ["", "=== ROUND 2 — blind claim-level critique ==="]
    if session.objections:
        for objection in session.objections:
            critic = session.students[objection.critic_seat].label
            target = session.students[objection.target_seat].label
            out.append(
                f"  {critic} -> {target} claim {objection.claim_n} "
                f"(seen as Sheet {objection.sheet_label}):"
            )
            out.append(f"      {objection.argument}")
    else:
        out.append("  (no objections recorded)")

    out += ["", "=== ROUND 3 — revision ==="]
    for student in session.students.values():
        if not student.present:
            continue
        if student.diff is None:
            out.append(f"  {student.label}: no revision submitted; opening sheet stands")
            continue
        moved = "CHANGED POSITION" if student.changed_position else "held position"
        out.append(f"  {student.label} ({student.model_id}): {moved}")
        if student.changed_position and student.position_change:
            out.append(f"      from: {student.position_change['from']}")
            out.append(f"      to:   {student.position_change['to']}")
        for dropped in student.diff.get("claims_dropped", []):
            out.append(f"      withdrew claim: {dropped['text']}")
        for added in student.diff.get("claims_added", []):
            out.append(f"      added claim: {added['text']}")
        for edited in student.diff.get("claims_edited", []):
            out.append(f"      edited claim {edited['before']['n']}: {edited['after']['text']}")
        delta = student.diff.get("confidence_delta", 0.0)
        if abs(delta) > 1e-9:
            out.append(f"      confidence {delta:+.2f}")
        if not student.diff.get("declaration_matches_diff", True):
            out.append(
                "      NOTE: declared change does not match the computed diff "
                f"(declared={student.declared_change})"
            )

    out += ["", "=== ROUND 4 — grading ==="]
    if session.verdict:
        out.append(f"  final answer: {session.verdict.final_answer}")
        out.append(f"  confidence:   {session.verdict.confidence_note}")
        if session.verdict.minority_report:
            out.append("  minority report:")
            for item in session.verdict.minority_report:
                out.append(f"    - [{item.source}, {item.kind}] {item.substance}")
        else:
            out.append("  minority report: (empty — nothing was recorded as left out)")
    else:
        out.append(f"  NO VERDICT: {session.failed_reason or 'unknown'}")

    if session.discarded:
        out += ["", "=== discarded attempts (paid for, thrown away) ==="]
        for attempt in session.discarded:
            out.append(
                f"  round {attempt['round']} {attempt['actor']}: "
                f"{attempt['reason']} — ${attempt['cost_est']:.4f}"
            )

    if session.absences:
        out += ["", "=== absences ==="]
        for absence in session.absences:
            out.append(
                f"  round {absence['round']} {absence['actor']}: "
                f"{absence['reason']} — {absence.get('detail', '')}"
            )

    out += [
        "",
        f"position-change rate: {session.position_change_rate:.0%}",
        f"tokens: {session.tokens_in} in / {session.tokens_out} out",
        f"estimated cost: ${session.cost_est:.4f}"
        + (
            ""
            if session.cost_is_complete
            else "  (LOWER BOUND — unpriced: "
            + ", ".join(session.unpriced_seats)
            + ")"
        ),
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 replay.py",
        description="Reconstruct Quorum sessions from a JSONL trace.",
    )
    parser.add_argument("trace", help="path to a .jsonl trace file")
    parser.add_argument(
        "--session", help="only replay this session_id", default=None
    )
    args = parser.parse_args(argv)

    sessions = replay_file(args.trace)
    if args.session:
        sessions = [s for s in sessions if s.session_id == args.session]
        if not sessions:
            print(f"no session {args.session!r} in {args.trace}", file=sys.stderr)
            return 1
    for index, session in enumerate(sessions):
        if index:
            print("\n" + "=" * 72 + "\n")
        print(render(session))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
