"""The session engine: four rounds, fixed, fail-closed.

Round 1 — silent exam.       Each student fills a sheet. Nobody sees anybody.
Round 2 — blind critique.    Sheets go out labelled A/B; objections come back
                             attached to numbered claims.
Round 3 — revision.          Each student answers the objections against it,
                             under a fresh critic blinding.
Round 4 — grading.           An arbiter that never debated synthesises, and
                             files a minority report for what it left out.

Three properties are enforced here rather than hoped for:

**Independence.** A round-1 prompt contains the task and nothing else. This is
the primary anti-sycophancy mechanism, and it is the one that costs nothing to
preserve and everything to lose — a single leaked peer answer converts three
independent priors into one prior and two echoes.

**Fixed rounds.** There is no "continue until they agree" loop. Unbounded
deliberation burns tokens to manufacture the consensus the protocol exists to
avoid manufacturing, and a session that cannot converge in one revision has
found real disagreement, which is the finding.

**Fail-closed, per round.** A student that errors or returns an unparseable
sheet is recorded absent for that round and the session says so. It is never
retried into compliance beyond the single documented re-prompt, never
silently coerced into a valid shape, and a session that ran with two students
is labelled a two-student session everywhere it surfaces. Absence is scoped to
the round: a student that fails round 1 has no sheet and is out; one that
fails round 2 raises no objections but still answers those raised against it;
one that fails round 3 keeps its original sheet as final.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import trace as tr
from .blinding import BlindingRound, build_blinding, invert
from .costs import Baseline, cost_multiple, single_model_baseline
from .council import Council, Seat
from .divergence import Disagreement, disagreement
from . import shield as shd
from .repair import CLEAN as CLEAN_REPAIR, RepairReport, recover
from .schemas import SCHEMA_FOR_ROUND
from .prompts import (
    build_critique_prompt,
    build_skeptic_critique_prompt,
    leading_sheet_label,
    build_critique_repair_prompt,
    build_revision_prompt,
    build_revision_repair_prompt,
    build_sheet_prompt,
    build_sheet_repair_prompt,
    build_verdict_prompt,
    build_verdict_repair_prompt,
    render_sheet,
)
from .providers.base import Completion, ProviderError, ProviderPool
from .sheets import (
    MAX_CLAIMS,
    MIN_ARGUMENT_CHARS,
    AnswerSheet,
    ObjectionRef,
    SheetDiff,
    SheetError,
    Verdict,
    diff_sheets,
    parse_critique,
    parse_revision,
    parse_sheet,
    parse_verdict,
)

ROUND_EXAM = 1
ROUND_CRITIQUE = 2
ROUND_REVISION = 3
ROUND_GRADING = 4

BLIND_SALT_CRITIQUE = "r2-sheets"
BLIND_SALT_REVISION = "r3-critics"


@dataclass(frozen=True)
class SessionConfig:
    min_council: int = 2
    """Below this many round-1 sheets there is no peer review to speak of, so
    the session closes without a verdict rather than presenting one model's
    opinion as a council's."""

    structured_output: bool = True
    """Ask providers that can constrain output to a schema to do so.

    Prevention rather than recovery: a schema-constrained reply cannot be the
    unterminated object that cost a seat its round. Feature-detected per
    provider — an adapter that does not advertise `schema_mode` is called
    exactly as before, so a lineup with one capable vendor and three
    incapable ones runs unchanged for the other three.

    The schemas restate the existing contract and nothing more, and the parser
    stays the authority regardless of what the provider promised."""

    repair_json: bool = True
    """Attempt deterministic local repair before spending a re-prompt.

    On by default because the repairs are syntactic and free: closing an
    object the model left open costs no inference and recovers a seat that
    would otherwise argue for nobody. Turn it off to reproduce the engine's
    pre-repair behaviour — which is how the malformed-seat fixture was
    recorded, and how the malformation rate is measured against a control.

    It never widens what counts as a valid response: `repair` only makes text
    parse, and the schema check that follows is unchanged. A seat that sent
    something genuinely wrong is still marked absent."""

    sheet_repairs: int = 1
    """Re-prompts allowed for a round-1 sheet that would not parse.

    Round 1 absences are the expensive ones — the student is gone from every
    later round, not just this one — and it was the last round with no budget
    at all."""

    critique_repairs: int = 1
    """Re-prompts allowed for a non-compliant critique. One: enough to rescue
    a good critic that missed a field, too few to let a model that cannot
    follow the format spend the session's budget failing."""

    revision_repairs: int = 1
    """Re-prompts allowed for a revised sheet that broke the schema.

    Was zero until a real model's revision was rejected for carrying six
    claims. Round 3 asks students to answer objections, answering adds
    material, and the cap forbids growth — so the round with the hardest
    schema to satisfy had the smallest budget for missing it."""

    verdict_repairs: int = 1

    max_tokens: int = 16384
    """Completion budget per call.

    Set by measurement, twice. 2048 was the guess; a round-1 sheet from a
    reasoning model cost 756 visible tokens, so 4096 looked generous. It was
    not: on a three-lab council the students raised 25 objections, and both a
    revision answering them and the arbiter reading the whole transcript blew
    straight through 4096. The session ended with no verdict.

    The load is not uniform across rounds and it grows with the *council's*
    output, not the question. An arbiter reads every sheet, every objection and
    every revision, then writes a synthesis plus a minority report holding one
    entry per surviving dissent — so its budget scales with how productive the
    debate was, which is exactly backwards from a fixed cap tuned on round 1.

    16384 is chosen because every seat tested accepts 32768, and because this
    is a cap rather than a reservation: you are billed for tokens produced, so
    headroom costs nothing until it is used. A per-round budget would be
    tighter and is not worth the configuration surface — the arbiter is the
    only round that needs the room, and it is one call."""

    skeptic_seat: int | None = None
    """Seat instructed to attack the most confident sheet regardless of whether
    it agrees. Off by default: it is an intervention whose effect on the
    position-change rate is an empirical question, and a protocol that ships
    an untested nudge as standard has stopped measuring itself."""

    baseline_model: str | None = None
    """Name a *seated* model to price the baseline against. Defaults to the
    most expensive seat — see `costs.pick_baseline_seat`. Naming a model that
    holds no seat is an error, because it is almost always a typo."""

    shield_policy: shd.ShieldPolicy = shd.DEFAULT_POLICY
    """How the shield treats text one participant wrote and another reads.

    Default fences, neutralizes forged structure, flags the rest and forwards
    everything. `shd.STRICT_POLICY` makes a structural forgery cost the author
    its seat for the session, which is the right trade when questions arrive
    from an inbox and the wrong one when the council is deliberating about
    prompt injection itself. `shd.OFF` restores the pre-shield wire format,
    which is what a trace comparison against an older run needs."""

    baseline_seat: Any = None
    """A `Seat` to price the baseline against, seated or not.

    Needed to compare lineups: `baseline_model` cannot express "one ruler for
    every arm" when the arms differ, because the moment a lineup drops that
    model the run either crashes or quietly switches yardsticks. Comparing
    councils requires a baseline that is independent of who is in them."""


@dataclass(frozen=True)
class Objection:
    """An objection in canonical form: seats, not labels.

    Labels are per-viewer and per-round, so storing "Sheet A" would make the
    record unreadable without also knowing who was reading. The label the
    critic actually wrote is kept alongside, because the report wants to show
    the objection as it was written.
    """

    critic_seat: int
    target_seat: int
    claim_n: int
    argument: str
    sheet_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic_seat": self.critic_seat,
            "target_seat": self.target_seat,
            "claim_n": self.claim_n,
            "argument": self.argument,
            "sheet_label": self.sheet_label,
        }


@dataclass(frozen=True)
class Absence:
    seat: int | None
    actor: str
    round: int
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "actor": self.actor,
            "round": self.round,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class StudentRecord:
    seat: int
    model_id: str
    provider: str
    initial: AnswerSheet | None = None
    final: AnswerSheet | None = None
    diff: SheetDiff | None = None
    declared_change: bool = False
    because: tuple[ObjectionRef, ...] = ()
    absent_rounds: tuple[int, ...] = ()

    @property
    def present(self) -> bool:
        return self.initial is not None

    @property
    def changed_position(self) -> bool:
        return bool(self.diff and self.diff.position_changed)

    @property
    def label(self) -> str:
        return f"Student {self.seat}"


@dataclass
class SessionResult:
    session_id: str
    task: str
    council: Council
    students: tuple[StudentRecord, ...]
    objections: tuple[Objection, ...]
    verdict: Verdict | None
    blinding: dict[int, BlindingRound]
    absences: tuple[Absence, ...]
    events: tuple[tr.TraceEvent, ...]
    tokens_in: int
    tokens_out: int
    cost_est: float
    cache: tr.CacheSummary = field(default_factory=tr.CacheSummary)
    """Prompt-cache accounting for this session.

    Defaulted rather than required so a result built by an older caller (or a
    replay of a trace written before cache fields existed) still constructs,
    and reports an all-zero summary — no caching observed, which is the
    truthful reading of a record that does not mention it."""
    failed_reason: str | None = None
    parse_attempts: int = 0
    parse_failures: int = 0
    provider_errors: int = 0
    baseline: Baseline | None = None
    disagreement: Disagreement | None = None
    findings: tuple[shd.Finding, ...] = ()
    """Everything the shield noticed in text that crossed between participants.

    Non-empty does not mean the session is compromised. It means somebody
    wrote something that reads like an instruction to its reader, which is
    worth a human's eye and is exactly the kind of thing a transcript-first
    tool should refuse to drop on the floor."""

    # -- headline properties ----------------------------------------------

    @property
    def ok(self) -> bool:
        return self.failed_reason is None and self.verdict is not None

    @property
    def flagged(self) -> bool:
        """Did anything trip the shield this session?"""
        return bool(self.findings)

    @property
    def worst_finding(self) -> str:
        return shd.worst(self.findings)

    def findings_by_actor(self) -> dict[str, tuple[shd.Finding, ...]]:
        """Who wrote the flagged text. The question a reader asks first."""
        out: dict[str, list[shd.Finding]] = {}
        for f in self.findings:
            out.setdefault(f.actor, []).append(f)
        return {actor: tuple(items) for actor, items in sorted(out.items())}

    @property
    def present_students(self) -> tuple[StudentRecord, ...]:
        return tuple(s for s in self.students if s.present)

    @property
    def reduced_council(self) -> bool:
        """True when fewer students answered than were seated. Every surface
        that shows a verdict must also show this; a two-student session
        presented as a full council is the one lie the protocol could tell."""
        return len(self.present_students) < len(self.council.students)

    @property
    def council_size(self) -> int:
        return len(self.present_students)

    @property
    def position_change_rate(self) -> float:
        """Share of students whose *computed* position diff moved.

        Computed, never declared. A model that says it reconsidered while
        resubmitting the same sentence is the exact failure this number is
        supposed to catch, so taking its word would make the metric agree with
        the pathology it measures."""
        revisers = [s for s in self.present_students if s.diff is not None]
        if not revisers:
            return 0.0
        return sum(1 for s in revisers if s.diff.position_changed) / len(revisers)

    @property
    def dissent_preserved(self) -> bool:
        return bool(self.verdict and self.verdict.minority_report)

    @property
    def compliance_rate(self) -> float:
        """Share of model responses that parsed into a valid, claim-referencing
        structure.

        Availability is not compliance. A provider outage produces no text to
        judge, so it is counted in `provider_errors` and kept out of this
        denominator entirely — otherwise a 503 would show up as the council
        failing to follow the format, and the published claim-compliance
        number would be measuring the platform's uptime.
        """
        if not self.parse_attempts:
            return 0.0
        return (self.parse_attempts - self.parse_failures) / self.parse_attempts

    @property
    def compliance_by_model(self) -> dict[str, dict[str, int]]:
        """Schema failures per model, from the trace.

        The session-wide rate hides the thing you can act on. A real three-lab
        run came in at 69% compliance, and every single failure belonged to one
        seat — the other two were perfect. Averaged together that reads as "the
        council struggles with the format"; broken out it reads as "one model
        cannot emit strict JSON reliably", which is a seat you can replace.

        It also prices the schema honestly. A cheap model that needs a repair
        on half its turns is not cheap, and this is where that shows up.
        """
        stats: dict[str, dict[str, int]] = {}
        counted = {
            tr.SHEET_SUBMITTED, tr.SHEET_REVISED, tr.VERDICT_DELIVERED,
            tr.ATTEMPT_DISCARDED, tr.STUDENT_ABSENT, tr.ARBITER_ABSENT,
        }
        failed = {tr.ATTEMPT_DISCARDED, tr.STUDENT_ABSENT, tr.ARBITER_ABSENT}
        for event in self.events:
            if event.event_type not in counted:
                continue
            model = event.payload.get("model_id")
            if not model:
                continue
            entry = stats.setdefault(model, {"attempts": 0, "failures": 0})
            entry["attempts"] += 1
            if event.event_type in failed:
                entry["failures"] += 1
        return stats

    @property
    def worst_complier(self) -> str | None:
        """The seat costing the most repairs, or None if everyone complied."""
        ranked = [
            (v["failures"], m) for m, v in self.compliance_by_model.items() if v["failures"]
        ]
        return max(ranked)[1] if ranked else None

    @property
    def discarded_calls(self) -> tuple[tr.TraceEvent, ...]:
        """Model calls that were paid for and thrown away (re-prompts)."""
        return tuple(e for e in self.events if e.event_type == tr.ATTEMPT_DISCARDED)

    @property
    def single_lab(self) -> bool:
        """Every student came from one provider. Correlated blind spots; see
        `Council.single_lab`. Surfaced everywhere the verdict is."""
        return self.council.single_lab

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.council.warnings

    @property
    def cost_multiple(self) -> float | None:
        """Session cost as a multiple of one single-model answer.

        `None` when the baseline seat carries no price — a missing number is
        information; a fabricated `1.0x` is not."""
        if self.baseline is None:
            return None
        return cost_multiple(self.cost_est, self.baseline)

    @property
    def unpriced_seats(self) -> tuple[str, ...]:
        """Model ids that took part but carry no price.

        `cost_est` is a lower bound whenever this is non-empty, and any
        surface that prints the cost has to say so — a report that shows
        "$0.0031" for a session where the arbiter was unpriced is not
        approximately right, it is wrong in the reassuring direction.
        """
        return tuple(
            seat.model_id for seat in self.council.seats() if not seat.cost.priced
        )

    @property
    def cost_is_complete(self) -> bool:
        return not self.unpriced_seats

    @property
    def repair_cost_est(self) -> float:
        """What the re-prompts cost. Small in a healthy session; if it isn't,
        the schema is fighting the models rather than disciplining them."""
        return sum(e.cost_est for e in self.discarded_calls)

    def objections_against(self, seat: int) -> tuple[Objection, ...]:
        return tuple(o for o in self.objections if o.target_seat == seat)

    def objections_by(self, seat: int) -> tuple[Objection, ...]:
        return tuple(o for o in self.objections if o.critic_seat == seat)

    def student(self, seat: int) -> StudentRecord:
        for record in self.students:
            if record.seat == seat:
                return record
        raise KeyError(f"no student in seat {seat}")

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "council_size": self.council_size,
            "reduced_council": self.reduced_council,
            "single_lab": self.single_lab,
            "warnings": list(self.warnings),
            "objections": len(self.objections),
            "position_change_rate": round(self.position_change_rate, 4),
            "dissent_preserved": self.dissent_preserved,
            "disagreement_score": (
                round(self.disagreement.score, 4) if self.disagreement else None
            ),
            "disagreement_label": self.disagreement.label if self.disagreement else None,
            "compliance_rate": round(self.compliance_rate, 4),
            "provider_errors": self.provider_errors,
            "compliance_by_model": self.compliance_by_model,
            "worst_complier": self.worst_complier,
            "discarded_calls": len(self.discarded_calls),
            "repair_cost_est": self.repair_cost_est,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_est": self.cost_est,
            "cache": self.cache.to_dict(),
            "baseline_model": self.baseline.model_id if self.baseline else None,
            "baseline_cost_est": self.baseline.cost_est if self.baseline else None,
            "cost_multiple": self.cost_multiple,
            "cost_is_complete": self.cost_is_complete,
            "unpriced_seats": list(self.unpriced_seats),
            "ok": self.ok,
            "failed_reason": self.failed_reason,
        }


class Session:
    """Runs one deliberation. Stateless between `run` calls except the trace."""

    def __init__(
        self,
        council: Council,
        providers: ProviderPool,
        *,
        trace_path: str | None = None,
        config: SessionConfig | None = None,
        clock: Callable[[], float] = time.time,
        writer: tr.TraceWriter | None = None,
    ) -> None:
        self.council = council
        self.providers = providers
        self.config = config or SessionConfig()
        self._clock = clock
        self._session_counter = 0
        self.writer = writer or tr.TraceWriter(trace_path, clock=clock)
        missing = [
            seat.provider
            for seat in council.seats()
            if not providers.has(seat.provider)
        ]
        if missing:
            # Discovering a missing provider three rounds in means paying for
            # two rounds of a session that cannot finish.
            raise KeyError(
                f"no provider registered for {sorted(set(missing))}; "
                f"available: {providers.names()}"
            )

    # -- plumbing ----------------------------------------------------------

    def _new_session_id(self, task: str) -> str:
        # The counter is not decoration. Two sessions inside one clock tick —
        # a fast loop, a benchmark harness, or any injected clock — would
        # otherwise share an id, and `replay` refuses a file whose events
        # belong to two sessions under one name. A collision would therefore
        # not corrupt one report; it would make both unreadable.
        self._session_counter += 1
        material = (
            f"{task}|{[s.model_id for s in self.council.seats()]}"
            f"|{self._clock()}|{self._session_counter}"
        )
        return "q-" + hashlib.blake2b(
            material.encode("utf-8"), digest_size=6
        ).hexdigest()

    def _recover(self, text: str, actor: str) -> tuple[Any, RepairReport]:
        """Decode a response, repairing it first when repair is enabled.

        The single gate for every parse site, so "repair is off" means exactly
        the pre-repair engine on all four rounds rather than three of them.
        """
        if not self.config.repair_json:
            return text, CLEAN_REPAIR
        return recover(text, actor=actor)

    def _call(self, seat: Seat, prompt: str, round: int = 0) -> Completion:
        provider = self.providers.get(seat.provider)
        schema = self._schema_for(provider, round)
        if schema is None:
            return provider.complete(seat.model_id, prompt, self.config.max_tokens)
        return provider.complete(
            seat.model_id, prompt, self.config.max_tokens, schema=schema
        )

    def _schema_for(self, provider: Any, round: int) -> tuple[str, dict] | None:
        """The schema to constrain this round with, or None to ask in prose.

        Feature detection rather than a registry of vendor names: a provider
        advertises `schema_mode`, and anything that does not is called through
        the original one-method signature. Third-party adapters written before
        this existed keep working untouched, which is the property that makes
        the fallback safe rather than merely present.
        """
        if not self.config.structured_output:
            return None
        if not getattr(provider, "schema_mode", None):
            return None
        return SCHEMA_FOR_ROUND.get(round)

    def _emit(
        self,
        session_id: str,
        round: int,
        actor: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        completion: Completion | None = None,
        seat: Seat | None = None,
    ) -> None:
        tokens_in = completion.input_tokens if completion else 0
        tokens_out = completion.output_tokens if completion else 0
        cache_read = completion.cache_read_tokens if completion else 0
        cache_write = completion.cache_write_tokens if completion else 0
        if seat and completion:
            cost = seat.cost.estimate(tokens_in, tokens_out, cache_read, cache_write)
            # Priced on the same seat, so the pair is comparable by
            # construction: the only difference between them is the cache.
            uncached = seat.cost.uncached_estimate(
                tokens_in, tokens_out, cache_read, cache_write
            )
        else:
            cost = uncached = 0.0
        self.writer.emit(
            session_id=session_id,
            round=round,
            actor=actor,
            event_type=event_type,
            payload=payload or {},
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_est=cost,
            cache_read=cache_read,
            cache_write=cache_write,
            uncached_cost_est=uncached,
        )

    def _nonce(self, session_id: str, recipient: Any, round: int) -> str:
        """The fence marker for one reader in one round.

        Empty when fencing is off, which is how `shd.OFF` gets the old prompt
        bytes back without a second code path through the builders."""
        policy = self.config.shield_policy
        if not (policy.enabled and policy.fence):
            return ""
        return shd.fence_nonce(session_id, recipient=recipient, round=round)

    def _flag(
        self,
        session_id: str,
        round: int,
        actor: str,
        findings: tuple[shd.Finding, ...],
    ) -> tuple[shd.Finding, ...]:
        """Record what the shield saw. One event per author per round.

        Batched rather than one event per finding because a single hostile
        field trips several patterns at once, and a trace where one sheet
        produces nine events reads like nine attacks."""
        # A disabled shield does not scan either. Flagging without fencing
        # would leave `shd.OFF` producing the old prompt bytes and a new class
        # of trace event, which is neither of the two things anybody turning
        # it off is asking for.
        if not self.config.shield_policy.enabled or not findings:
            return ()
        self._emit(
            session_id,
            round,
            actor,
            tr.INJECTION_FLAGGED,
            {
                "worst": shd.worst(findings),
                "count": len(findings),
                "findings": [f.to_dict() for f in findings],
                "action": "flagged; structural forgeries neutralized in the "
                "rendered prompt, wording left as written",
            },
        )
        return findings

    def _checkpoint(
        self,
        session_id: str,
        round: int,
        records: dict[int, StudentRecord],
        absences: list[Absence],
        objections: list[Objection],
    ) -> None:
        """Mark a round finished, so resume knows what has been paid for.

        The payload is a summary, not a second copy of the state. Everything
        needed to rebuild a session is already in the events this marker
        follows, and duplicating it here would create a parallel format that
        can disagree with the canonical one — the exact failure the resume
        design was told to avoid.
        """
        self._emit(
            session_id,
            round,
            tr.SYSTEM,
            tr.ROUND_COMPLETED,
            {
                "round": round,
                "present": [s for s in sorted(records) if records[s].present],
                "absent": sorted({a.seat for a in absences if a.seat is not None}),
                "objections": len(objections),
            },
        )

    # -- the protocol ------------------------------------------------------

    def resume(
        self,
        events: list[tr.TraceEvent],
        *,
        lock_path: str | None = None,
    ) -> SessionResult:
        """Continue an interrupted session from its last completed round.

        Rounds already in the trace are not re-run and not re-billed. Rounds
        that remain run live, under the recorded session id so their prompts
        are byte-identical to the ones the interrupted process would have
        sent.

        Resuming a session that already closed is a no-op: the recorded result
        is rebuilt and returned without a single model call, so a retry loop
        that fires twice cannot turn a finished debate into a second bill.
        """
        from .resume import ResumeError, TraceLock, inspect, rebuild

        checkpoint = inspect(events)
        session_id = checkpoint.session_id
        lock = TraceLock(lock_path or self.writer.path or session_id)

        with lock:
            first_event = len(self.writer.events)
            self.writer.adopt(events)

            state = rebuild(events, self.council)
            records: dict[int, StudentRecord] = state["records"]
            absences: list[Absence] = state["absences"]
            objections: list[Objection] = state["objections"]
            counters: dict[str, int] = state["counters"]
            findings: list[shd.Finding] = []
            blinding: dict[int, BlindingRound] = {}
            task = checkpoint.task

            if checkpoint.complete:
                # Idempotent. The debate is over; rebuilding it costs nothing
                # and re-running it would cost everything twice.
                return self._close(
                    session_id, task, records, objections, self._recorded_verdict(events),
                    blinding, absences, first_event, counters, findings,
                    failed_reason=None,
                )
            if not checkpoint.resumable:
                raise ResumeError(
                    f"session {session_id} has no completed round to resume from"
                )

            self._emit(
                session_id,
                checkpoint.last_round,
                tr.SYSTEM,
                tr.SESSION_RESUMED,
                {
                    "resumed_after_round": checkpoint.last_round,
                    "rounds_replayed": checkpoint.last_round,
                    "note": "rounds at or below resumed_after_round were read "
                            "from the trace and were not re-billed",
                },
            )

            present = [s for s in self.council.student_seats() if records[s].present]
            if len(present) < self.config.min_council:
                return self._close(
                    session_id, task, records, objections, None, blinding, absences,
                    first_event, counters, findings,
                    failed_reason=(
                        f"only {len(present)} of {len(self.council.students)} "
                        f"students submitted a sheet; a council of at least "
                        f"{self.config.min_council} is required for peer review"
                    ),
                )

            # Blinding is recomputed, never restored: same session id, same
            # seats, same salt, so it comes back identical by construction.
            if checkpoint.last_round < ROUND_CRITIQUE:
                blinding[ROUND_CRITIQUE] = build_blinding(
                    session_id, present, salt=BLIND_SALT_CRITIQUE
                )
                self._round2(
                    session_id, task, records, present, blinding[ROUND_CRITIQUE],
                    objections, absences, counters, findings,
                )
                self._checkpoint(
                    session_id, ROUND_CRITIQUE, records, absences, objections
                )

            if checkpoint.last_round < ROUND_REVISION:
                blinding[ROUND_REVISION] = build_blinding(
                    session_id, present, salt=BLIND_SALT_REVISION
                )
                self._round3(
                    session_id, task, records, present, blinding[ROUND_REVISION],
                    objections, absences, counters, findings,
                )
                self._checkpoint(
                    session_id, ROUND_REVISION, records, absences, objections
                )

            verdict, failure = self._round4(
                session_id, task, records, present, objections, absences, counters,
                findings,
            )
            return self._close(
                session_id, task, records, objections, verdict, blinding, absences,
                first_event, counters, findings, failed_reason=failure,
            )

    def _recorded_verdict(self, events: list[tr.TraceEvent]) -> Verdict | None:
        """The verdict a finished session already reached, read back."""
        from .sheets import parse_verdict

        for event in events:
            if event.event_type == tr.VERDICT_DELIVERED:
                raw = event.payload.get("verdict")
                if isinstance(raw, dict):
                    sources = tuple(
                        str(item.get("source", ""))
                        for item in raw.get("minority_report", [])
                    )
                    try:
                        return parse_verdict(
                            raw, allowed_sources=sources or ("Student 1",),
                            actor=tr.ARBITER,
                        )
                    except Exception:  # noqa: BLE001 - record wins over strictness
                        return None
        return None

    def run(self, task: str, *, session_id: str | None = None) -> SessionResult:
        if not task or not task.strip():
            raise ValueError("a session needs a question")
        session_id = session_id or self._new_session_id(task)
        first_event = len(self.writer.events)

        records = {
            seat: StudentRecord(
                seat=seat,
                model_id=self.council.student(seat).model_id,
                provider=self.council.student(seat).provider,
            )
            for seat in self.council.student_seats()
        }
        absences: list[Absence] = []
        objections: list[Objection] = []
        blinding: dict[int, BlindingRound] = {}
        counters = {"attempts": 0, "failures": 0, "provider_errors": 0}
        findings: list[shd.Finding] = []

        self._emit(
            session_id,
            0,
            tr.SYSTEM,
            tr.TASK_POSED,
            {
                "task": task,
                "students": [
                    {"seat": s, "model_id": records[s].model_id, "provider": records[s].provider}
                    for s in self.council.student_seats()
                ],
                "arbiter": {
                    "model_id": self.council.arbiter.model_id,
                    "provider": self.council.arbiter.provider,
                },
                "labs": list(self.council.labs()),
                "single_lab": self.council.single_lab,
                "council_warnings": list(self.council.warnings),
                "config": {
                    "min_council": self.config.min_council,
                    "critique_repairs": self.config.critique_repairs,
                    "max_tokens": self.config.max_tokens,
                },
            },
        )

        # The question is scanned before anybody is asked to answer it. It is
        # the only input that does not come from a model, which is exactly why
        # it is the one people forget: a task pasted out of a ticket or a
        # retrieved document arrives carrying whatever that document carried,
        # and round 1 hands it to every seat at once.
        findings.extend(self._flag(session_id, 0, tr.SYSTEM, shd.scan_task(task)))

        self._round1(session_id, task, records, absences, counters, findings)
        self._checkpoint(session_id, ROUND_EXAM, records, absences, objections)

        present = [s for s in self.council.student_seats() if records[s].present]
        if len(present) < self.config.min_council:
            return self._close(
                session_id,
                task,
                records,
                objections,
                None,
                blinding,
                absences,
                first_event,
                counters,
                findings,
                failed_reason=(
                    f"only {len(present)} of {len(self.council.students)} students "
                    f"submitted a sheet; a council of at least "
                    f"{self.config.min_council} is required for peer review"
                ),
            )

        blinding[ROUND_CRITIQUE] = build_blinding(
            session_id, present, salt=BLIND_SALT_CRITIQUE
        )
        self._round2(
            session_id, task, records, present, blinding[ROUND_CRITIQUE], objections,
            absences, counters, findings,
        )

        self._checkpoint(session_id, ROUND_CRITIQUE, records, absences, objections)

        blinding[ROUND_REVISION] = build_blinding(
            session_id, present, salt=BLIND_SALT_REVISION
        )
        self._round3(
            session_id, task, records, present, blinding[ROUND_REVISION], objections,
            absences, counters, findings,
        )

        self._checkpoint(session_id, ROUND_REVISION, records, absences, objections)

        verdict, failure = self._round4(
            session_id, task, records, present, objections, absences, counters,
            findings,
        )

        return self._close(
            session_id,
            task,
            records,
            objections,
            verdict,
            blinding,
            absences,
            first_event,
            counters,
            findings,
            failed_reason=failure,
        )

    # -- round 1 -----------------------------------------------------------

    def _round1(
        self,
        session_id: str,
        task: str,
        records: dict[int, StudentRecord],
        absences: list[Absence],
        counters: dict[str, int],
        findings: list[shd.Finding],
    ) -> None:
        prompt = build_sheet_prompt(task)
        for seat_no in self.council.student_seats():
            seat = self.council.student(seat_no)
            actor = tr.student_actor(seat_no)
            try:
                completion = self._call(seat, prompt, ROUND_EXAM)
            except ProviderError as exc:
                counters["provider_errors"] += 1
                self._absent(
                    session_id, ROUND_EXAM, seat_no, actor, "provider_error", str(exc),
                    absences, records,
                )
                continue
            counters["attempts"] += 1
            sheet = None
            last_error = ""
            last_completion = completion
            repair_report = CLEAN_REPAIR
            for attempt in range(self.config.sheet_repairs + 1):
                if attempt:
                    try:
                        completion = self._call(seat, attempt_prompt, ROUND_EXAM)
                    except ProviderError as exc:
                        counters["provider_errors"] += 1
                        last_error = str(exc)
                        break
                    counters["attempts"] += 1
                    last_completion = completion
                try:
                    # Repair before re-asking: a response that was merely cut
                    # off is recoverable here for nothing, and a seat that
                    # survives is a seat that keeps arguing.
                    data, repair_report = self._recover(completion.text, actor)
                    sheet = parse_sheet(data, actor=actor)
                    break
                except SheetError as exc:
                    counters["failures"] += 1
                    last_error = str(exc)
                    sheet = None
                    if attempt < self.config.sheet_repairs:
                        self._discarded(
                            session_id, ROUND_EXAM, actor, seat_no, seat,
                            completion, "malformed_sheet", last_error,
                        )
                        attempt_prompt = build_sheet_repair_prompt(prompt, last_error)

            if sheet is None:
                self._absent(
                    session_id, ROUND_EXAM, seat_no, actor, "malformed_sheet", last_error,
                    absences, records, completion=last_completion, seat=seat,
                )
                continue

            # Scanned on arrival rather than at render time. A sheet is
            # flagged for what its author wrote, and waiting until round 2
            # would attribute it to the round in which somebody else read it.
            findings.extend(
                self._flag(
                    session_id, ROUND_EXAM, actor, shd.scan_sheet(sheet, actor=actor)
                )
            )

            records[seat_no].initial = sheet
            records[seat_no].final = sheet  # stands unless round 3 replaces it
            self._emit(
                session_id,
                ROUND_EXAM,
                actor,
                tr.SHEET_SUBMITTED,
                {
                    "seat": seat_no,
                    "model_id": seat.model_id,
                    "provider": seat.provider,
                    "sheet": sheet.to_dict(),
                    **repair_report.as_payload(),
                },
                completion=completion,
                seat=seat,
            )

    # -- round 2 -----------------------------------------------------------

    def _round2(
        self,
        session_id: str,
        task: str,
        records: dict[int, StudentRecord],
        present: list[int],
        blinding: BlindingRound,
        objections: list[Objection],
        absences: list[Absence],
        counters: dict[str, int],
        findings: list[shd.Finding],
    ) -> None:
        self._emit(
            session_id,
            ROUND_CRITIQUE,
            tr.SYSTEM,
            tr.SHEETS_BLINDED,
            {
                "salt": blinding.salt,
                "mapping": blinding.to_dict()["by_recipient"],
                "note": "label -> seat, per recipient; participants see only labels",
            },
        )

        for seat_no in present:
            seat = self.council.student(seat_no)
            actor = tr.student_actor(seat_no)
            mapping = blinding.by_recipient[seat_no]
            blinded = {
                label: records[target].initial for label, target in mapping.items()
            }
            allowed = {
                label: records[target].initial.claim_numbers
                for label, target in mapping.items()
            }
            # One nonce per critic. Seat 2 is never shown the marker that
            # closes seat 3's fence, so a sheet seat 2 wrote in round 1 cannot
            # close a block it will never be handed.
            nonce = self._nonce(session_id, seat_no, ROUND_CRITIQUE)
            policy = self.config.shield_policy
            if self.config.skeptic_seat == seat_no:
                prompt = build_skeptic_critique_prompt(
                    task, blinded, leading_sheet_label(blinded),  # type: ignore[arg-type]
                    nonce=nonce, policy=policy,
                )
            else:
                prompt = build_critique_prompt(
                    task, blinded, nonce=nonce, policy=policy  # type: ignore[arg-type]
                )

            raw = None
            last_error = ""
            last_completion: Completion | None = None
            repair_report = CLEAN_REPAIR
            attempt_prompt = prompt
            for attempt in range(self.config.critique_repairs + 1):
                try:
                    completion = self._call(seat, attempt_prompt, ROUND_CRITIQUE)
                except ProviderError as exc:
                    counters["provider_errors"] += 1
                    last_error = str(exc)
                    self._absent(
                        session_id, ROUND_CRITIQUE, seat_no, actor, "provider_error",
                        last_error, absences, records,
                    )
                    raw = None
                    break
                counters["attempts"] += 1
                last_completion = completion
                try:
                    data, repair_report = self._recover(completion.text, actor)
                    raw = parse_critique(data, allowed=allowed, actor=actor)
                    break
                except SheetError as exc:
                    counters["failures"] += 1
                    last_error = str(exc)
                    raw = None
                    if attempt < self.config.critique_repairs:
                        # Bill the thrown-away call before asking for another.
                        self._discarded(
                            session_id, ROUND_CRITIQUE, actor, seat_no, seat,
                            completion, "non_compliant_critique", last_error,
                        )
                        attempt_prompt = build_critique_repair_prompt(
                            prompt, last_error, MIN_ARGUMENT_CHARS
                        )

            if raw is None:
                if last_error and not any(
                    a.seat == seat_no and a.round == ROUND_CRITIQUE for a in absences
                ):
                    self._absent(
                        session_id, ROUND_CRITIQUE, seat_no, actor,
                        "non_compliant_critique", last_error, absences, records,
                        completion=last_completion, seat=seat,
                    )
                continue

            for objection in raw:
                findings.extend(
                    self._flag(
                        session_id,
                        ROUND_CRITIQUE,
                        actor,
                        shd.scan_objection(
                            objection.argument, actor=actor, target=objection.sheet
                        ),
                    )
                )
                target_seat = blinding.seat_for(seat_no, objection.sheet)
                target = records[target_seat]
                claim = target.initial.claim(objection.claim_n)  # type: ignore[union-attr]
                record = Objection(
                    critic_seat=seat_no,
                    target_seat=target_seat,
                    claim_n=objection.claim_n,
                    argument=objection.argument,
                    sheet_label=objection.sheet,
                )
                objections.append(record)
                self._emit(
                    session_id,
                    ROUND_CRITIQUE,
                    actor,
                    tr.OBJECTION_RAISED,
                    {
                        **record.to_dict(),
                        "critic_model": seat.model_id,
                        "target_model": target.model_id,
                        "claim_text": claim.text if claim else "",
                        **repair_report.as_payload(),
                    },
                    completion=completion,
                    seat=seat,
                )
                completion = None  # bill the call once, not once per objection

    # -- round 3 -----------------------------------------------------------

    def _round3(
        self,
        session_id: str,
        task: str,
        records: dict[int, StudentRecord],
        present: list[int],
        blinding: BlindingRound,
        objections: list[Objection],
        absences: list[Absence],
        counters: dict[str, int],
        findings: list[shd.Finding],
    ) -> None:
        self._emit(
            session_id,
            ROUND_REVISION,
            tr.SYSTEM,
            tr.SHEETS_BLINDED,
            {
                "salt": blinding.salt,
                "mapping": blinding.to_dict()["by_recipient"],
                "note": "critic label -> seat, per recipient; relabelled so a "
                "student cannot align its critics with the sheets it read",
            },
        )

        for seat_no in present:
            seat = self.council.student(seat_no)
            actor = tr.student_actor(seat_no)
            record = records[seat_no]
            initial = record.initial
            assert initial is not None

            critic_label = invert(blinding.by_recipient[seat_no])
            against = sorted(
                (o for o in objections if o.target_seat == seat_no),
                key=lambda o: (critic_label[o.critic_seat], o.claim_n),
            )
            rendered = [
                (critic_label[o.critic_seat], o.claim_n, o.argument) for o in against
            ]
            allowed: dict[str, tuple[int, ...]] = {}
            for label, claim_n, _ in rendered:
                allowed[label] = allowed.get(label, ()) + (claim_n,)

            prompt = build_revision_prompt(
                task,
                json.dumps(initial.to_dict(), indent=2, ensure_ascii=False),
                rendered,
                nonce=self._nonce(session_id, seat_no, ROUND_REVISION),
                policy=self.config.shield_policy,
            )
            try:
                completion = self._call(seat, prompt, ROUND_REVISION)
            except ProviderError as exc:
                counters["provider_errors"] += 1
                self._absent(
                    session_id, ROUND_REVISION, seat_no, actor, "provider_error",
                    str(exc), absences, records,
                )
                continue
            counters["attempts"] += 1
            revision = None
            last_error = ""
            last_completion = completion
            repair_report = CLEAN_REPAIR
            for attempt in range(self.config.revision_repairs + 1):
                if attempt:
                    try:
                        completion = self._call(seat, attempt_prompt, ROUND_REVISION)
                    except ProviderError as exc:
                        counters["provider_errors"] += 1
                        last_error = str(exc)
                        break
                    counters["attempts"] += 1
                    last_completion = completion
                try:
                    data, repair_report = self._recover(completion.text, actor)
                    revision = parse_revision(data, allowed=allowed, actor=actor)
                    break
                except SheetError as exc:
                    counters["failures"] += 1
                    last_error = str(exc)
                    revision = None
                    if attempt < self.config.revision_repairs:
                        self._discarded(
                            session_id, ROUND_REVISION, actor, seat_no, seat,
                            completion, "malformed_revision", last_error,
                        )
                        attempt_prompt = build_revision_repair_prompt(
                            prompt, last_error, MAX_CLAIMS
                        )

            if revision is None:
                self._absent(
                    session_id, ROUND_REVISION, seat_no, actor, "malformed_revision",
                    last_error, absences, records,
                    completion=last_completion, seat=seat,
                )
                continue

            # A clean opening sheet and a hostile revision is the obvious
            # play: round 3's output is what the arbiter reads as the final
            # position, so the revision is scanned in its own right rather
            # than inheriting round 1's verdict.
            findings.extend(
                self._flag(
                    session_id,
                    ROUND_REVISION,
                    actor,
                    shd.scan_sheet(revision.sheet, actor=actor),
                )
            )

            diff = diff_sheets(
                initial, revision.sheet, declared_change=revision.changed_position
            )
            record.final = revision.sheet
            record.diff = diff
            record.declared_change = revision.changed_position
            record.because = revision.because

            self._emit(
                session_id,
                ROUND_REVISION,
                actor,
                tr.SHEET_REVISED,
                {
                    "seat": seat_no,
                    "model_id": seat.model_id,
                    "sheet": revision.sheet.to_dict(),
                    "diff": diff.to_dict(),
                    "declared_change": revision.changed_position,
                    "because": [
                        {"critic_label": ref.critic, "claim_n": ref.claim_n,
                         "critic_seat": blinding.seat_for(seat_no, ref.critic)}
                        for ref in revision.because
                    ],
                    **repair_report.as_payload(),
                },
                completion=completion,
                seat=seat,
            )

            if diff.position_changed:
                self._emit(
                    session_id,
                    ROUND_REVISION,
                    actor,
                    tr.POSITION_CHANGED,
                    {
                        "seat": seat_no,
                        "model_id": seat.model_id,
                        "from": initial.position,
                        "to": revision.sheet.position,
                        "declared": revision.changed_position,
                        "declaration_matches_diff": diff.declaration_matches_diff,
                        "because": [
                            {"critic_label": ref.critic, "claim_n": ref.claim_n,
                             "critic_seat": blinding.seat_for(seat_no, ref.critic)}
                            for ref in revision.because
                        ],
                    },
                )

    # -- round 4 -----------------------------------------------------------

    def _round4(
        self,
        session_id: str,
        task: str,
        records: dict[int, StudentRecord],
        present: list[int],
        objections: list[Objection],
        absences: list[Absence],
        counters: dict[str, int],
        findings: list[shd.Finding],
    ) -> tuple[Verdict | None, str | None]:
        seat = self.council.arbiter
        sources = tuple(records[s].label for s in present)
        transcript = build_transcript(
            records, present, objections, policy=self.config.shield_policy
        )
        # The arbiter is the seat worth attacking: it writes the only output
        # anybody reads, and it reads everything. It gets its own nonce, which
        # no student has ever been shown.
        prompt = build_verdict_prompt(
            task,
            transcript,
            nonce=self._nonce(session_id, tr.ARBITER, ROUND_GRADING),
            policy=self.config.shield_policy,
        )

        attempt_prompt = prompt
        last_error = ""
        last_completion: Completion | None = None
        repair_report = CLEAN_REPAIR
        for attempt in range(self.config.verdict_repairs + 1):
            try:
                completion = self._call(seat, attempt_prompt, ROUND_GRADING)
            except ProviderError as exc:
                counters["provider_errors"] += 1
                last_error = str(exc)
                break
            counters["attempts"] += 1
            last_completion = completion
            try:
                data, repair_report = self._recover(completion.text, tr.ARBITER)
                verdict = parse_verdict(
                    data, allowed_sources=sources, actor=tr.ARBITER
                )
            except SheetError as exc:
                counters["failures"] += 1
                last_error = str(exc)
                if attempt < self.config.verdict_repairs:
                    self._discarded(
                        session_id, ROUND_GRADING, tr.ARBITER, None, seat,
                        completion, "malformed_verdict", last_error,
                    )
                    attempt_prompt = build_verdict_repair_prompt(prompt, last_error, sources)
                continue

            # The verdict is scanned too, and it is the one scan that is not
            # about the arbiter's honesty. A student that got an instruction
            # past the fence shows up here: an exfiltration URL or a role
            # token in `final_answer` is text a human is about to be handed in
            # a report, and it did not originate with the arbiter.
            verdict_text = "\n".join(
                [verdict.final_answer, verdict.confidence_note]
                + [item.substance for item in verdict.minority_report]
            )
            findings.extend(
                self._flag(
                    session_id,
                    ROUND_GRADING,
                    tr.ARBITER,
                    shd.scan(verdict_text, where="verdict", actor=tr.ARBITER),
                )
            )

            self._emit(
                session_id,
                ROUND_GRADING,
                tr.ARBITER,
                tr.VERDICT_DELIVERED,
                {
                    "model_id": seat.model_id,
                    "provider": seat.provider,
                    "verdict": verdict.to_dict(),
                    "council_size": len(present),
                    "reduced_council": len(present) < len(self.council.students),
                    **repair_report.as_payload(),
                },
                completion=completion,
                seat=seat,
            )
            for item in verdict.minority_report:
                source_seat = next(
                    (s for s in present if records[s].label == item.source), None
                )
                self._emit(
                    session_id,
                    ROUND_GRADING,
                    tr.ARBITER,
                    tr.MINORITY_RECORDED,
                    {
                        "source": item.source,
                        "source_seat": source_seat,
                        "source_model": records[source_seat].model_id if source_seat else "",
                        "kind": item.kind,
                        "substance": item.substance,
                    },
                )
            return verdict, None

        reason = f"arbiter produced no valid verdict: {last_error}"
        absences.append(
            Absence(seat=None, actor=tr.ARBITER, round=ROUND_GRADING,
                    reason="arbiter_failed", detail=last_error)
        )
        self._emit(
            session_id,
            ROUND_GRADING,
            tr.ARBITER,
            tr.ARBITER_ABSENT,
            {
                "model_id": seat.model_id,
                "reason": "arbiter_failed",
                "detail": last_error,
                "raw": last_completion.text if last_completion else "",
            },
            completion=last_completion,
            seat=seat,
        )
        return None, reason

    # -- shared ------------------------------------------------------------

    def _discarded(
        self,
        session_id: str,
        round: int,
        actor: str,
        seat_no: int | None,
        seat: Seat,
        completion: Completion,
        reason: str,
        detail: str,
    ) -> None:
        """Record a model call whose output was thrown away.

        Without this, a re-prompted critique is a real call that costs real
        money and leaves no trace, so a session with repairs reports a *lower*
        cost than a clean one — a cost guardrail that errs downward on exactly
        the paths that burn extra money is worse than no guardrail.
        """
        self._emit(
            session_id,
            round,
            actor,
            tr.ATTEMPT_DISCARDED,
            {
                "seat": seat_no,
                "model_id": seat.model_id,
                "reason": reason,
                "detail": detail,
                "raw": completion.text,
            },
            completion=completion,
            seat=seat,
        )

    def _absent(
        self,
        session_id: str,
        round: int,
        seat_no: int,
        actor: str,
        reason: str,
        detail: str,
        absences: list[Absence],
        records: dict[int, StudentRecord],
        completion: Completion | None = None,
        seat: Seat | None = None,
    ) -> None:
        absences.append(
            Absence(seat=seat_no, actor=actor, round=round, reason=reason, detail=detail)
        )
        record = records[seat_no]
        record.absent_rounds = record.absent_rounds + (round,)
        self._emit(
            session_id,
            round,
            actor,
            tr.STUDENT_ABSENT,
            {
                "seat": seat_no,
                "model_id": record.model_id,
                "reason": reason,
                "detail": detail,
                # The raw text is kept because "the model returned six claims"
                # and "the model returned an apology" are different bugs, and
                # the trace is the only place that distinction survives.
                "raw": completion.text if completion else "",
            },
            completion=completion,
            seat=seat,
        )

    def _close(
        self,
        session_id: str,
        task: str,
        records: dict[int, StudentRecord],
        objections: list[Objection],
        verdict: Verdict | None,
        blinding: dict[int, BlindingRound],
        absences: list[Absence],
        first_event: int,
        counters: dict[str, int],
        findings: list[shd.Finding],
        *,
        failed_reason: str | None,
    ) -> SessionResult:
        events = self.writer.events[first_event:]
        tokens_in = sum(e.tokens_in for e in events)
        tokens_out = sum(e.tokens_out for e in events)
        cost = sum(e.cost_est for e in events)
        cache = tr.CacheSummary.from_events(events)
        present = [s for s in self.council.student_seats() if records[s].present]
        baseline = single_model_baseline(
            self.council, events, override=self.config.baseline_model,
            seat=self.config.baseline_seat,
        )
        spread = disagreement([records[s].initial for s in present])

        self._emit(
            session_id,
            ROUND_GRADING if verdict else 0,
            tr.SYSTEM,
            tr.SESSION_CLOSED,
            {
                "council_size": len(present),
                "seated": len(self.council.students),
                "reduced_council": len(present) < len(self.council.students),
                "objections": len(objections),
                "absences": [a.to_dict() for a in absences],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_est": cost,
                "cache": cache.to_dict(),
                "malformation": {
                    model: rate.to_dict()
                    for model, rate in tr.malformation_by_model(events).items()
                },
                **baseline.to_dict(),
                "disagreement": spread.to_dict(),
                "single_lab": self.council.single_lab,
                "council_warnings": list(self.council.warnings),
                "skeptic_seat": self.config.skeptic_seat,
                "cost_multiple": cost_multiple(cost, baseline),
                "cost_is_complete": all(
                    seat.cost.priced for seat in self.council.seats()
                ),
                "unpriced_seats": [
                    seat.model_id
                    for seat in self.council.seats()
                    if not seat.cost.priced
                ],
                "parse_attempts": counters["attempts"],
                "parse_failures": counters["failures"],
                "provider_errors": counters["provider_errors"],
                "shield_findings": len(findings),
                "shield_worst": shd.worst(findings) if findings else "",
                "failed_reason": failed_reason,
            },
        )

        return SessionResult(
            session_id=session_id,
            task=task,
            council=self.council,
            students=tuple(records[s] for s in self.council.student_seats()),
            objections=tuple(objections),
            verdict=verdict,
            blinding=blinding,
            absences=tuple(absences),
            events=self.writer.events[first_event:],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_est=cost,
            cache=cache,
            failed_reason=failed_reason,
            parse_attempts=counters["attempts"],
            parse_failures=counters["failures"],
            provider_errors=counters["provider_errors"],
            baseline=baseline,
            disagreement=spread,
            findings=tuple(findings),
        )


# --------------------------------------------------------------------------
# the arbiter's briefing
# --------------------------------------------------------------------------


def build_transcript(
    records: dict[int, StudentRecord],
    present: list[int],
    objections: list[Objection],
    *,
    policy: shd.ShieldPolicy = shd.DEFAULT_POLICY,
) -> str:
    """Render the whole debate for the arbiter, under seat labels only.

    Ordered by participant rather than by round on purpose: the arbiter's job
    is to weigh positions and notice what got dropped, and a chronological
    transcript makes "what did Student 2 end up believing, and why" a
    reconstruction task. Both the opening and the final sheet are shown,
    because the *movement* is the evidence — a claim withdrawn under objection
    means something a final sheet alone cannot say.
    """
    blocks: list[str] = []
    for seat_no in present:
        record = records[seat_no]
        assert record.initial is not None
        lines = [f"--- {record.label} ---", "OPENING SHEET"]
        lines.append(
            render_sheet(
                record.initial, str(seat_no), include_nuance=True, policy=policy
            ).split("\n", 1)[1]
        )

        raised = [o for o in objections if o.critic_seat == seat_no]
        lines.append("OBJECTIONS THIS PARTICIPANT RAISED")
        if raised:
            lines += [
                f"  - against {records[o.target_seat].label} claim {o.claim_n}: "
                + shd.armor(o.argument, where="objection", policy=policy)[0]
                for o in sorted(raised, key=lambda o: (o.target_seat, o.claim_n))
            ]
        else:
            lines.append("  - (none recorded)")

        received = [o for o in objections if o.target_seat == seat_no]
        lines.append("OBJECTIONS RAISED AGAINST IT")
        if received:
            lines += [
                f"  - on claim {o.claim_n}: "
                + shd.armor(o.argument, where="objection", policy=policy)[0]
                for o in sorted(received, key=lambda o: o.claim_n)
            ]
        else:
            lines.append("  - (none recorded)")

        lines.append("FINAL SHEET")
        final = record.final or record.initial
        lines.append(
            render_sheet(
                final, str(seat_no), include_nuance=True, policy=policy
            ).split("\n", 1)[1]
        )
        if record.diff is not None:
            moved = "yes" if record.diff.position_changed else "no"
            lines.append(f"POSITION CHANGED: {moved}")
            if record.diff.claims_dropped:
                lines.append(
                    "CLAIMS WITHDRAWN: "
                    + "; ".join(
                        shd.armor(c.text, where="withdrawn claim", policy=policy)[0]
                        for c in record.diff.claims_dropped
                    )
                )
        else:
            lines.append("POSITION CHANGED: no revision submitted")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def convene(
    task: str,
    council: Council,
    providers: ProviderPool,
    *,
    trace_path: str | None = None,
    config: SessionConfig | None = None,
    session_id: str | None = None,
) -> SessionResult:
    """One-call convenience wrapper: build a session, run it, return the result."""
    return Session(
        council, providers, trace_path=trace_path, config=config
    ).run(task, session_id=session_id)
