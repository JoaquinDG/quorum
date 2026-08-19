"""Serve a recorded session back to the engine, so development costs nothing.

Iterating on the engine against live models is the most expensive habit this
project has: every run of the pipeline during development pays four labs for
output nobody reads. A recorded session is a perfectly good stand-in for that
loop — the protocol only cares what text arrives, not who wrote it — so this
module turns a trace into a provider and lets the whole pipeline run end to
end with no network and no keys.

**Replay must pin the session id.** Blinding labels and shield fence nonces
are both derived from it, so a recorded critique that objects to "Sheet A" is
only coherent under the blinding that produced it:

    events = read_trace(path)
    session.run(task, session_id=recorded_session_id(events))

Run it under a fresh id and the labels still resolve, but to different seats,
and the critique the trace recorded gets attributed to sheets nobody wrote.
`session_from_trace` returns the id for exactly this reason, and the fixture
helpers below pass it for you.

What this is not: a byte-for-byte recording. The trace stores parsed content,
not the original response text, so what a seat "says" on replay is that
content re-serialized — the same JSON, not the same whitespace, key order, or
model preamble. That is the right fidelity for exercising the pipeline and the
wrong fidelity for asserting on raw model output. The one exception is
deliberate: responses that *failed* are recorded verbatim by the engine, and
those are replayed exactly as they arrived, because a malformed response is
only useful as a fixture if it is malformed in precisely the original way.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .. import trace as tr
from ..prompts import (
    CRITIQUE_PROMPT_HEADER,
    REVISION_PROMPT_HEADER,
    SHEET_PROMPT_HEADER,
    VERDICT_PROMPT_HEADER,
)
from .base import Completion, ProviderError

# Which round a prompt belongs to, read off the line the model actually reads.
# The same trick `MockProvider` uses, and for the same reason: a test-only
# marker would drift from the prompt it stands in for.
_ROUND_HEADERS = (
    (SHEET_PROMPT_HEADER, 1),
    (CRITIQUE_PROMPT_HEADER, 2),
    (REVISION_PROMPT_HEADER, 3),
    (VERDICT_PROMPT_HEADER, 4),
)


class RecordedResponseMissing(ProviderError):
    """The trace has nothing recorded for this seat in this round.

    Raised rather than improvised. A mock that invents a reply when the
    recording runs out would turn a fixture gap into a passing test, which is
    the one thing a fixture must never do.
    """


def round_of(prompt: str) -> int:
    for header, number in _ROUND_HEADERS:
        if header in prompt:
            return number
    return 0


def recorded_session_id(events: Iterable[tr.TraceEvent]) -> str:
    for event in events:
        return event.session_id
    raise ValueError("no events, so no session id to replay under")


def _revision_text(payload: dict[str, Any]) -> str:
    """Rebuild a round-3 response from what the trace kept of it."""
    sheet = dict(payload.get("sheet") or {})
    sheet["changed_position"] = bool(payload.get("declared_change", False))
    sheet["because"] = [
        {"critic": ref.get("critic_label"), "claim_n": ref.get("claim_n")}
        for ref in payload.get("because") or []
    ]
    return json.dumps(sheet, ensure_ascii=False)


def responses_from_trace(
    events: Iterable[tr.TraceEvent],
) -> dict[tuple[int, str], list[tuple[str, int, int]]]:
    """Index a trace as `(round, model_id) -> [(text, tokens_in, tokens_out)]`.

    Queued in file order, which is what makes failure fixtures work: the
    engine emits the discarded attempt before the reply that replaced it, so
    replaying the queue reproduces the failure and then the recovery in the
    order they originally happened.

    Keyed by model rather than by seat because that is all a provider is told.
    A council seating the same `model_id` twice therefore shares one queue;
    the fixtures here do not, and a lineup that does should key its own.
    """
    out: dict[tuple[int, str], list[tuple[str, int, int]]] = {}
    critiques: dict[tuple[str, str], list[dict[str, Any]]] = {}
    critique_usage: dict[tuple[str, str], tuple[int, int]] = {}

    def push(round_no: int, model_id: str, text: str, t_in: int, t_out: int) -> None:
        out.setdefault((round_no, model_id), []).append((text, t_in, t_out))

    for event in events:
        payload = event.payload
        kind = event.event_type
        model = str(payload.get("model_id") or payload.get("critic_model") or "")

        if kind in (tr.ATTEMPT_DISCARDED, tr.STUDENT_ABSENT, tr.ARBITER_ABSENT):
            # The verbatim failure. Replayed exactly, so a malformed-response
            # fixture stays malformed in the original way.
            raw = payload.get("raw")
            if raw:
                push(event.round, model, str(raw), event.tokens_in, event.tokens_out)
            continue

        if kind == tr.SHEET_SUBMITTED:
            push(1, model, json.dumps(payload.get("sheet") or {}, ensure_ascii=False),
                 event.tokens_in, event.tokens_out)
        elif kind == tr.OBJECTION_RAISED:
            # One response produced many events; regroup them.
            key = (event.actor, model)
            critiques.setdefault(key, []).append({
                "sheet": payload.get("sheet_label"),
                "claim_n": payload.get("claim_n"),
                "argument": payload.get("argument"),
            })
            if event.tokens_in or event.tokens_out:
                critique_usage[key] = (event.tokens_in, event.tokens_out)
        elif kind == tr.SHEET_REVISED:
            push(3, model, _revision_text(payload), event.tokens_in, event.tokens_out)
        elif kind == tr.VERDICT_DELIVERED:
            push(4, model, json.dumps(payload.get("verdict") or {}, ensure_ascii=False),
                 event.tokens_in, event.tokens_out)

    for (_actor, model), objections in critiques.items():
        t_in, t_out = critique_usage.get((_actor, model), (0, 0))
        push(2, model, json.dumps({"objections": objections}, ensure_ascii=False),
             t_in, t_out)
    return out


class RecordedProvider:
    """A provider whose answers came out of a trace file.

    Satisfies the same one-method protocol as every other provider, so the
    engine cannot tell the difference and nothing in the pipeline needs a test
    mode.
    """

    def __init__(
        self,
        responses: dict[tuple[int, str], list[tuple[str, int, int]]],
        name: str = "recorded",
    ) -> None:
        self.name = name
        self._queues = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[int, str]] = []

    @classmethod
    def from_events(
        cls, events: Iterable[tr.TraceEvent], name: str = "recorded"
    ) -> RecordedProvider:
        return cls(responses_from_trace(events), name=name)

    def complete(
        self, model_id: str, prompt: str, max_tokens: int = 2048
    ) -> Completion:
        round_no = round_of(prompt)
        key = (round_no, model_id)
        self.calls.append(key)
        queue = self._queues.get(key)
        if not queue:
            raise RecordedResponseMissing(
                f"nothing recorded for {model_id} in round {round_no}",
                provider=self.name,
                model_id=model_id,
            )
        # The last entry repeats, matching ScriptedProvider: a re-ask that the
        # recording did not anticipate gets the same answer again rather than
        # a spurious outage.
        text, t_in, t_out = queue.pop(0) if len(queue) > 1 else queue[0]
        return Completion(
            text=text, model_id=model_id, input_tokens=t_in, output_tokens=t_out
        )


def council_from_trace(events: Iterable[tr.TraceEvent]) -> Any:
    """Rebuild the council that produced a trace, seats and providers intact.

    Prices come back as zero. That is deliberate: a replayed session did not
    spend anything, and a fixture that reported a bill would put fictional
    money into the cost accounting the previous workstream just made
    trustworthy. Tests that need prices set them on the returned seats.
    """
    from ..council import Council, Seat

    events = list(events)
    seats: dict[int, tuple[str, str]] = {}
    arbiter: tuple[str, str] | None = None
    for event in events:
        payload = event.payload
        if event.event_type == tr.SHEET_SUBMITTED:
            seat_no = int(payload.get("seat", 0))
            if seat_no and seat_no not in seats:
                seats[seat_no] = (
                    str(payload.get("model_id", "")),
                    str(payload.get("provider", "recorded")),
                )
        elif event.event_type == tr.STUDENT_ABSENT:
            seat_no = int(payload.get("seat", 0))
            if seat_no and seat_no not in seats:
                seats[seat_no] = (str(payload.get("model_id", "")), "recorded")
        elif event.event_type in (tr.VERDICT_DELIVERED, tr.ARBITER_ABSENT):
            if arbiter is None:
                arbiter = (
                    str(payload.get("model_id", "")),
                    str(payload.get("provider", "recorded")),
                )
    if not seats or arbiter is None:
        raise ValueError("trace does not describe a full council")
    return Council(
        students=tuple(
            Seat(seats[n][0], seats[n][1]) for n in sorted(seats)
        ),
        arbiter=Seat(arbiter[0], arbiter[1]),
    )


def replay_pool(events: Iterable[tr.TraceEvent]) -> Any:
    """A `ProviderPool` whose every provider answers from the recording."""
    from .base import ProviderPool

    events = list(events)
    council = council_from_trace(events)
    index = responses_from_trace(events)
    names = sorted({seat.provider for seat in council.seats()})
    # One provider object per registered name, all sharing the same index:
    # the engine routes by provider name, and the recording is keyed by model.
    return ProviderPool([RecordedProvider(index, name=n) for n in names])


def task_of(events: Iterable[tr.TraceEvent]) -> str:
    for event in events:
        if event.event_type == tr.TASK_POSED:
            return str(event.payload.get("task", ""))
    raise ValueError("trace has no task_posed event")
