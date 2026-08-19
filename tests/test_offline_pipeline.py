"""The whole pipeline, end to end, against recorded sessions and no network.

Development against live models is the quietest way this project wastes money:
every iteration on the engine pays four labs for output that exists only to
make a test go green. A recorded session substitutes for that perfectly well —
the protocol cares what text arrives, not who wrote it.

Sockets are disabled for this module rather than merely unused. "No network
calls" asserted by inspection is a claim that decays the first time somebody
adds a provider; asserted by making a socket raise, it stays true or the suite
goes red.
"""

import json
import os
import socket
import unittest

from quorum import Session, SessionConfig
from quorum import trace as tr
from quorum.providers.recorded import (
    RecordedProvider,
    RecordedResponseMissing,
    council_from_trace,
    recorded_session_id,
    replay_pool,
    responses_from_trace,
    round_of,
    task_of,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

_real_socket = socket.socket


class _NoNetwork(socket.socket):
    def __init__(self, *args, **kwargs):  # pragma: no cover - never completes
        raise AssertionError(
            "the offline pipeline tried to open a socket; it must run entirely "
            "from recorded fixtures"
        )


def setUpModule():
    socket.socket = _NoNetwork


def tearDownModule():
    socket.socket = _real_socket


def load(name):
    return tr.read_trace(os.path.join(FIXTURES, name))


def replay(name, **config):
    events = load(name)
    session = Session(
        council_from_trace(events), replay_pool(events),
        config=SessionConfig(**config),
    )
    return events, session.run(task_of(events),
                               session_id=recorded_session_id(events))


class TheGuardItself(unittest.TestCase):
    def test_sockets_really_are_blocked(self):
        """If this passes trivially the rest of the module proves nothing."""
        with self.assertRaises(AssertionError):
            socket.socket()


class CleanSessionReplays(unittest.TestCase):
    def test_full_pipeline_runs_from_a_recording(self):
        _, result = replay("clean_session.jsonl")
        self.assertIsNotNone(result.verdict)
        self.assertEqual(sum(1 for s in result.students if s.present), 3)
        self.assertEqual(result.absences, ())

    def test_replayed_session_produces_a_replayable_trace(self):
        """The output of a replay is itself a canonical trace."""
        from quorum.replay import replay as build_view
        from quorum.report import render_html, render_markdown

        _, result = replay("clean_session.jsonl")
        view = build_view(list(result.events))
        self.assertTrue(render_markdown(view))
        self.assertTrue(render_html(view))

    def test_no_response_is_invented_when_the_recording_runs_out(self):
        provider = RecordedProvider({})
        with self.assertRaises(RecordedResponseMissing):
            provider.complete("model-alpha", "You are sitting a silent exam.")


class RoundDetection(unittest.TestCase):
    def test_each_round_is_identified_by_the_line_the_model_reads(self):
        from quorum.prompts import (
            CRITIQUE_PROMPT_HEADER, REVISION_PROMPT_HEADER,
            SHEET_PROMPT_HEADER, VERDICT_PROMPT_HEADER,
        )
        for header, expected in ((SHEET_PROMPT_HEADER, 1),
                                 (CRITIQUE_PROMPT_HEADER, 2),
                                 (REVISION_PROMPT_HEADER, 3),
                                 (VERDICT_PROMPT_HEADER, 4)):
            self.assertEqual(round_of(f"{header}\n\nbody"), expected)

    def test_an_unrecognised_prompt_is_round_zero(self):
        self.assertEqual(round_of("something else entirely"), 0)


class MalformedSeatIsRecovered(unittest.TestCase):
    """Acceptance: the 2026-08-19 incident, recorded and then survived.

    The fixture was recorded with repair disabled, so it holds the failure as
    the pre-repair engine saw it — the raw truncated critique, verbatim, on a
    discard event. Replaying it with repair enabled is therefore a real
    before/after on the same bytes.
    """

    def test_the_recording_shows_the_seat_being_dropped(self):
        events = load("malformed_seat.jsonl")
        absent = [e for e in events if e.event_type == tr.STUDENT_ABSENT]
        self.assertTrue(absent, "fixture should record the seat going absent")
        self.assertEqual(absent[0].round, 2)

    def test_replaying_it_recovers_the_seat(self):
        _, result = replay("malformed_seat.jsonl")
        self.assertEqual([a for a in result.absences if a.seat == 3], [])
        self.assertTrue([o for o in result.objections if o.critic_seat == 3])

    def test_the_recovery_is_disclosed_in_the_trace(self):
        _, result = replay("malformed_seat.jsonl")
        repaired = [e for e in result.events
                    if e.payload.get("repair", {}).get("repaired")]
        self.assertTrue(repaired, "a repaired response must say so in the trace")
        self.assertTrue(repaired[0].payload["repair"]["truncated"])

    def test_the_recovered_argument_is_verbatim(self):
        """Content survives the repair unchanged, or it is not a repair."""
        events = load("malformed_seat.jsonl")
        raw = next(e.payload["raw"] for e in events
                   if e.event_type == tr.ATTEMPT_DISCARDED)
        _, result = replay("malformed_seat.jsonl")
        for objection in [o for o in result.objections if o.critic_seat == 3]:
            self.assertIn(objection.argument, raw)

    def test_with_repair_off_the_seat_is_dropped_again(self):
        """The control. Repair is the only difference between the two runs."""
        _, result = replay("malformed_seat.jsonl", repair_json=False)
        self.assertTrue([a for a in result.absences if a.seat == 3])


class RecordingIndex(unittest.TestCase):
    def test_objections_regroup_into_one_response_per_critic(self):
        events = load("clean_session.jsonl")
        index = responses_from_trace(events)
        for (round_no, _model), queue in index.items():
            if round_no == 2:
                payload = json.loads(queue[0][0])
                self.assertIn("objections", payload)
                self.assertGreater(len(payload["objections"]), 0)

    def test_failed_attempts_are_queued_before_their_replacements(self):
        index = responses_from_trace(load("malformed_seat.jsonl"))
        queue = index[(2, "model-gamma")]
        self.assertGreaterEqual(len(queue), 1)
        # The first thing served is the response that failed.
        with self.assertRaises(ValueError):
            json.loads(queue[0][0])

    def test_council_is_rebuilt_from_the_recording(self):
        council = council_from_trace(load("clean_session.jsonl"))
        self.assertEqual([s.model_id for s in council.students],
                         ["model-alpha", "model-beta", "model-gamma"])
        self.assertEqual(council.arbiter.model_id, "model-arbiter")

    def test_a_replayed_council_is_unpriced(self):
        """A replay spent nothing, so it must not report a bill."""
        council = council_from_trace(load("clean_session.jsonl"))
        self.assertFalse(any(seat.cost.priced for seat in council.seats()))


class FixturesAreClean(unittest.TestCase):
    def test_fixtures_carry_no_secrets(self):
        needles = ("sk-", "api_key", "apikey", "authorization", "bearer",
                   "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        for name in os.listdir(FIXTURES):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
                body = handle.read().lower()
            for needle in needles:
                self.assertNotIn(needle.lower(), body, f"{name} may contain a secret")


if __name__ == "__main__":
    unittest.main()
