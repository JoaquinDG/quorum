"""Resuming an interrupted debate without paying for it twice.

The property under test is narrow and expensive to get wrong: rounds already
recorded must not be re-run. Everything else here — the lock, the idempotent
no-op, the annotation — exists to protect that one property from the ways a
retry loop or a second worker would otherwise defeat it.
"""

import json
import os
import shutil
import socket
import tempfile
import unittest

from quorum import ProviderPool, Session, SessionConfig
from quorum import trace as tr
from quorum.providers.recorded import council_from_trace, replay_pool
from quorum.resume import (
    Checkpoint, ResumeError, SessionLocked, TraceLock, inspect, rebuild,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

_real_socket = socket.socket


class _NoNetwork(socket.socket):
    def __init__(self, *a, **k):  # pragma: no cover
        raise AssertionError("resume tests must not touch the network")


def setUpModule():
    socket.socket = _NoNetwork


def tearDownModule():
    socket.socket = _real_socket


def load(name):
    return tr.read_trace(os.path.join(FIXTURES, name))


class NeverCalled:
    """Fails the test if the engine asks it for anything."""

    name = "labs"

    def complete(self, *args, **kwargs):
        raise AssertionError("a model was called when none should have been")


class InspectingATrace(unittest.TestCase):
    def test_a_complete_session_is_not_resumable(self):
        checkpoint = inspect(load("clean_session.jsonl"))
        self.assertTrue(checkpoint.complete)
        self.assertFalse(checkpoint.resumable)

    def test_an_interrupted_session_reports_its_last_finished_round(self):
        checkpoint = inspect(load("interrupted_session.jsonl"))
        self.assertFalse(checkpoint.complete)
        self.assertEqual(checkpoint.last_round, 1)
        self.assertTrue(checkpoint.resumable)

    def test_the_task_survives_the_interruption(self):
        self.assertIn("ingestion pipeline",
                      inspect(load("interrupted_session.jsonl")).task)

    def test_an_empty_trace_is_an_error_not_a_fresh_session(self):
        with self.assertRaises(ResumeError):
            inspect([])

    def test_a_trace_of_two_sessions_is_refused(self):
        mixed = load("clean_session.jsonl") + load("malformed_seat.jsonl")
        with self.assertRaises(ResumeError):
            inspect(mixed)


class RebuildingState(unittest.TestCase):
    def test_round_one_sheets_come_back(self):
        events = load("interrupted_session.jsonl")
        state = rebuild(events, council_from_trace(load("clean_session.jsonl")))
        present = [r for r in state["records"].values() if r.present]
        self.assertEqual(len(present), 3)

    def test_objections_come_back_in_canonical_form(self):
        events = load("clean_session.jsonl")
        state = rebuild(events, council_from_trace(events))
        self.assertTrue(state["objections"])
        first = state["objections"][0]
        self.assertIsInstance(first.critic_seat, int)
        self.assertIsInstance(first.target_seat, int)


class ResumingRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "session.jsonl")
        shutil.copy(os.path.join(FIXTURES, "interrupted_session.jsonl"), self.path)
        self.full = load("clean_session.jsonl")
        self.council = council_from_trace(self.full)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def resume(self):
        writer = tr.TraceWriter(path=self.path)
        session = Session(self.council, replay_pool(self.full),
                          config=SessionConfig(), writer=writer)
        return session.resume(tr.read_trace(self.path))

    def test_resuming_completes_the_debate(self):
        result = self.resume()
        self.assertIsNotNone(result.verdict)
        self.assertEqual(sum(1 for s in result.students if s.present), 3)

    def test_round_one_is_billed_exactly_once(self):
        """The whole point. Re-running round 1 is the waste being eliminated."""
        self.resume()
        final = tr.read_trace(self.path)
        sheets = [e for e in final if e.event_type == tr.SHEET_SUBMITTED
                  and e.round == 1]
        self.assertEqual(len(sheets), 3)
        seats = sorted(e.payload["seat"] for e in sheets)
        self.assertEqual(seats, [1, 2, 3])

    def test_the_resume_is_annotated_in_the_trace(self):
        self.resume()
        marks = [e for e in tr.read_trace(self.path)
                 if e.event_type == tr.SESSION_RESUMED]
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0].payload["resumed_after_round"], 1)

    def test_the_final_trace_replays_through_the_canonical_pipeline(self):
        from quorum.replay import replay as build_view
        from quorum.report import render_html, render_markdown

        self.resume()
        view = build_view(tr.read_trace(self.path))
        self.assertTrue(render_markdown(view))
        self.assertTrue(render_html(view))

    def test_the_session_id_survives_so_prompts_stay_identical(self):
        result = self.resume()
        self.assertEqual(result.session_id, "fixture-interrupted")


class Idempotency(unittest.TestCase):
    def test_resuming_a_finished_session_calls_no_models(self):
        session = Session(
            council_from_trace(load("clean_session.jsonl")),
            ProviderPool([NeverCalled()]),
            config=SessionConfig(), writer=tr.TraceWriter(path=None),
        )
        result = session.resume(load("clean_session.jsonl"))
        self.assertIsNotNone(result.verdict)

    def test_resuming_a_finished_session_twice_is_stable(self):
        events = load("clean_session.jsonl")
        council = council_from_trace(events)
        first = Session(council, ProviderPool([NeverCalled()]),
                        config=SessionConfig(),
                        writer=tr.TraceWriter(path=None)).resume(events)
        second = Session(council, ProviderPool([NeverCalled()]),
                         config=SessionConfig(),
                         writer=tr.TraceWriter(path=None)).resume(events)
        self.assertEqual(first.session_id, second.session_id)

    def test_a_session_with_no_completed_round_cannot_resume(self):
        events = [e for e in load("interrupted_session.jsonl")
                  if e.event_type != tr.ROUND_COMPLETED]
        session = Session(council_from_trace(load("clean_session.jsonl")),
                          ProviderPool([NeverCalled()]),
                          config=SessionConfig(), writer=tr.TraceWriter(path=None))
        with self.assertRaises(ResumeError):
            session.resume(events)


class Locking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "s.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_a_second_holder_is_refused_rather_than_queued(self):
        """Two resumes of one session would each re-run the remaining rounds."""
        with TraceLock(self.path):
            with self.assertRaises(SessionLocked):
                TraceLock(self.path).acquire()

    def test_the_lock_is_released_on_exit(self):
        with TraceLock(self.path):
            pass
        TraceLock(self.path).acquire()  # must not raise
        TraceLock(self.path).release()

    def test_the_lock_is_released_even_when_the_body_raises(self):
        with self.assertRaises(ValueError):
            with TraceLock(self.path):
                raise ValueError("boom")
        TraceLock(self.path).acquire()
        TraceLock(self.path).release()

    def test_a_concurrent_resume_cannot_double_run(self):
        shutil.copy(os.path.join(FIXTURES, "interrupted_session.jsonl"), self.path)
        full = load("clean_session.jsonl")
        council = council_from_trace(full)
        held = TraceLock(self.path)
        held.acquire()
        try:
            session = Session(council, replay_pool(full), config=SessionConfig(),
                              writer=tr.TraceWriter(path=self.path))
            with self.assertRaises(SessionLocked):
                session.resume(tr.read_trace(self.path))
        finally:
            held.release()


if __name__ == "__main__":
    unittest.main()
