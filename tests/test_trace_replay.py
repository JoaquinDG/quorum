"""Trace and replay tests: the completeness proof.

The design rule is that any future UI is a player for the trace file. The only
honest way to test a claim like that is to run a session, throw the session
object away, rebuild it from the file, and compare — which is what
`ReplayCompletenessTests` does, field by field, including the structural diffs
and the blinding maps a renderer would need to show "Student 1 objected to
what it knew only as Sheet B".

The diff test is the sharpest one: it recomputes each diff from the replayed
sheets and asserts it matches the diff the engine recorded. If the trace ever
starts storing a summary instead of the sheets themselves, that test fails.
"""

import json
import os
import tempfile
import unittest

from quorum import (
    Session,
    TraceEvent,
    convene,
    demo_council,
    diff_sheets,
    group_sessions,
    mock_pool,
    read_trace,
    render,
    replay,
    replay_file,
)
from quorum import trace as tr

from test_session import TASK, default_script, scripted_council


class TraceWriterTests(unittest.TestCase):
    def test_unknown_event_type_is_refused(self):
        writer = tr.TraceWriter()
        with self.assertRaises(ValueError):
            writer.emit(
                session_id="s", round=1, actor="system", event_type="gossip_exchanged"
            )

    def test_events_are_written_and_read_back_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "t.jsonl")
            writer = tr.TraceWriter(path, clock=lambda: 1.0)
            writer.emit(
                session_id="s",
                round=1,
                actor="student:1",
                event_type=tr.SHEET_SUBMITTED,
                payload={"seat": 1},
                tokens_in=10,
                tokens_out=20,
                cost_est=0.5,
            )
            events = read_trace(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].to_dict(), writer.events[0].to_dict())

    def test_totals(self):
        writer = tr.TraceWriter(clock=lambda: 0.0)
        for _ in range(3):
            writer.emit(
                session_id="s",
                round=1,
                actor="system",
                event_type=tr.TASK_POSED,
                tokens_in=5,
                tokens_out=7,
                cost_est=0.25,
            )
        self.assertEqual(writer.totals(), (15, 21, 0.75))

    def test_a_malformed_line_is_an_error_not_a_shrug(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"not": "json"\n')
            with self.assertRaises(ValueError) as ctx:
                read_trace(path)
        self.assertIn("malformed", str(ctx.exception))

    def test_an_unknown_event_type_on_disk_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "ts": 1.0,
                            "session_id": "s",
                            "round": 1,
                            "actor": "system",
                            "event_type": "invented",
                            "payload": {},
                        }
                    )
                    + "\n"
                )
            with self.assertRaises(ValueError):
                read_trace(path)

    def test_missing_fields_are_an_error(self):
        with self.assertRaises(ValueError):
            TraceEvent.from_dict({"ts": 1.0, "session_id": "s"})

    def test_the_clock_is_injectable_so_ordering_is_deterministic(self):
        ticks = iter(range(100))
        writer = tr.TraceWriter(clock=lambda: float(next(ticks)))
        council = demo_council()
        Session(council, mock_pool(council), writer=writer).run(TASK, session_id="clk")
        stamps = [e.ts for e in writer.events]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(len(set(stamps)), len(stamps))


class TraceContentTests(unittest.TestCase):
    def setUp(self):
        council = demo_council()
        self.result = convene(TASK, council, mock_pool(council), session_id="tc-1")

    def test_every_spec_event_type_is_emitted_by_a_clean_session(self):
        emitted = {e.event_type for e in self.result.events}
        expected = {
            tr.TASK_POSED,
            tr.SHEET_SUBMITTED,
            tr.SHEETS_BLINDED,
            tr.OBJECTION_RAISED,
            tr.SHEET_REVISED,
            tr.POSITION_CHANGED,
            tr.VERDICT_DELIVERED,
            tr.MINORITY_RECORDED,
            tr.SESSION_CLOSED,
        }
        self.assertEqual(expected - emitted, set())

    def test_every_event_round_trips_through_json(self):
        for event in self.result.events:
            restored = TraceEvent.from_dict(json.loads(json.dumps(event.to_dict())))
            self.assertEqual(restored.to_dict(), event.to_dict())

    def test_cost_is_billed_once_per_call_not_once_per_objection(self):
        objections = [e for e in self.result.events if e.event_type == tr.OBJECTION_RAISED]
        billed = [e for e in objections if e.tokens_out > 0]
        self.assertEqual(len(billed), len(self.result.council.students))
        self.assertEqual(len(objections), 6)

    def test_the_trace_records_the_blinding_maps(self):
        blinded = [e for e in self.result.events if e.event_type == tr.SHEETS_BLINDED]
        self.assertEqual(len(blinded), 2)
        for event in blinded:
            self.assertEqual(sorted(event.payload["mapping"]), ["1", "2", "3"])

    def test_session_totals_match_the_sum_of_events(self):
        self.assertEqual(
            self.result.tokens_in, sum(e.tokens_in for e in self.result.events)
        )
        self.assertAlmostEqual(
            self.result.cost_est, sum(e.cost_est for e in self.result.events)
        )


class ReplayCompletenessTests(unittest.TestCase):
    """Given only the trace, replay must lose nothing."""

    def setUp(self):
        council = demo_council()
        self.live = convene(TASK, council, mock_pool(council), session_id="rp-1")
        self.replayed = replay(list(self.live.events))

    def test_task_and_council_survive(self):
        self.assertEqual(self.replayed.task, self.live.task)
        self.assertEqual(self.replayed.session_id, self.live.session_id)
        self.assertEqual(self.replayed.council_size, self.live.council_size)
        self.assertEqual(self.replayed.reduced_council, self.live.reduced_council)
        self.assertEqual(
            self.replayed.arbiter["model_id"], self.live.council.arbiter.model_id
        )

    def test_every_position_survives(self):
        for student in self.live.students:
            other = self.replayed.students[student.seat]
            self.assertEqual(other.model_id, student.model_id)
            self.assertEqual(other.initial.to_dict(), student.initial.to_dict())
            self.assertEqual(other.final.to_dict(), student.final.to_dict())

    def test_every_objection_survives_with_the_label_the_critic_saw(self):
        self.assertEqual(len(self.replayed.objections), len(self.live.objections))
        for replayed, live in zip(self.replayed.objections, self.live.objections):
            self.assertEqual(replayed.critic_seat, live.critic_seat)
            self.assertEqual(replayed.target_seat, live.target_seat)
            self.assertEqual(replayed.claim_n, live.claim_n)
            self.assertEqual(replayed.argument, live.argument)
            self.assertEqual(replayed.sheet_label, live.sheet_label)
            # The claim text is denormalised into the event so a renderer can
            # show what was attacked without cross-referencing sheets.
            self.assertEqual(
                replayed.claim_text,
                self.live.student(live.target_seat).initial.claim(live.claim_n).text,
            )

    def test_every_diff_survives_and_can_be_recomputed_from_the_replayed_sheets(self):
        for student in self.live.students:
            other = self.replayed.students[student.seat]
            self.assertEqual(other.diff, student.diff.to_dict())
            recomputed = diff_sheets(
                other.initial, other.final, declared_change=other.declared_change
            )
            self.assertEqual(recomputed.to_dict(), student.diff.to_dict())

    def test_position_changes_survive_with_their_citations(self):
        movers = [s for s in self.live.students if s.changed_position]
        self.assertTrue(movers)
        for student in movers:
            change = self.replayed.students[student.seat].position_change
            self.assertEqual(change["from"], student.initial.position)
            self.assertEqual(change["to"], student.final.position)
            self.assertEqual(len(change["because"]), len(student.because))

    def test_the_verdict_and_minority_report_survive(self):
        self.assertEqual(
            self.replayed.verdict.to_dict(), self.live.verdict.to_dict()
        )
        self.assertEqual(
            len(self.replayed.minority), len(self.live.verdict.minority_report)
        )
        self.assertEqual(self.replayed.minority[0]["source_model"],
                         self.live.council.student(
                             int(self.replayed.minority[0]["source_seat"])).model_id)

    def test_the_blinding_maps_survive(self):
        for round_no, blinding in self.live.blinding.items():
            self.assertEqual(
                self.replayed.blinding[round_no].by_recipient, blinding.by_recipient
            )

    def test_derived_metrics_agree(self):
        self.assertAlmostEqual(
            self.replayed.position_change_rate, self.live.position_change_rate
        )
        self.assertEqual(self.replayed.tokens_in, self.live.tokens_in)
        self.assertEqual(self.replayed.tokens_out, self.live.tokens_out)
        self.assertAlmostEqual(self.replayed.cost_est, self.live.cost_est)

    def test_replay_from_disk_matches_replay_from_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.jsonl")
            council = demo_council()
            live = convene(
                TASK, council, mock_pool(council), trace_path=path, session_id="disk"
            )
            from_disk = replay_file(path)[0]

        in_memory = replay(list(live.events))
        self.assertEqual(render(from_disk), render(in_memory))
        self.assertEqual(
            from_disk.verdict.to_dict(), in_memory.verdict.to_dict()
        )


class ReplayFailureModeTests(unittest.TestCase):
    def test_a_reduced_council_session_replays_as_reduced(self):
        script = default_script()
        script["model-2"] = ["prose"] + script["model-2"][1:]
        script["model-1"][1] = '{"objections": [{"sheet": "A", "claim_n": 1, "argument": "' \
            + "x" * 60 + '"}]}'
        script["model-3"][1] = script["model-1"][1]
        council, providers = scripted_council(script)
        live = Session(council, providers).run(TASK, session_id="rp-reduced")

        replayed = replay(list(live.events))

        self.assertTrue(replayed.reduced_council)
        self.assertEqual(replayed.council_size, 2)
        self.assertFalse(replayed.students[2].present)
        self.assertEqual(len(replayed.students[2].absences), 1)
        self.assertIn("REDUCED COUNCIL", render(replayed))

    def test_a_verdictless_session_replays_and_says_so(self):
        script = default_script()
        script["arbiter-model"] = ["prose", "more prose"]
        council, providers = scripted_council(script)
        live = Session(council, providers).run(TASK, session_id="rp-noverdict")

        replayed = replay(list(live.events))

        self.assertIsNone(replayed.verdict)
        self.assertIn("arbiter produced no valid verdict", replayed.failed_reason)
        self.assertIn("NO VERDICT", render(replayed))

    def test_replaying_an_empty_trace_is_an_error(self):
        with self.assertRaises(ValueError):
            replay([])

    def test_replaying_a_mixed_trace_is_an_error(self):
        council = demo_council()
        first = convene(TASK, council, mock_pool(council), session_id="a")
        second = convene(TASK, council, mock_pool(council), session_id="b")
        with self.assertRaises(ValueError):
            replay(list(first.events) + list(second.events))

    def test_a_multi_session_file_splits_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.jsonl")
            council = demo_council()
            pool = mock_pool(council)
            session = Session(council, pool, trace_path=path)
            session.run(TASK, session_id="one")
            session.run("A different question entirely?", session_id="two")
            events = read_trace(path)
            groups = dict(group_sessions(events))
            self.assertEqual(sorted(groups), ["one", "two"])
            self.assertEqual(len(replay_file(path)), 2)


class RenderTests(unittest.TestCase):
    def test_the_transcript_shows_the_shape_of_the_debate(self):
        council = demo_council()
        live = convene(TASK, council, mock_pool(council), session_id="render-1")
        text = render(replay(list(live.events)))

        self.assertIn("ROUND 1 — silent exam", text)
        self.assertIn("ROUND 2 — blind claim-level critique", text)
        self.assertIn("CHANGED POSITION", text)
        self.assertIn("minority report", text)
        self.assertIn("position-change rate", text)
        for student in live.students:
            self.assertIn(student.initial.position, text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
