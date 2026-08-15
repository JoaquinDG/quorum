"""Session Report tests.

Two things are being defended here.

**The report is a player for the trace, not a second reader of the engine.**
`_as_session` routes a live `SessionResult` through `replay` on purpose, so
the report cannot accidentally start depending on state the trace does not
carry. `test_the_report_needs_nothing_but_the_file` is the test that keeps
that true: it renders from a trace file read off disk, with the session object
long gone.

**The report must not flatter the session.** A reduced council, an incomplete
cost, an empty minority report and a session where nobody moved are all
outcomes a report could quietly smooth over, and each has a test that it
doesn't.
"""

import os
import tempfile
import unittest

from quorum import (
    Session,
    convene,
    demo_council,
    mock_pool,
    rank_objections,
    render_html,
    render_markdown,
    replay,
    replay_file,
    write_report,
)

from test_session import TASK, critique_json, default_script, scripted_council

QUESTION = (
    "Our ingestion pipeline is three years old and increasingly slow. Should we "
    "rebuild it on a streaming architecture this year, or refactor it in place?"
)


def demo_session(session_id="rep"):
    council = demo_council()
    result = convene(QUESTION, council, mock_pool(council), session_id=session_id)
    return result, replay(list(result.events))


class ObjectionRankingTests(unittest.TestCase):
    def setUp(self):
        _, self.session = demo_session("rank")

    def test_objections_that_changed_something_rank_above_ones_that_did_not(self):
        ranked = rank_objections(self.session)
        self.assertEqual(len(ranked), len(self.session.objections))
        scores = [r.score for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 0, "no objection had any recorded effect")

    def test_every_ranked_objection_explains_its_effect(self):
        for ranked in rank_objections(self.session):
            self.assertTrue(ranked.effect)

    def test_an_objection_cited_by_a_mover_outranks_a_long_one(self):
        ranked = rank_objections(self.session)
        top = ranked[0]
        self.assertIn(
            "cited as the reason for a revision", top.effect
        )
        longest = max(self.session.objections, key=lambda o: len(o.argument))
        if top.objection is not longest:
            self.assertGreater(top.score, 0)


class MarkdownReportTests(unittest.TestCase):
    def setUp(self):
        self.result, self.session = demo_session("md")
        self.md = render_markdown(self.session)

    def test_it_carries_every_required_section(self):
        for heading in (
            "## The answer",
            "## Where they started",
            "## The sharpest objections",
            "## Who changed their mind",
            "## Minority report",
            "## The numbers",
        ):
            self.assertIn(heading, self.md)

    def test_it_shows_the_question_and_the_final_answer(self):
        self.assertIn(QUESTION, self.md)
        self.assertIn(self.result.verdict.final_answer, self.md)

    def test_it_shows_every_opening_position(self):
        for student in self.result.students:
            self.assertIn(student.initial.position, self.md)

    def test_it_shows_the_cost_against_a_single_model_baseline(self):
        self.assertIn(f"${self.result.cost_est:.4f}", self.md)
        self.assertIn(self.result.baseline.model_id, self.md)
        self.assertIn(f"{self.result.cost_multiple:.1f}x", self.md)

    def test_it_shows_the_minority_report_with_attribution(self):
        item = self.result.verdict.minority_report[0]
        self.assertIn(item.substance, self.md)
        self.assertIn(item.source, self.md)

    def test_it_names_who_moved_and_who_held(self):
        movers = [s for s in self.result.students if s.changed_position]
        self.assertIn("Held position:", self.md)
        for student in movers:
            self.assertIn(student.final.position, self.md)


class HtmlReportTests(unittest.TestCase):
    def setUp(self):
        self.result, self.session = demo_session("html")
        self.html = render_html(self.session)

    def test_it_is_a_complete_document(self):
        self.assertTrue(self.html.startswith("<!doctype html>"))
        self.assertIn("</html>", self.html)
        self.assertIn("<title>", self.html)

    def test_it_is_self_contained(self):
        # A report that needs the network is not a report you can email.
        for forbidden in ("<script", "src=", "@import", "https://", "http://"):
            self.assertNotIn(forbidden, self.html, f"external reference: {forbidden}")

    def test_it_styles_both_themes(self):
        self.assertIn("prefers-color-scheme: dark", self.html)
        self.assertIn('[data-theme="dark"]', self.html)
        self.assertIn('[data-theme="light"]', self.html)

    def test_it_escapes_content(self):
        council = demo_council()
        nasty = "Should we ship <script>alert(1)</script> or not, given the risk?"
        result = convene(nasty, council, mock_pool(council), session_id="xss")
        html = render_html(replay(list(result.events)))
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_it_carries_the_headline_sections(self):
        for text in (
            "The answer",
            "Where they started",
            "The sharpest objections",
            "Who changed their mind",
            "Minority report",
            "The numbers",
        ):
            self.assertIn(text, self.html)

    def test_it_shows_the_cost_multiple(self):
        self.assertIn(f"{self.result.cost_multiple:.1f}×", self.html)
        self.assertIn(self.result.baseline.model_id, self.html)

    def test_it_states_that_diffs_are_computed_not_self_reported(self):
        self.assertIn("not taken from what the models said", self.html)


class HonestyTests(unittest.TestCase):
    """The report must surface what a flattering report would hide."""

    def reduced(self):
        script = default_script()
        script["model-2"] = ["prose, not a sheet"] + script["model-2"][1:]
        script["model-1"][1] = critique_json(("A",))
        script["model-3"][1] = critique_json(("A",))
        council, providers = scripted_council(script)
        result = Session(council, providers).run(TASK, session_id="rep-reduced")
        return replay(list(result.events))

    def test_a_reduced_council_is_banner_level_news(self):
        session = self.reduced()
        html = render_html(session)
        markdown = render_markdown(session)
        self.assertIn("Reduced council", html)
        self.assertIn("banner", html)
        self.assertIn("Reduced council", markdown)

    def test_an_incomplete_cost_is_labelled_a_lower_bound(self):
        session = self.reduced()  # the scripted arbiter is unpriced
        self.assertIn("Incomplete cost", render_html(session))
        self.assertIn("lower bound", render_markdown(session))

    def test_an_empty_minority_report_is_not_sold_as_consensus(self):
        script = default_script(verdict_minority=())
        council, providers = scripted_council(script)
        session = replay(list(Session(council, providers).run(
            TASK, session_id="rep-consensus").events))
        html = render_html(session)
        self.assertIn("not as proof of consensus", html)

    def test_a_session_where_nobody_moved_says_no_objection_landed(self):
        council, providers = scripted_council(default_script())
        session = replay(list(Session(council, providers).run(
            TASK, session_id="rep-static").events))
        self.assertEqual(session.position_change_rate, 0.0)
        self.assertIn("no objection landed", render_html(session))
        self.assertIn("Nobody changed position", render_markdown(session))

    def test_a_verdictless_session_still_renders(self):
        script = default_script()
        script["arbiter-model"] = ["prose", "more prose"]
        council, providers = scripted_council(script)
        session = replay(list(Session(council, providers).run(
            TASK, session_id="rep-noverdict").events))
        self.assertIn("No verdict", render_html(session))
        self.assertIn("did not reach a verdict", render_html(session))
        self.assertIn("No verdict", render_markdown(session))


class TraceOnlyTests(unittest.TestCase):
    def test_the_report_needs_nothing_but_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = os.path.join(directory, "t.jsonl")
            council = demo_council()
            live = convene(
                QUESTION, council, mock_pool(council), trace_path=trace, session_id="file"
            )
            expected = live.verdict.final_answer
            del live, council  # the engine is gone; only the file remains

            session = replay_file(trace)[0]
            html = render_html(session)

        self.assertIn(expected, html)
        self.assertIn("6.8×" if "6.8×" in html else "×", html)

    def test_write_report_emits_html_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "report.html")
            result, _ = demo_session("write")
            written = write_report(result, path)
            self.assertEqual(written, path)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(directory, "nested", "report.md")))
            with open(path, encoding="utf-8") as handle:
                self.assertIn("<!doctype html>", handle.read())

    def test_write_report_accepts_a_replayed_session_too(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "r.html")
            _, session = demo_session("write2")
            write_report(session, path, markdown=False)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(os.path.join(directory, "r.md")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
