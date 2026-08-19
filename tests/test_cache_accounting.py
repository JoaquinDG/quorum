"""Prompt-cache accounting: parsing it, pricing it, and carrying it in the trace.

The measurement exists because Quorum's prompts do not currently cache. Each
round builds a fresh prompt with its own opening header, so round N shares no
prefix with round N-1, and the four seats run on four different models whose
caches are separate. These tests therefore assert two different things, and
the second matters more than the first:

  - that a cache hit, when one happens, is parsed and priced correctly;
  - that when nothing caches, the report says so, rather than reporting a
    saving it cannot substantiate.

The per-provider parsing tests are the load-bearing ones. The three vendors
disagree about what their own prompt-token count means — Anthropic excludes
cached tokens from `input_tokens`, OpenAI and DeepSeek include them — and the
fields look interchangeable enough to be read wrong without anything failing.
"""

import json
import unittest

from quorum import CacheSummary, ModelCost
from quorum import trace as tr
from quorum.providers.http import AnthropicProvider, OpenAICompatibleProvider


class StubTransport:
    """Feeds one canned response body through an adapter's parsing path."""

    def __init__(self, body):
        self.body = body

    def __call__(self, url, body, headers, model_id):
        return self.body


def anthropic_with(usage):
    p = AnthropicProvider(api_key="k")
    p._request = StubTransport(
        {"content": [{"type": "text", "text": "hi"}], "usage": usage}
    )
    return p


def openai_with(usage, *, name="openai"):
    p = OpenAICompatibleProvider(api_key="k", name=name)
    p._request = StubTransport(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
         "usage": usage}
    )
    return p


class AnthropicCacheParsing(unittest.TestCase):
    def test_reads_both_cache_counters(self):
        c = anthropic_with({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 200,
        }).complete("claude-sonnet-5", "prompt")
        self.assertEqual(c.cache_read_tokens, 900)
        self.assertEqual(c.cache_write_tokens, 200)

    def test_input_tokens_left_alone(self):
        """Anthropic already excludes cached tokens; subtracting would double-count."""
        c = anthropic_with({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 900,
        }).complete("claude-sonnet-5", "prompt")
        self.assertEqual(c.input_tokens, 100)
        self.assertEqual(c.gross_input_tokens, 1000)

    def test_absent_cache_fields_are_zero_not_missing(self):
        c = anthropic_with({"input_tokens": 100, "output_tokens": 50}).complete(
            "claude-sonnet-5", "prompt"
        )
        self.assertEqual((c.cache_read_tokens, c.cache_write_tokens), (0, 0))
        self.assertEqual(c.gross_input_tokens, 100)


class OpenAICacheParsing(unittest.TestCase):
    def test_cached_tokens_subtracted_from_prompt_tokens(self):
        """`prompt_tokens` includes the cached part; `input_tokens` must not."""
        c = openai_with({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900},
        }).complete("gpt-5.1", "prompt")
        self.assertEqual(c.input_tokens, 100)
        self.assertEqual(c.cache_read_tokens, 900)
        # The total the vendor reported has to survive the conversion.
        self.assertEqual(c.gross_input_tokens, 1000)

    def test_no_write_premium_is_reported(self):
        """OpenAI caches as a side effect and bills no write. Zero is the fact."""
        c = openai_with({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900},
        }).complete("gpt-5.1", "prompt")
        self.assertEqual(c.cache_write_tokens, 0)

    def test_incoherent_usage_cannot_credit_the_session(self):
        """A vendor whose numbers disagree must not produce negative input."""
        c = openai_with({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 900},
        }).complete("gpt-5.1", "prompt")
        self.assertEqual(c.input_tokens, 0)


class DeepSeekCacheParsing(unittest.TestCase):
    """DeepSeek uses the OpenAI body with its own flat cache fields."""

    def test_hit_tokens_read_from_flat_field(self):
        c = openai_with({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 900,
            "prompt_cache_miss_tokens": 100,
        }, name="deepseek").complete("deepseek-v4-pro", "prompt")
        self.assertEqual(c.cache_read_tokens, 900)
        self.assertEqual(c.input_tokens, 100)
        self.assertEqual(c.gross_input_tokens, 1000)

    def test_miss_only_reports_no_caching(self):
        c = openai_with({
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 1000,
        }, name="deepseek").complete("deepseek-v4-pro", "prompt")
        self.assertEqual(c.cache_read_tokens, 0)
        self.assertEqual(c.input_tokens, 1000)


class CachePricing(unittest.TestCase):
    def test_reads_and_writes_use_their_own_rates(self):
        cost = ModelCost(
            input_per_mtok=3.0, output_per_mtok=15.0,
            cache_read_per_mtok=0.3, cache_write_per_mtok=3.75,
        )
        # 100 fresh + 900 read + 200 written, 50 out
        got = cost.estimate(100, 50, 900, 200)
        want = (100 * 3.0 + 900 * 0.3 + 200 * 3.75 + 50 * 15.0) / 1e6
        self.assertAlmostEqual(got, want)

    def test_unpriced_cache_bills_at_full_input_rate(self):
        """Erring upward: a missing cache price must not invent a discount."""
        cost = ModelCost(input_per_mtok=3.0, output_per_mtok=15.0)
        self.assertAlmostEqual(
            cost.estimate(100, 0, 900, 0), cost.estimate(1000, 0)
        )

    def test_uncached_counterfactual_prices_every_token_fresh(self):
        cost = ModelCost(
            input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.3
        )
        self.assertAlmostEqual(
            cost.uncached_estimate(100, 50, 900, 0), cost.estimate(1000, 50)
        )

    def test_two_argument_calls_are_unchanged(self):
        """The pre-existing signature keeps its old number."""
        cost = ModelCost(input_per_mtok=3.0, output_per_mtok=15.0)
        self.assertAlmostEqual(cost.estimate(1000, 500), (3000 + 7500) / 1e6)


class SummaryArithmetic(unittest.TestCase):
    def test_hit_share_counts_reads_only(self):
        """A write is the full-price token that made the entry, not a hit."""
        s = CacheSummary(tokens_in=0, cache_read=0, cache_write=1000)
        self.assertEqual(s.cache_hit_share, 0.0)
        self.assertEqual(s.gross_tokens_in, 1000)

    def test_hit_share_is_zero_on_an_empty_session(self):
        self.assertEqual(CacheSummary().cache_hit_share, 0.0)

    def test_write_only_session_reports_a_negative_saving(self):
        """The documented edge case: entries written, none read, premium paid.

        Reported rather than clamped, because a session that lost money to the
        cache is exactly what a cost report must not round up to zero.
        """
        s = CacheSummary(cache_write=1000, cost_est=0.00375, uncached_cost_est=0.003)
        self.assertLess(s.saved, 0)

    def test_summary_sums_events(self):
        events = [
            tr.TraceEvent(0.0, "s", 1, "a", tr.SHEET_SUBMITTED,
                          tokens_in=10, cache_read=90, cache_write=5,
                          cost_est=1.0, uncached_cost_est=2.0),
            tr.TraceEvent(0.0, "s", 2, "a", tr.SHEET_SUBMITTED,
                          tokens_in=20, cache_read=10, cache_write=0,
                          cost_est=0.5, uncached_cost_est=0.75),
        ]
        s = CacheSummary.from_events(events)
        self.assertEqual((s.tokens_in, s.cache_read, s.cache_write), (30, 100, 5))
        self.assertAlmostEqual(s.saved, 1.25)


class TraceSchemaStaysBackwardCompatible(unittest.TestCase):
    """The wrapper repo replays existing traces; the new fields must be optional."""

    def test_event_without_cache_fields_still_parses(self):
        old = {
            "ts": 1.0, "session_id": "q-1", "round": 1, "actor": "student:1",
            "event_type": tr.SHEET_SUBMITTED, "payload": {},
            "tokens_in": 100, "tokens_out": 50, "cost_est": 0.01,
        }
        e = tr.TraceEvent.from_dict(old)
        self.assertEqual((e.cache_read, e.cache_write), (0, 0))

    def test_pre_cache_event_reports_no_saving_not_a_free_session(self):
        """`uncached_cost_est` defaults to the effective cost, so saved == 0."""
        old = {
            "ts": 1.0, "session_id": "q-1", "round": 1, "actor": "student:1",
            "event_type": tr.SHEET_SUBMITTED, "payload": {},
            "tokens_in": 100, "tokens_out": 50, "cost_est": 0.01,
        }
        s = CacheSummary.from_events([tr.TraceEvent.from_dict(old)])
        self.assertEqual(s.saved, 0.0)
        self.assertEqual(s.cache_hit_share, 0.0)

    def test_round_trips_through_json(self):
        e = tr.TraceEvent(1.0, "q-1", 2, "student:1", tr.SHEET_SUBMITTED,
                          tokens_in=1, tokens_out=2, cost_est=0.5,
                          cache_read=7, cache_write=3, uncached_cost_est=0.9)
        back = tr.TraceEvent.from_dict(json.loads(json.dumps(e.to_dict())))
        self.assertEqual(back, e)


if __name__ == "__main__":
    unittest.main()


class PromptAssemblyIsDeterministic(unittest.TestCase):
    """Same session state in, identical bytes out.

    A caching prerequisite, and worth holding independently of caching: replay
    rebuilds prompts from the trace, so an assembly step that varied run to run
    would make the record unverifiable. The audit that accompanied these tests
    found the path already clean — no wall-clock reads, no random identifiers,
    no set iteration, and `fence_nonce` a pure hash of (session, recipient,
    round). These tests keep it that way.
    """

    def setUp(self):
        from quorum import prompts, shield
        from quorum.sheets import parse_sheet
        self.prompts, self.shield = prompts, shield
        self.sheet = parse_sheet({
            "position": "Refactor in place.",
            "claims": [{"n": 1, "text": "The ingest path is stable enough to keep."},
                       {"n": 2, "text": "A rebuild re-earns three years of edge cases."}],
            "assumptions": ["traffic stays within an order of magnitude"],
            "would_change_my_mind": ["a throughput ceiling we cannot lift"],
            "confidence": 0.7,
            "nuance": "",
        })

    def test_sheet_prompt_is_stable(self):
        a = self.prompts.build_sheet_prompt("Rebuild or refactor?")
        b = self.prompts.build_sheet_prompt("Rebuild or refactor?")
        self.assertEqual(a, b)

    def test_critique_prompt_is_stable(self):
        blinded = {"A": self.sheet, "B": self.sheet}
        kw = dict(nonce="deadbeef")
        self.assertEqual(
            self.prompts.build_critique_prompt("Q", blinded, **kw),
            self.prompts.build_critique_prompt("Q", blinded, **kw),
        )

    def test_sheet_rendering_does_not_depend_on_dict_order(self):
        """Peer sheets are ordered by label, not by insertion."""
        forward = {"A": self.sheet, "B": self.sheet}
        reverse = {"B": self.sheet, "A": self.sheet}
        self.assertEqual(
            self.prompts.build_critique_prompt("Q", forward, nonce="n"),
            self.prompts.build_critique_prompt("Q", reverse, nonce="n"),
        )

    def test_revision_prompt_serializes_the_sheet_stably(self):
        payload = json.dumps(self.sheet.to_dict(), indent=2, ensure_ascii=False)
        objections = [("A", 1, "The stability record predates the new sources.")]
        kw = dict(nonce="cafe")
        self.assertEqual(
            self.prompts.build_revision_prompt("Q", payload, objections, **kw),
            self.prompts.build_revision_prompt("Q", payload, objections, **kw),
        )

    def test_fence_nonce_is_reproducible(self):
        """Replay rebuilds prompts, so the fence marker cannot be random."""
        args = ("q-abc123",)
        kw = dict(recipient=2, round=2)
        self.assertEqual(
            self.shield.fence_nonce(*args, **kw), self.shield.fence_nonce(*args, **kw)
        )

    def test_fence_nonce_differs_per_recipient(self):
        """The security property: no participant sees another's marker."""
        self.assertNotEqual(
            self.shield.fence_nonce("q-abc", recipient=2, round=2),
            self.shield.fence_nonce("q-abc", recipient=3, round=2),
        )


class RoundsShareNoPrefix(unittest.TestCase):
    """Documents why the cache counters above are expected to read zero.

    The prompt-caching plan this instrumentation came from assumed round N's
    prompt is round N-1's plus appended content. It is not: each round is
    assembled from its own template with its own opening header, so
    consecutive rounds share no prefix at all. That is a deliberate property
    of the protocol — round 2 is a blind critique with a fresh frame, and it
    must not carry the critic's own round-1 context — so this test asserts the
    architecture rather than a defect.

    If a future change does make the rounds share a prefix, this test fails,
    and that failure is the signal to re-run the caching analysis rather than
    to edit the assertion.
    """

    def test_consecutive_rounds_share_no_opening_bytes(self):
        from quorum import prompts
        from quorum.sheets import parse_sheet
        sheet = parse_sheet({
            "position": "p", "claims": [{"n": 1, "text": "c"}],
            "assumptions": ["a"], "would_change_my_mind": ["w"],
            "confidence": 0.5, "nuance": "",
        })
        task = "Rebuild or refactor?"
        r1 = prompts.build_sheet_prompt(task)
        r2 = prompts.build_critique_prompt(task, {"A": sheet}, nonce="n1")
        r3 = prompts.build_revision_prompt(
            task, json.dumps(sheet.to_dict()), [("A", 1, "argument")], nonce="n2"
        )
        for earlier, later in ((r1, r2), (r2, r3)):
            self.assertFalse(later.startswith(earlier))
            self.assertNotEqual(earlier[0], later[0])
