"""HTTP adapter and single-lab tests. Never touches the network.

Two properties are load-bearing and both are about *not* leaking or lying:

- Keys come from the environment and nowhere else, and a missing key raises
  `ProviderConfigError` — which the session treats as a bug to stop on, not an
  outage to route around. A missing key marked as an absent student would run
  a two-model session and label it as one, hiding a deployment error behind a
  protocol feature built for a different problem.
- A single-lab council runs but says so, everywhere a verdict appears.
"""

import unittest
import unittest.mock as mock

from quorum import (
    AnthropicProvider,
    ProviderTruncated,
    SessionConfig,
    Council,
    ModelCost,
    OpenAICompatibleProvider,
    ProviderConfigError,
    ProviderError,
    ProviderPool,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    Seat,
    Session,
    ScriptedProvider,
    demo_council,
    render_html,
    render_markdown,
    replay,
)

from test_session import TASK, default_script, scripted_council


class KeyHandlingTests(unittest.TestCase):
    def test_the_key_comes_from_the_environment(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-from-env"}):
            self.assertEqual(AnthropicProvider().api_key, "sk-from-env")

    def test_a_missing_key_is_a_config_error_not_an_outage(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderConfigError):
                AnthropicProvider().complete("claude-x", "hello")
            with self.assertRaises(ProviderConfigError):
                OpenAICompatibleProvider().complete("gpt-x", "hello")

    def test_a_config_error_is_not_treated_as_a_retryable_outage(self):
        # ProviderConfigError is a ProviderError, so the session marks the
        # student absent — but the reason text names the key, which is what
        # makes the deployment bug findable rather than mysterious.
        self.assertTrue(issubclass(ProviderConfigError, ProviderError))
        self.assertFalse(issubclass(ProviderConfigError, ProviderUnavailable))
        self.assertFalse(issubclass(ProviderConfigError, ProviderRateLimited))

    def test_no_key_is_ever_written_into_a_trace_or_report(self):
        council = demo_council()
        script = default_script()
        _, providers = scripted_council(script)
        result = Session(*scripted_council(default_script())).run(
            TASK, session_id="nokey"
        )
        blob = render_html(replay(list(result.events))) + str(
            [e.to_dict() for e in result.events]
        )
        for marker in ("api_key", "ANTHROPIC_API_KEY", "x-api-key", "authorization"):
            self.assertNotIn(marker, blob)

    def test_the_openai_adapter_picks_its_token_parameter_by_host(self):
        self.assertEqual(
            OpenAICompatibleProvider()._max_tokens_param, "max_completion_tokens"
        )
        self.assertEqual(
            OpenAICompatibleProvider(base_url="https://api.together.xyz")._max_tokens_param,
            "max_tokens",
        )
        self.assertEqual(
            OpenAICompatibleProvider(max_tokens_param="max_tokens")._max_tokens_param,
            "max_tokens",
        )

    def test_a_custom_env_var_is_honoured(self):
        with mock.patch.dict("os.environ", {"TOGETHER_API_KEY": "k"}, clear=True):
            provider = OpenAICompatibleProvider(
                base_url="https://api.together.xyz", env_var="TOGETHER_API_KEY",
                name="together",
            )
            self.assertEqual(provider.api_key, "k")
            self.assertEqual(provider.name, "together")


class BackoffTests(unittest.TestCase):
    def test_a_server_named_delay_wins_over_computed_backoff(self):
        provider = AnthropicProvider(api_key="k")
        self.assertEqual(provider._backoff(0, retry_after=3.0), 3.0)

    def test_backoff_is_capped(self):
        provider = AnthropicProvider(api_key="k", backoff_cap=2.0)
        self.assertLessEqual(provider._backoff(10, retry_after=None), 2.0)
        self.assertLessEqual(provider._backoff(0, retry_after=99.0), 2.0)

    def test_an_http_date_retry_after_falls_back_rather_than_crashing(self):
        provider = AnthropicProvider(api_key="k")
        self.assertIsNone(provider._retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}))


class SingleLabTests(unittest.TestCase):
    def council(self, providers):
        return Council(
            students=tuple(
                Seat(f"model-{i}", p, ModelCost(1, 2))
                for i, p in enumerate(providers, start=1)
            ),
            arbiter=Seat("arbiter-model", "lab-x", ModelCost(1, 2)),
        )

    def test_one_provider_across_students_is_flagged(self):
        council = self.council(["anthropic", "anthropic", "anthropic"])
        self.assertTrue(council.single_lab)
        self.assertEqual(council.labs(), ("anthropic",))
        self.assertIn("blind spots correlate", council.warnings[0])

    def test_a_mixed_council_carries_no_warning(self):
        council = self.council(["anthropic", "openai", "google"])
        self.assertFalse(council.single_lab)
        self.assertEqual(council.warnings, ())

    def test_two_labs_out_of_three_is_not_single_lab(self):
        council = self.council(["anthropic", "anthropic", "openai"])
        self.assertFalse(council.single_lab)

    def test_the_demo_council_is_multi_lab(self):
        self.assertFalse(demo_council().single_lab)

    def run_single_lab(self):
        script = default_script()
        council = Council(
            students=tuple(
                Seat(f"model-{i}", "onelab", ModelCost(1, 2)) for i in (1, 2, 3)
            ),
            arbiter=Seat("arbiter-model", "lab-x", ModelCost(1, 2)),
        )
        pool = ProviderPool(
            [
                ScriptedProvider(script, name="onelab"),
                ScriptedProvider(script, name="lab-x"),
            ]
        )
        return Session(council, pool).run(TASK, session_id="onelab")

    def test_the_session_surfaces_it(self):
        result = self.run_single_lab()
        self.assertTrue(result.single_lab)
        self.assertTrue(result.warnings)
        self.assertTrue(result.stats()["single_lab"])

    def test_it_survives_replay(self):
        replayed = replay(list(self.run_single_lab().events))
        self.assertTrue(replayed.single_lab)
        self.assertTrue(replayed.council_warnings)

    def test_the_report_says_so_in_both_formats(self):
        replayed = replay(list(self.run_single_lab().events))
        self.assertIn("Single-lab council", render_html(replayed))
        self.assertIn("Single-lab council", render_markdown(replayed))

    def test_a_mixed_council_report_carries_no_such_banner(self):
        council = demo_council()
        from quorum import convene, mock_pool

        result = convene(TASK, council, mock_pool(council), session_id="mixed")
        self.assertNotIn("Single-lab council", render_html(replay(list(result.events))))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TruncationTests(unittest.TestCase):
    """Truncation must not masquerade as a schema failure.

    Found against a live reasoning model: given a budget of 16 tokens it spent
    all 16 reasoning and returned empty content with `finish_reason: length`.
    The adapter handed back `text=""`, the parser called it an empty response,
    and the session would have recorded a healthy model as producing a
    malformed sheet — blaming the model for our configuration and corrupting
    the claim-compliance metric on the way past.
    """

    class _FakeProvider(OpenAICompatibleProvider):
        def __init__(self, payload, **kw):
            super().__init__(api_key="k", **kw)
            self._payload = payload

        def _request(self, url, body, headers, model_id):
            return self._payload

    def test_a_length_finish_raises_rather_than_returning_empty_text(self):
        provider = self._FakeProvider({
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            "usage": {"completion_tokens": 16,
                      "completion_tokens_details": {"reasoning_tokens": 16}},
        })
        with self.assertRaises(ProviderTruncated) as ctx:
            provider.complete("gpt-x", "hello", 16)
        message = str(ctx.exception)
        self.assertIn("16-token completion budget", message)
        self.assertIn("reasoning", message)
        self.assertIn("max_tokens", message)

    def test_a_normal_finish_still_returns_the_text(self):
        provider = self._FakeProvider({
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })
        self.assertEqual(provider.complete("gpt-x", "hello", 100).text, "ok")

    def test_truncation_is_distinguishable_from_every_other_failure(self):
        for other in (ProviderConfigError, ProviderRateLimited, ProviderUnavailable):
            self.assertFalse(issubclass(ProviderTruncated, other))
            self.assertFalse(issubclass(other, ProviderTruncated))
        self.assertTrue(issubclass(ProviderTruncated, ProviderError))

    def test_the_default_budget_clears_measured_real_usage(self):
        """A real reasoning-model sheet cost 756 visible tokens, and critiques
        run larger. The default must leave room for that plus reasoning."""
        self.assertGreaterEqual(SessionConfig().max_tokens, 4096)


class BaseUrlNormalisationTests(unittest.TestCase):
    """Vendors document base URLs with and without `/v1`.

    Pasting a documented URL verbatim produced `/v1/v1/chat/completions` and a
    404 whose body named neither the model nor the mistake. Both forms now
    resolve to the same endpoint.
    """

    def test_a_trailing_v1_is_absorbed(self):
        for url in ("https://api.moonshot.ai/v1", "https://api.moonshot.ai/v1/",
                    "https://api.moonshot.ai"):
            with self.subTest(url=url):
                provider = OpenAICompatibleProvider(api_key="k", base_url=url)
                self.assertEqual(provider.base_url, "https://api.moonshot.ai")

    def test_a_path_that_merely_ends_in_v1_something_is_untouched(self):
        provider = OpenAICompatibleProvider(api_key="k", base_url="https://host/api/v10")
        self.assertEqual(provider.base_url, "https://host/api/v10")


class ChatPathTests(unittest.TestCase):
    """"OpenAI-compatible" does not imply OpenAI's URL.

    Google's compatibility layer serves `/v1beta/openai/chat/completions`.
    With the path hardcoded, the adapter's claim to cover any compatible
    vendor was really a claim to cover vendors that also copied the routing.
    """

    def test_the_default_path_is_unchanged(self):
        self.assertEqual(
            OpenAICompatibleProvider(api_key="k").chat_path, "/v1/chat/completions"
        )

    def test_a_vendor_specific_path_is_honoured(self):
        provider = OpenAICompatibleProvider(
            api_key="k",
            base_url="https://generativelanguage.googleapis.com",
            chat_path="/v1beta/openai/chat/completions",
        )
        self.assertEqual(provider.chat_path, "/v1beta/openai/chat/completions")

    def test_slashes_are_normalised(self):
        for given in ("v1/chat/completions", "/v1/chat/completions/"):
            self.assertEqual(
                OpenAICompatibleProvider(api_key="k", chat_path=given).chat_path,
                "/v1/chat/completions",
            )


class TransportResilienceTests(unittest.TestCase):
    """A transient socket error must not escape as an unhandled exception.

    urllib wraps most transport problems in URLError, but a reset *after* the
    request is sent surfaces as a bare ConnectionResetError — an OSError that
    slipped past every handler. It killed a 20-task benchmark run partway
    through: a blip taking down the exact long run the retries exist for.
    """

    class _ResettingProvider(OpenAICompatibleProvider):
        def __init__(self, fail_times, **kw):
            super().__init__(api_key="k", sleep=lambda _: None, **kw)
            self.fail_times, self.calls = fail_times, 0

        def _raw(self):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise ConnectionResetError(54, "Connection reset by peer")
            return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def test_connection_reset_is_classified_as_an_outage(self):
        import socket
        import urllib.error
        import urllib.request
        from unittest import mock as m

        provider = OpenAICompatibleProvider(api_key="k", max_retries=1,
                                            sleep=lambda _: None)
        with m.patch.object(urllib.request, "urlopen",
                            side_effect=ConnectionResetError(54, "reset")):
            with self.assertRaises(ProviderUnavailable) as ctx:
                provider.complete("gpt-x", "hello", 100)
        self.assertIn("ConnectionResetError", str(ctx.exception))
        self.assertIn("transport failed", str(ctx.exception))

    def test_a_reset_that_clears_on_retry_returns_normally(self):
        import urllib.request
        from unittest import mock as m

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                import json as j
                return j.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()

        provider = OpenAICompatibleProvider(api_key="k", max_retries=2,
                                            sleep=lambda _: None)
        with m.patch.object(urllib.request, "urlopen",
                            side_effect=[ConnectionResetError(54, "reset"), FakeResponse()]):
            self.assertEqual(provider.complete("gpt-x", "hello", 100).text, "ok")


class WallClockDeadlineTests(unittest.TestCase):
    """A per-read timeout cannot bound a stalled stream.

    A benchmark run held one ESTABLISHED connection for five and a half hours
    with 2.6 seconds of CPU and no output. The 120-second `timeout=` never
    fired, because it bounds each individual recv rather than the call, and a
    server that dribbles bytes resets it on every read.
    """

    def test_a_call_that_never_returns_raises_rather_than_hanging(self):
        import time as t
        import urllib.request
        from unittest import mock as m

        provider = OpenAICompatibleProvider(api_key="k", deadline=0.3, max_retries=0,
                                            sleep=lambda _: None)

        def never_returns(*args, **kwargs):
            t.sleep(30)  # the daemon thread is abandoned, not awaited

        started = t.time()
        with m.patch.object(urllib.request, "urlopen", side_effect=never_returns):
            with self.assertRaises(ProviderTimeout) as ctx:
                provider.complete("gpt-x", "hello", 100)
        self.assertLess(t.time() - started, 5, "the deadline did not bound the call")
        self.assertIn("stalled stream", str(ctx.exception))

    def test_the_deadline_does_not_interfere_with_a_normal_response(self):
        import json as j
        import urllib.request
        from unittest import mock as m

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return j.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()

        provider = OpenAICompatibleProvider(api_key="k", deadline=10)
        with m.patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
            self.assertEqual(provider.complete("gpt-x", "hello", 100).text, "ok")

    def test_the_default_deadline_exceeds_the_per_read_timeout(self):
        provider = OpenAICompatibleProvider(api_key="k")
        self.assertGreater(provider.deadline, provider.timeout)
