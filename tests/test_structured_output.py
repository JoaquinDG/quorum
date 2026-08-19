"""Schema-constrained output: the request shape, and the fallback.

Two things need guarding. The first is that the schemas describe the contract
`sheets.py` already enforces and not some adjacent contract of their own —
they are a way of asking for the existing format, and a schema that drifted
would quietly become a second definition of what is acceptable. The second is
that a provider without schema support is called exactly as it was before, so
enabling this feature cannot change behaviour for three quarters of a lineup.
"""

import json
import unittest

from quorum import ModelCost, ProviderPool, Seat, Session, SessionConfig, Council
from quorum.providers.http import AnthropicProvider, OpenAICompatibleProvider
from quorum.schemas import (
    CRITIQUE_SCHEMA, REVISION_SCHEMA, SCHEMA_FOR_ROUND, SCHEMAS, SHEET_SCHEMA,
    VERDICT_SCHEMA,
)
from quorum.sheets import (
    MAX_CLAIMS, parse_critique, parse_revision, parse_sheet, parse_verdict,
)


def _council(provider_name="openai"):
    """A council of the minimum legal size (the engine requires 2-5 students)."""
    return Council(
        students=(Seat("m1", provider_name, ModelCost(1, 1)),
                  Seat("m2", provider_name, ModelCost(1, 1))),
        arbiter=Seat("a", provider_name, ModelCost(1, 1)),
    )


class Capture:
    """Records the request body an adapter would have sent."""

    def __init__(self, response):
        self.response = response
        self.body = None

    def __call__(self, url, body, headers, model_id):
        self.body = json.loads(body.decode())
        return self.response


ANTHROPIC_OK = {"content": [{"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}
OPENAI_OK = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class SchemasMatchTheContract(unittest.TestCase):
    """The schemas must not invent constraints the parser does not have."""

    def test_claim_cap_is_the_parser_s_cap(self):
        self.assertEqual(SHEET_SCHEMA["properties"]["claims"]["maxItems"], MAX_CLAIMS)

    def test_a_sheet_matching_the_schema_parses(self):
        sheet = {
            "position": "Refactor in place.",
            "claims": [{"n": 1, "text": "The edge cases live in the current code."}],
            "assumptions": ["traffic stays flat"],
            "would_change_my_mind": ["a throughput ceiling"],
            "confidence": 0.6,
            "nuance": "",
        }
        self.assertEqual(set(SHEET_SCHEMA["required"]) - set(sheet), set())
        parsed = parse_sheet(sheet)
        self.assertEqual(parsed.position, "Refactor in place.")

    def test_a_critique_matching_the_schema_parses(self):
        critique = {"objections": [{
            "sheet": "A", "claim_n": 1,
            "argument": "The base rate cited predates the new ingest sources entirely.",
        }]}
        parsed = parse_critique(critique, allowed={"A": (1,)})
        self.assertEqual(len(parsed), 1)

    def test_a_verdict_matching_the_schema_parses(self):
        verdict = {
            "final_answer": "Refactor in place.",
            "confidence_note": "Not fully settled.",
            "minority_report": [
                {"source": "Student 1", "kind": "claim", "substance": "Rebuild."},
            ],
        }
        parse_verdict(verdict, allowed_sources=("Student 1",))

    def test_verdict_kinds_are_the_ones_the_parser_accepts(self):
        kinds = VERDICT_SCHEMA["properties"]["minority_report"]["items"]["properties"]["kind"]["enum"]
        self.assertEqual(sorted(kinds), ["claim", "objection"])

    def test_revision_adds_only_the_two_round_three_fields(self):
        extra = set(REVISION_SCHEMA["properties"]) - set(SHEET_SCHEMA["properties"])
        self.assertEqual(extra, {"changed_position", "because"})

    def test_every_round_has_a_schema(self):
        self.assertEqual(sorted(SCHEMA_FOR_ROUND), [1, 2, 3, 4])


class StrictModeRequiresEveryProperty(unittest.TestCase):
    """OpenAI's strict structured outputs require every key in `properties` to
    appear in `required`, at every level of nesting. There is no way to say
    "this one may be omitted": a field the protocol treats as optional has to
    be declared nullable and sent as an explicit null.

    This is not a tidiness rule, it is a precondition for the call happening at
    all. The API rejects a non-conforming schema with an HTTP 400 naming the
    field, before the model runs, so the seat is marked absent and the council
    deliberates a student short while the trace records a provider error rather
    than a format one. `SHEET_SCHEMA` omitted `nuance` and `REVISION_SCHEMA`
    omitted `nuance` and `because`, which is why gpt-5.1 never answered in
    session q-ee46e00b6082.
    """

    def violations(self, node, path):
        """Objects in the tree that declare a property they do not require.

        Walks dicts and lists alike, because the constraint applies to the
        schema of every array item and every nested object, not just the root.
        """
        found = []
        if isinstance(node, dict):
            if "properties" in node:
                missing = sorted(set(node["properties"]) - set(node.get("required", [])))
                if missing:
                    found.append((path, missing))
            for key, value in node.items():
                found += self.violations(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found += self.violations(value, f"{path}[{index}]")
        return found

    def test_every_schema_requires_every_property_it_declares(self):
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                self.assertEqual(self.violations(schema, name), [])

    def test_the_walk_reaches_below_the_top_level(self):
        """A check that only read the root would have passed the broken schemas
        as well, so it has to be shown failing on a nested violation."""
        nested = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                        "required": ["a"],
                    },
                },
            },
            "required": ["rows"],
        }
        self.assertEqual(
            self.violations(nested, "nested"),
            [("nested.properties.rows.items", ["b"])],
        )

    def test_the_optional_fields_are_nullable_rather_than_absent(self):
        """Requiring them is only sound because null is a legal value for them."""
        self.assertIn("null", SHEET_SCHEMA["properties"]["nuance"]["type"])
        self.assertIn("null", REVISION_SCHEMA["properties"]["nuance"]["type"])
        self.assertIn("null", REVISION_SCHEMA["properties"]["because"]["type"])


class AnExplicitNullReadsAsAnOmission(unittest.TestCase):
    """Strict mode buys its guarantee by forbidding omissions, so a model with
    no nuance to add now sends `"nuance": null` where it used to send nothing
    at all. If the parser read those two differently, requiring the fields
    would have moved the bug rather than removed it."""

    BASE = {
        "position": "Refactor in place.",
        "claims": [{"n": 1, "text": "The edge cases live in the current code."}],
        "assumptions": ["traffic stays flat"],
        "would_change_my_mind": ["a throughput ceiling"],
        "confidence": 0.6,
    }

    def test_a_null_nuance_is_an_empty_nuance(self):
        self.assertEqual(parse_sheet(dict(self.BASE, nuance=None), actor="A").nuance, "")

    def test_a_null_nuance_parses_to_the_same_sheet_as_no_nuance(self):
        self.assertEqual(
            parse_sheet(dict(self.BASE, nuance=None), actor="A"),
            parse_sheet(dict(self.BASE), actor="A"),
        )

    def test_a_null_because_cites_nobody(self):
        revision = parse_revision(
            dict(self.BASE, changed_position=False, nuance=None, because=None),
            allowed={},
            actor="A",
        )
        self.assertEqual(revision.because, ())

    def test_a_null_because_parses_to_the_same_revision_as_no_because(self):
        self.assertEqual(
            parse_revision(
                dict(self.BASE, changed_position=False, nuance=None, because=None),
                allowed={}, actor="A",
            ),
            parse_revision(
                dict(self.BASE, changed_position=False), allowed={}, actor="A",
            ),
        )


class AnthropicRequestShape(unittest.TestCase):
    def test_schema_is_sent_as_a_forced_tool_call(self):
        provider = AnthropicProvider(api_key="k")
        capture = Capture(ANTHROPIC_OK)
        provider._request = capture
        provider.complete("claude-sonnet-5", "prompt", schema=("answer_sheet", SHEET_SCHEMA))
        self.assertEqual(capture.body["tool_choice"],
                         {"type": "tool", "name": "answer_sheet"})
        self.assertEqual(capture.body["tools"][0]["input_schema"], SHEET_SCHEMA)

    def test_the_prompt_is_not_modified(self):
        """The cacheable prefix must survive schema enforcement untouched."""
        provider = AnthropicProvider(api_key="k")
        capture = Capture(ANTHROPIC_OK)
        provider._request = capture
        provider.complete("claude-sonnet-5", "EXACT PROMPT",
                          schema=("answer_sheet", SHEET_SCHEMA))
        self.assertEqual(capture.body["messages"],
                         [{"role": "user", "content": "EXACT PROMPT"}])

    def test_no_schema_means_the_old_request(self):
        provider = AnthropicProvider(api_key="k")
        capture = Capture(ANTHROPIC_OK)
        provider._request = capture
        provider.complete("claude-sonnet-5", "prompt")
        self.assertNotIn("tools", capture.body)
        self.assertNotIn("tool_choice", capture.body)

    def test_a_tool_use_reply_is_returned_as_text(self):
        """Downstream parsing must not learn that tool use happened."""
        provider = AnthropicProvider(api_key="k")
        provider._request = Capture({
            "content": [{"type": "tool_use", "name": "answer_sheet",
                         "input": {"position": "refactor"}}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        completion = provider.complete("claude-sonnet-5", "p",
                                       schema=("answer_sheet", SHEET_SCHEMA))
        self.assertEqual(json.loads(completion.text), {"position": "refactor"})


class OpenAICompatibleRequestShape(unittest.TestCase):
    def test_openai_gets_a_json_schema(self):
        provider = OpenAICompatibleProvider(api_key="k")
        capture = Capture(OPENAI_OK)
        provider._request = capture
        provider.complete("gpt-5.1", "p", schema=("critique", CRITIQUE_SCHEMA))
        fmt = capture.body["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["schema"], CRITIQUE_SCHEMA)
        self.assertTrue(fmt["json_schema"]["strict"])

    def test_deepseek_gets_json_mode(self):
        """The seat that failed. JSON mode is what it supports."""
        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://api.deepseek.com", name="deepseek"
        )
        capture = Capture(OPENAI_OK)
        provider._request = capture
        provider.complete("deepseek-chat", "p", schema=("critique", CRITIQUE_SCHEMA))
        self.assertEqual(capture.body["response_format"], {"type": "json_object"})

    def test_schema_mode_none_sends_nothing(self):
        provider = OpenAICompatibleProvider(api_key="k", schema_mode=None)
        capture = Capture(OPENAI_OK)
        provider._request = capture
        provider.complete("m", "p", schema=("critique", CRITIQUE_SCHEMA))
        self.assertNotIn("response_format", capture.body)

    def test_auto_and_explicit_none_are_different_intentions(self):
        self.assertEqual(OpenAICompatibleProvider(api_key="k").schema_mode,
                         "json_schema")
        self.assertIsNone(
            OpenAICompatibleProvider(api_key="k", schema_mode=None).schema_mode
        )


class FeatureDetectionAndFallback(unittest.TestCase):
    """A provider that predates this feature must be called as it always was."""

    def test_a_legacy_provider_is_never_offered_a_schema(self):
        calls = []

        class Legacy:
            """The original one-method protocol. No schema keyword at all."""
            name = "legacy"

            def complete(self, model_id, prompt, max_tokens=2048):
                calls.append((model_id, max_tokens))
                return __import__("quorum").providers.base.Completion(
                    text=json.dumps({
                        "position": "hold",
                        "claims": [{"n": 1, "text": "The current path is adequate."}],
                        "assumptions": ["a"], "would_change_my_mind": ["b"],
                        "confidence": 0.5, "nuance": "",
                    }),
                    model_id=model_id,
                )

        council = Council(
            students=(Seat("m1", "legacy", ModelCost(1, 1)),
                      Seat("m2", "legacy", ModelCost(1, 1))),
            arbiter=Seat("m3", "legacy", ModelCost(1, 1)),
        )
        session = Session(council, ProviderPool([Legacy()]),
                          config=SessionConfig(structured_output=True))
        session.run("Should we hold or change?")
        self.assertTrue(calls, "the legacy provider was never called")

    def test_structured_output_can_be_switched_off(self):
        provider = OpenAICompatibleProvider(api_key="k")
        session = Session(_council(), ProviderPool([provider]),
                          config=SessionConfig(structured_output=False))
        self.assertIsNone(session._schema_for(provider, 1))

    def test_schema_is_selected_by_round(self):
        provider = OpenAICompatibleProvider(api_key="k")
        session = Session(_council(), ProviderPool([provider]),
                          config=SessionConfig(structured_output=True))
        self.assertEqual(session._schema_for(provider, 1)[1], SHEET_SCHEMA)
        self.assertEqual(session._schema_for(provider, 2)[1], CRITIQUE_SCHEMA)
        self.assertEqual(session._schema_for(provider, 4)[1], VERDICT_SCHEMA)
        self.assertIsNone(session._schema_for(provider, 0))


if __name__ == "__main__":
    unittest.main()
