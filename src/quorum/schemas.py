"""JSON Schemas for the four responses the protocol asks for.

These exist to stop malformation before it happens. A provider that can
constrain generation to a schema will not emit the unterminated object that
cost a seat its round on 2026-08-19, and a failure prevented is cheaper than a
failure repaired — no discarded generation, no re-ask, no absent seat.

**These schemas describe the existing contract; they do not extend it.** Every
constraint here is one `sheets.py` already enforces, and the parser remains
the authority: a provider that ignores the schema, or one that has no schema
support at all, is validated exactly as before. That is what keeps the fallback
path honest — the schema is a way of asking, never a second definition of what
is acceptable. `tests/test_structured_output.py` checks the two against each
other so they cannot drift apart silently.

The prose instructions in `prompts.py` are deliberately left alone. They are
what a model without schema support reads, they are the shared prefix the
caching work depends on, and duplicating them here as schema descriptions
would create a second place for the format to be specified and a first place
for it to disagree with itself.
"""

from __future__ import annotations

from typing import Any

from .sheets import MAX_CLAIMS

# Values `sheets.parse_verdict` accepts for a minority-report entry's `kind`.
MINORITY_KINDS = ("claim", "objection")


def _sheet_properties() -> dict[str, Any]:
    return {
        "position": {"type": "string"},
        "claims": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_CLAIMS,
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["n", "text"],
                "additionalProperties": False,
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "would_change_my_mind": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "nuance": {"type": "string"},
    }


SHEET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _sheet_properties(),
    "required": [
        "position", "claims", "assumptions", "would_change_my_mind", "confidence",
    ],
    "additionalProperties": False,
}


CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objections": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "sheet": {"type": "string"},
                    "claim_n": {"type": "integer"},
                    "argument": {"type": "string"},
                },
                "required": ["sheet", "claim_n", "argument"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objections"],
    "additionalProperties": False,
}


def _revision_properties() -> dict[str, Any]:
    props = _sheet_properties()
    props["changed_position"] = {"type": "boolean"}
    props["because"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "critic": {"type": "string"},
                "claim_n": {"type": "integer"},
            },
            "required": ["critic", "claim_n"],
            "additionalProperties": False,
        },
    }
    return props


REVISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _revision_properties(),
    "required": [
        "position", "claims", "assumptions", "would_change_my_mind",
        "confidence", "changed_position",
    ],
    "additionalProperties": False,
}


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "confidence_note": {"type": "string"},
        "minority_report": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "kind": {"type": "string", "enum": list(MINORITY_KINDS)},
                    "substance": {"type": "string"},
                },
                "required": ["source", "kind", "substance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["final_answer", "confidence_note", "minority_report"],
    "additionalProperties": False,
}


# Named so an adapter can label the tool or schema it sends.
SCHEMAS: dict[str, dict[str, Any]] = {
    "answer_sheet": SHEET_SCHEMA,
    "critique": CRITIQUE_SCHEMA,
    "revision": REVISION_SCHEMA,
    "verdict": VERDICT_SCHEMA,
}

SCHEMA_FOR_ROUND: dict[int, tuple[str, dict[str, Any]]] = {
    1: ("answer_sheet", SHEET_SCHEMA),
    2: ("critique", CRITIQUE_SCHEMA),
    3: ("revision", REVISION_SCHEMA),
    4: ("verdict", VERDICT_SCHEMA),
}
