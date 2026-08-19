"""Deterministic repair of malformed model JSON, before anyone pays to re-ask.

A model that emits an unterminated object has usually done the *work* — the
argument is there, the closing brace is not. Dropping the seat for that costs
the tokens twice over: once for the generation that was thrown away, and again
in the debate, because a seat that vanishes in round 2 leaves its round-1
position standing undefended. That is not a hypothetical. On 2026-08-19 a
`deepseek-chat` critique was cut mid-object, the seat was marked absent for the
round, and its "run as-is" position survived because nobody was left to attack
it.

**Repair here is syntactic and nothing else.** It closes brackets, drops
separators that lead nowhere, and escapes quotes that were never escaped. It
never invents a claim, completes a sentence, guesses a number, or fills a
missing field. The distinction matters more than it looks: a repaired object
that reads as complete but was silently finished by this module would be a
fabricated argument attributed to a real model, which is precisely the failure
the project exists to make impossible.

So the contract is: content survives verbatim or not at all. Where truncation
cost us a trailing fragment, the fragment is dropped rather than guessed, and
`RepairReport` records that it happened. Every repair is reported, every report
reaches the trace, and the report can disclose it. A repair nobody can see is
indistinguishable from the model having got it right, and those must never look
the same.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .sheets import SheetParseError, extract_json

# How far back the truncation repair will walk looking for a point the object
# can be legally closed. Bounded because the search tries a parse per
# candidate: unbounded backtracking over a long malformed sheet turns a cheap
# recovery into a slow one, and a response that needs thousands of characters
# removed is not truncated, it is garbage, and should be re-asked instead.
MAX_CUT_CANDIDATES = 64


@dataclass(frozen=True)
class RepairReport:
    """What was done to a response to make it parse. Rides into the trace.

    `repaired` false with `steps` empty is the ordinary case: the model got it
    right and nothing was touched.
    """

    repaired: bool = False
    truncated: bool = False
    dropped_trailing: bool = False
    steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repaired": self.repaired,
            "truncated": self.truncated,
            "dropped_trailing": self.dropped_trailing,
            "steps": list(self.steps),
        }

    def as_payload(self) -> dict[str, Any]:
        """Trace payload fragment: empty unless something was actually repaired.

        Empty on the clean path so an untouched event carries no new keys and
        an existing replay sees exactly what it saw before. When a repair did
        happen the keys are present and searchable, which is the whole point:
        a patched response that looks identical to a correct one is the one
        outcome this must not produce.
        """
        return {"repair": self.to_dict()} if self.repaired else {}

    @classmethod
    def from_dict(cls, data: Any) -> RepairReport:
        if not isinstance(data, dict):
            return cls()
        return cls(
            repaired=bool(data.get("repaired", False)),
            truncated=bool(data.get("truncated", False)),
            dropped_trailing=bool(data.get("dropped_trailing", False)),
            steps=tuple(data.get("steps", ()) or ()),
        )


CLEAN = RepairReport()


# --------------------------------------------------------------------------
# string-aware scanning
# --------------------------------------------------------------------------


def _scan(text: str) -> tuple[list[int], list[str], bool, bool]:
    """One pass over a JSON-ish string.

    Returns the indices that are *inside* a string literal, the stack of
    containers still open at the end, whether the scan ended mid-string, and
    whether it ended on a dangling backslash. Everything else in this module
    is built on this, because every repair has to know what is structure and
    what is somebody's prose — claim text routinely contains braces, brackets
    and quotes, and a repair that cannot tell them apart will corrupt content
    while believing it is fixing syntax.
    """
    inside: list[int] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            inside.append(i)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            inside.append(i)
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    return inside, stack, in_string, escaped


def _outside(text: str) -> set[int]:
    inside, _, _, _ = _scan(text)
    return set(range(len(text))) - set(inside)


# --------------------------------------------------------------------------
# individual repairs
# --------------------------------------------------------------------------


def _strip_wrapper(text: str) -> str:
    """Reduce a response to the first thing that looks like a JSON object.

    `sheets.extract_json` already does this for the responses it can parse;
    repeated here because the repairs below need the same starting point on
    responses it could not.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        body = stripped[3:]
        if body[:4].lower().startswith("json"):
            body = body[4:]
        end = body.rfind("```")
        stripped = (body[:end] if end != -1 else body).strip()
    start = stripped.find("{")
    return stripped[start:] if start > 0 else stripped


def _drop_trailing_commas(text: str) -> str:
    """`{"a": 1,}` and `[1, 2, ]` — legal in JavaScript, not in JSON."""
    outside = _outside(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "," and i in outside:
            rest = text[i + 1 :]
            if rest.lstrip()[:1] in ("}", "]"):
                continue  # a comma with nothing after it but a closer
        out.append(ch)
    return "".join(out)


def _escape_inner_quotes(text: str) -> str:
    """Escape quotes a model opened inside a string and never escaped.

    The riskiest repair here, and the last one tried. A quote inside a string
    value is only distinguishable from the quote that *ends* that value by
    what follows it: a real closing quote is followed by `,` `}` `]` or `:`,
    possibly after whitespace. Anything else and the model was quoting
    somebody. That heuristic is not a parser and it can be wrong, so the step
    is recorded by name in the report — a reader who distrusts it can find
    every response it touched.

    Content is preserved either way: escaping a quote changes the encoding of
    a string, never its text.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            nxt = text[i + 1 :].lstrip()[:1]
            if nxt in (",", "}", "]", ":", ""):
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')  # a quote the model meant literally
            continue
        out.append(ch)
    return "".join(out)


def _cut_points(text: str) -> list[int]:
    """Indices where the text could plausibly be cut and then closed.

    Walking back from the end of a truncated response, these are the places a
    value or a comma ended — the only places where appending closers can
    produce something legal.
    """
    inside, _, _, _ = _scan(text)
    inside_set = set(inside)
    points: list[int] = []
    for i, ch in enumerate(text):
        if i in inside_set:
            continue
        if ch in ',"}]' or ch.isdigit() or ch in "eltn":  # true/false/null tails
            points.append(i + 1)
    return points


def _close(text: str) -> str | None:
    """Append whatever closers the open containers need. None if none are."""
    _, stack, in_string, escaped = _scan(text)
    if not stack and not in_string:
        return None
    out = text
    if escaped:
        # A trailing backslash would escape the quote we are about to add.
        out = out[:-1]
    if in_string:
        out += '"'
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def _repair_truncated(text: str) -> tuple[Any, bool] | None:
    """Close an object the model never finished. `(value, dropped_trailing)`.

    Tries the longest completion first, so the maximum amount of what the
    model actually wrote survives. Only when that will not parse does it walk
    back to earlier cut points, discarding the trailing fragment — never
    completing it.
    """
    closed = _close(text)
    if closed is not None:
        try:
            return json.loads(closed), False
        except json.JSONDecodeError:
            pass
    for cut in sorted(_cut_points(text), reverse=True)[:MAX_CUT_CANDIDATES]:
        head = text[:cut]
        # Strip separators that now lead nowhere, then close.
        trimmed = head.rstrip()
        while trimmed and trimmed[-1] in ",:":
            trimmed = trimmed[:-1].rstrip()
        candidate = _close(trimmed) or trimmed
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def recover(text: Any, *, actor: str = "") -> tuple[Any, RepairReport]:
    """Parse `text`, repairing it if that is the only way, and say what was done.

    Returns the decoded value and a report. Raises `SheetParseError` when even
    repair cannot produce JSON — the caller then re-asks or marks the seat
    absent exactly as it did before this module existed.

    The clean path is first and untouched: a well-formed response takes the
    same route through `extract_json` it always took, and comes back with an
    empty report. Nothing about the common case changes.
    """
    if isinstance(text, (dict, list)):
        return text, CLEAN
    try:
        return extract_json(text, actor=actor), CLEAN
    except SheetParseError as exc:
        # Rebound deliberately: Python unbinds the `as` name when the except
        # block ends, and this error is re-raised much further down as the
        # honest failure when no repair worked.
        original = exc

    if not isinstance(text, str) or not text.strip():
        raise original

    base = _strip_wrapper(text)
    steps: list[str] = []

    # Ordered cheapest and safest first. Each is cumulative: a response can
    # need several, and the report names every one that was applied.
    candidate = base
    for name, fn in (
        ("stripped_wrapper", lambda s: s),  # already applied; recorded if it changed
        ("dropped_trailing_commas", _drop_trailing_commas),
        ("escaped_inner_quotes", _escape_inner_quotes),
    ):
        if name == "stripped_wrapper":
            if base != text.strip():
                steps.append(name)
            continue
        nxt = fn(candidate)
        if nxt != candidate:
            candidate = nxt
            steps.append(name)
        try:
            return json.loads(candidate), RepairReport(
                repaired=True, steps=tuple(steps)
            )
        except json.JSONDecodeError:
            continue

    truncated = _repair_truncated(candidate)
    if truncated is not None:
        value, dropped = truncated
        steps.append("closed_truncated_json")
        if dropped:
            steps.append("dropped_incomplete_trailing_value")
        return value, RepairReport(
            repaired=True,
            truncated=True,
            dropped_trailing=dropped,
            steps=tuple(steps),
        )

    raise original
