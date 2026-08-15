"""The answer sheet: Quorum's entire medium of exchange.

Students never send each other prose. They fill in a rigid template and
exchange *that*. This is the anonymization mechanism, not a side effect of
one: most of the style signal that would identify a model family is never
generated in the first place, so there is nothing to detect and nothing to
paraphrase away. It is also what makes position tracking possible — a claim
with a number can be objected to, dropped, or edited, and all three are
mechanically detectable. Free-form debate supports none of that.

Parsing follows Switchboard's split: **tolerant of format, strict about
meaning.** Fenced JSON, prose-wrapped JSON, `"0.7"` for `0.7`, and a claim
that arrives as `"3. foo"` instead of `{"n": 3, "text": "foo"}` are all
accepted, because those are transport accidents. Six claims when the cap is
five, a confidence of `1.4`, or claims numbered `1, 2, 4` are rejected,
because those change what the sheet *says*. Nothing is silently coerced: a
sheet that cannot be parsed makes its author absent for the round, which the
session reports, rather than becoming a quietly repaired sheet that the final
answer then rests on.

One rule is deliberately *not* enforced as a parse error: "one sentence
each". Sentence counting is a heuristic (abbreviations, decimals, quoted
text), and hard-failing a substantive claim over a semicolon would throw away
real content to satisfy a regex. Multi-sentence claims are recorded as
`compliance_warnings` instead, which is exactly what the claim-compliance
success metric needs — a rate, not a crash.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

MAX_CLAIMS = 5
"""Hard cap on claims per sheet. The cap is the point: a position that cannot
be compressed to five one-sentence claims is a position that cannot be
critiqued claim by claim, which is the only kind of critique this protocol
can track. The `nuance` field is the escape valve."""

MIN_ARGUMENT_CHARS = 40
"""Below this, an "objection" is a reaction. Coarse, and deliberately so —
see `_VAGUE_AGREEMENT` for the other half of the filter and its limits."""

MAX_SENTENCE_CHARS = 400

_VAGUE_AGREEMENT = re.compile(
    r"^\s*(i\s+)?("
    r"agree|concur|no\s+objection|nothing\s+to\s+add|looks?\s+good|"
    r"sounds?\s+(right|good)|makes?\s+sense|seems?\s+(right|fine|reasonable)|"
    r"this\s+is\s+(right|correct|fine)|n/?a|none"
    r")\b[\s.!]*$",
    re.IGNORECASE,
)
"""Catches the degenerate case the format is meant to make impossible: a
critique slot filled with assent. It is a substring rule and a model that
wants to be agreeable in more words will get past it. The structural defence
is the schema (you must name a sheet and a claim number); this only stops the
laziest evasion, and the position-change rate is what actually reveals
whether critique is happening."""

_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z\"'(\[])")


class SheetError(ValueError):
    """Base for every schema violation. Carries the actor for the trace."""

    def __init__(self, message: str, *, actor: str = "") -> None:
        super().__init__(message)
        self.actor = actor


class SheetParseError(SheetError):
    """No JSON object could be recovered from the response at all."""


class SheetSchemaError(SheetError):
    """JSON was recovered but it is not a valid sheet / critique / verdict."""


class NonCompliantCritique(SheetSchemaError):
    """A critique parsed, but does not engage claims as the format requires."""


# --------------------------------------------------------------------------
# tolerant JSON extraction
# --------------------------------------------------------------------------


def extract_json(text: str, *, actor: str = "") -> Any:
    """Recover the first complete JSON value from a model response.

    Models wrap JSON in fences, preambles ("Here is my answer sheet:"), and
    trailing commentary. All three are format noise. Scanning for a balanced
    object beats a regex because claim text routinely contains braces and
    quotes.
    """
    if not isinstance(text, str) or not text.strip():
        raise SheetParseError("empty response", actor=actor)

    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise SheetParseError(
            f"no JSON object found in response: {stripped[:120]!r}", actor=actor
        )

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise SheetParseError(
                        f"malformed JSON object: {exc}", actor=actor
                    ) from None
    raise SheetParseError("unterminated JSON object in response", actor=actor)


# --------------------------------------------------------------------------
# the sheet
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    number: int
    text: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.number}. {self.text}"


@dataclass(frozen=True)
class AnswerSheet:
    """One student's position, in the only shape the protocol transports."""

    position: str
    claims: tuple[Claim, ...]
    assumptions: tuple[str, ...]
    would_change_my_mind: tuple[str, ...]
    confidence: float
    nuance: str = ""
    compliance_warnings: tuple[str, ...] = ()

    def claim(self, number: int) -> Claim | None:
        for c in self.claims:
            if c.number == number:
                return c
        return None

    @property
    def claim_numbers(self) -> tuple[int, ...]:
        return tuple(c.number for c in self.claims)

    def to_dict(self, *, include_nuance: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "position": self.position,
            "claims": [{"n": c.number, "text": c.text} for c in self.claims],
            "assumptions": list(self.assumptions),
            "would_change_my_mind": list(self.would_change_my_mind),
            "confidence": self.confidence,
        }
        if include_nuance and self.nuance:
            out["nuance"] = self.nuance
        if self.compliance_warnings:
            out["compliance_warnings"] = list(self.compliance_warnings)
        return out

    @classmethod
    def from_dict(cls, data: Any, *, actor: str = "") -> AnswerSheet:
        return parse_sheet(data, actor=actor)


def _require_text(value: Any, label: str, actor: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SheetSchemaError(f"{label} must be a non-empty string", actor=actor)
    return value.strip()


def _require_str_list(value: Any, label: str, actor: str) -> tuple[str, ...]:
    if isinstance(value, str):
        # A single falsifier written as a bare string is a format slip, not a
        # different meaning.
        value = [value]
    if not isinstance(value, list) or not value:
        raise SheetSchemaError(
            f"{label} must be a non-empty list of strings", actor=actor
        )
    items = tuple(_require_text(v, f"{label}[{i}]", actor) for i, v in enumerate(value))
    return items


def _parse_confidence(value: Any, actor: str) -> float:
    if isinstance(value, bool):
        raise SheetSchemaError("confidence must be a number, not a boolean", actor=actor)
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%")) / (
                100.0 if value.strip().endswith("%") else 1.0
            )
        except ValueError:
            raise SheetSchemaError(
                f"confidence is not a number: {value!r}", actor=actor
            ) from None
    if not isinstance(value, (int, float)):
        raise SheetSchemaError(
            f"confidence must be a number, got {type(value).__name__}", actor=actor
        )
    value = float(value)
    if not math.isfinite(value):
        raise SheetSchemaError("confidence must be finite", actor=actor)
    if not 0.0 <= value <= 1.0:
        # Clamping would launder a schema violation into a usable number, and
        # confidence is the one field the report quotes directly.
        raise SheetSchemaError(
            f"confidence {value} outside [0, 1]", actor=actor
        )
    return value


_NUMBER_PREFIX = re.compile(r"^\s*(\d+)\s*[.)\]:-]\s+(.*)$", re.DOTALL)


def _parse_claims(raw: Any, actor: str) -> tuple[Claim, ...]:
    if not isinstance(raw, list):
        raise SheetSchemaError("claims must be a list", actor=actor)
    if not raw:
        raise SheetSchemaError("a sheet must carry at least one claim", actor=actor)
    if len(raw) > MAX_CLAIMS:
        raise SheetSchemaError(
            f"{len(raw)} claims exceeds the cap of {MAX_CLAIMS}", actor=actor
        )

    claims: list[Claim] = []
    for index, item in enumerate(raw, start=1):
        declared: int | None = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text_value = item.get("text", item.get("claim"))
            text = _require_text(text_value, f"claims[{index}].text", actor)
            number_value = item.get("n", item.get("number"))
            if number_value is not None:
                if isinstance(number_value, bool) or not isinstance(
                    number_value, (int, str)
                ):
                    raise SheetSchemaError(
                        f"claims[{index}] number must be an integer", actor=actor
                    )
                try:
                    declared = int(number_value)
                except ValueError:
                    raise SheetSchemaError(
                        f"claims[{index}] number is not an integer: {number_value!r}",
                        actor=actor,
                    ) from None
        else:
            raise SheetSchemaError(
                f"claims[{index}] must be a string or an object", actor=actor
            )

        text = _require_text(text, f"claims[{index}]", actor)
        prefixed = _NUMBER_PREFIX.match(text)
        if prefixed:
            # "3. foo" carries its own number; honour it so contiguity is
            # checked against what the model actually meant.
            if declared is None:
                declared = int(prefixed.group(1))
            text = prefixed.group(2).strip()
            text = _require_text(text, f"claims[{index}]", actor)

        claims.append(Claim(number=declared if declared is not None else index, text=text))

    numbers = [c.number for c in claims]
    if sorted(numbers) != list(range(1, len(claims) + 1)):
        # Objections address claims by number. Gaps or duplicates make an
        # objection ambiguous, and an ambiguous objection cannot be tracked
        # through the revision round — which is the whole product.
        raise SheetSchemaError(
            f"claims must be numbered 1..{len(claims)} with no gaps or duplicates, "
            f"got {numbers}",
            actor=actor,
        )
    return tuple(sorted(claims, key=lambda c: c.number))


def _sentence_count(text: str) -> int:
    return len(_SENTENCE_BREAK.split(text.strip()))


def _compliance_warnings(position: str, claims: tuple[Claim, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    if _sentence_count(position) > 1:
        warnings.append("position is more than one sentence")
    if len(position) > MAX_SENTENCE_CHARS:
        warnings.append(f"position exceeds {MAX_SENTENCE_CHARS} characters")
    for claim in claims:
        if _sentence_count(claim.text) > 1:
            warnings.append(f"claim {claim.number} is more than one sentence")
        if len(claim.text) > MAX_SENTENCE_CHARS:
            warnings.append(
                f"claim {claim.number} exceeds {MAX_SENTENCE_CHARS} characters"
            )
    return tuple(warnings)


_SHEET_FIELDS = {
    "position",
    "claims",
    "assumptions",
    "would_change_my_mind",
    "confidence",
    "nuance",
    "compliance_warnings",
}

_REVISION_FIELDS = {"changed_position", "because"}


def parse_sheet(
    response: Any, *, actor: str = "", allow_revision_fields: bool = False
) -> AnswerSheet:
    """Parse an answer sheet. Raises `SheetError` on any violation; the caller
    marks the student absent rather than repairing it.

    `allow_revision_fields` is off by default so that a *round-1* sheet
    carrying `changed_position` is an error rather than a field quietly
    ignored. A student that reports changing its mind before it has seen
    anyone else's answer has misunderstood the round it is in, and the sheet's
    other contents are then equally suspect.
    """
    data = response if isinstance(response, dict) else extract_json(response, actor=actor)
    if not isinstance(data, dict):
        raise SheetSchemaError("an answer sheet must be a JSON object", actor=actor)

    permitted = _SHEET_FIELDS | (_REVISION_FIELDS if allow_revision_fields else set())
    unknown = set(data) - permitted
    if unknown:
        # Extra keys mean the model answered a different question than the one
        # the template asked, and the fields we do read may be equally
        # improvised. Loud beats lenient here.
        raise SheetSchemaError(
            f"unexpected fields on answer sheet: {sorted(unknown)}", actor=actor
        )

    position = _require_text(data.get("position"), "position", actor)
    claims = _parse_claims(data.get("claims"), actor)
    assumptions = _require_str_list(data.get("assumptions"), "assumptions", actor)
    falsifiers = _require_str_list(
        data.get("would_change_my_mind"), "would_change_my_mind", actor
    )
    confidence = _parse_confidence(data.get("confidence"), actor)

    nuance = data.get("nuance", "")
    if nuance is None:
        nuance = ""
    if not isinstance(nuance, str):
        raise SheetSchemaError("nuance must be a string", actor=actor)

    return AnswerSheet(
        position=position,
        claims=claims,
        assumptions=assumptions,
        would_change_my_mind=falsifiers,
        confidence=confidence,
        nuance=nuance.strip(),
        compliance_warnings=_compliance_warnings(position, claims),
    )


# --------------------------------------------------------------------------
# critiques
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawObjection:
    """An objection as the critic wrote it, against a *blinded* sheet label."""

    sheet: str
    claim_n: int
    argument: str


def parse_critique(
    response: Any,
    *,
    allowed: dict[str, tuple[int, ...]],
    actor: str = "",
) -> tuple[RawObjection, ...]:
    """Parse a round-2 critique against the labels the critic was shown.

    `allowed` maps each blinded sheet label the critic received to the claim
    numbers that exist on it, so a hallucinated claim 7 is caught here rather
    than silently dropping out of the objection count. The format requires at
    least one objection per foreign sheet: "engage every other position" is a
    protocol invariant, not a suggestion, and a critique that skips a sheet is
    non-compliant even if its other objections are excellent.
    """
    data = response if isinstance(response, dict) else extract_json(response, actor=actor)
    if isinstance(data, dict):
        raw = data.get("objections", data.get("critique"))
    else:
        raw = data
    if not isinstance(raw, list):
        raise SheetSchemaError(
            "a critique must carry an 'objections' list", actor=actor
        )
    if not raw:
        raise NonCompliantCritique("critique contains no objections", actor=actor)

    objections: list[RawObjection] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SheetSchemaError(
                f"objections[{index}] must be an object with sheet, claim_n, argument",
                actor=actor,
            )
        sheet = item.get("sheet", item.get("sheet_label"))
        if not isinstance(sheet, str) or not sheet.strip():
            raise NonCompliantCritique(
                f"objections[{index}] names no sheet; vague agreement is not a critique",
                actor=actor,
            )
        sheet = sheet.strip().upper().removeprefix("SHEET").strip()
        if sheet not in allowed:
            raise SheetSchemaError(
                f"objections[{index}] references unknown sheet {sheet!r}; "
                f"this critic was shown {sorted(allowed)}",
                actor=actor,
            )

        claim_value = item.get("claim_n", item.get("claim"))
        if isinstance(claim_value, bool) or not isinstance(claim_value, (int, str)):
            raise NonCompliantCritique(
                f"objections[{index}] references no claim number", actor=actor
            )
        try:
            claim_n = int(claim_value)
        except ValueError:
            raise SheetSchemaError(
                f"objections[{index}] claim_n is not an integer: {claim_value!r}",
                actor=actor,
            ) from None
        if claim_n not in allowed[sheet]:
            raise SheetSchemaError(
                f"objections[{index}] references claim {claim_n} on sheet {sheet}, "
                f"which has claims {list(allowed[sheet])}",
                actor=actor,
            )

        argument = item.get("argument", item.get("objection"))
        argument = _require_text(argument, f"objections[{index}].argument", actor)
        if _VAGUE_AGREEMENT.match(argument):
            raise NonCompliantCritique(
                f"objections[{index}] is agreement, not an objection: {argument!r}",
                actor=actor,
            )
        if len(argument) < MIN_ARGUMENT_CHARS:
            raise NonCompliantCritique(
                f"objections[{index}] argument is {len(argument)} characters; "
                f"at least {MIN_ARGUMENT_CHARS} are required to count as engagement",
                actor=actor,
            )
        objections.append(RawObjection(sheet=sheet, claim_n=claim_n, argument=argument))

    covered = {o.sheet for o in objections}
    missing = sorted(set(allowed) - covered)
    if missing:
        raise NonCompliantCritique(
            f"no objection raised against sheet(s) {missing}; "
            "the format requires the strongest objection to every other sheet",
            actor=actor,
        )
    return tuple(objections)


# --------------------------------------------------------------------------
# revisions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectionRef:
    """A revision's citation of the objection that moved it.

    Critics are blinded in round 3 too — a student is told *what* was argued
    against claim 2, not who argued it — so the reference names a critic label
    from that round's mapping, never a seat or a model.
    """

    critic: str
    claim_n: int


@dataclass(frozen=True)
class Revision:
    sheet: AnswerSheet
    changed_position: bool
    because: tuple[ObjectionRef, ...]


def parse_revision(
    response: Any,
    *,
    allowed: dict[str, tuple[int, ...]],
    actor: str = "",
) -> Revision:
    """Parse a round-3 revision: a full replacement sheet plus its own account
    of what moved it. `allowed` maps critic labels to the claim numbers they
    objected to, so `because` cannot cite an objection nobody raised."""
    data = response if isinstance(response, dict) else extract_json(response, actor=actor)
    if not isinstance(data, dict):
        raise SheetSchemaError("a revision must be a JSON object", actor=actor)

    sheet = parse_sheet(data, actor=actor, allow_revision_fields=True)

    declared = data.get("changed_position", False)
    if isinstance(declared, str):
        declared = declared.strip().lower() in {"true", "yes", "1"}
    if not isinstance(declared, bool):
        raise SheetSchemaError("changed_position must be a boolean", actor=actor)

    raw_because = data.get("because", [])
    if raw_because is None:
        raw_because = []
    if isinstance(raw_because, dict):
        raw_because = [raw_because]
    if not isinstance(raw_because, list):
        raise SheetSchemaError("because must be a list of objection references", actor=actor)

    refs: list[ObjectionRef] = []
    for index, item in enumerate(raw_because, start=1):
        if not isinstance(item, dict):
            raise SheetSchemaError(
                f"because[{index}] must be an object with critic and claim_n",
                actor=actor,
            )
        critic = item.get("critic", item.get("sheet"))
        if not isinstance(critic, str) or not critic.strip():
            raise SheetSchemaError(f"because[{index}] names no critic", actor=actor)
        critic = critic.strip().upper().removeprefix("CRITIC").removeprefix("SHEET").strip()
        claim_value = item.get("claim_n", item.get("claim"))
        if isinstance(claim_value, bool) or not isinstance(claim_value, (int, str)):
            raise SheetSchemaError(
                f"because[{index}] names no claim number", actor=actor
            )
        try:
            claim_n = int(claim_value)
        except ValueError:
            raise SheetSchemaError(
                f"because[{index}] claim_n is not an integer: {claim_value!r}", actor=actor
            ) from None
        if critic not in allowed or claim_n not in allowed[critic]:
            raise SheetSchemaError(
                f"because[{index}] cites critic {critic!r} on claim {claim_n}, "
                f"which raised no such objection (available: "
                f"{ {k: list(v) for k, v in allowed.items()} })",
                actor=actor,
            )
        refs.append(ObjectionRef(critic=critic, claim_n=claim_n))

    return Revision(sheet=sheet, changed_position=declared, because=tuple(refs))


# --------------------------------------------------------------------------
# structural diffing
# --------------------------------------------------------------------------

_EDIT_SIMILARITY = 0.6
"""Above this, two claims are the same claim reworded; below it, one was
dropped and another added. The threshold is a judgement call and it is
recorded on the diff so a report can show its work."""


@dataclass(frozen=True)
class ClaimEdit:
    before: Claim
    after: Claim
    similarity: float


@dataclass(frozen=True)
class SheetDiff:
    """What actually changed between a student's round-1 and round-3 sheets.

    Computed by us, never taken on the model's word. `declaration_matches_diff`
    is the interesting field: a model that says it changed its mind while its
    position is byte-identical is a sycophancy signal, and one that quietly
    rewrites its position while reporting no change is the opposite. Both are
    invisible unless the diff is mechanical."""

    position_changed: bool
    position_similarity: float
    claims_added: tuple[Claim, ...]
    claims_dropped: tuple[Claim, ...]
    claims_edited: tuple[ClaimEdit, ...]
    confidence_delta: float
    declared_change: bool
    declaration_matches_diff: bool
    edit_threshold: float = _EDIT_SIMILARITY

    @property
    def changed(self) -> bool:
        return bool(
            self.position_changed
            or self.claims_added
            or self.claims_dropped
            or self.claims_edited
            or abs(self.confidence_delta) > 1e-9
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_changed": self.position_changed,
            "position_similarity": round(self.position_similarity, 4),
            "claims_added": [{"n": c.number, "text": c.text} for c in self.claims_added],
            "claims_dropped": [
                {"n": c.number, "text": c.text} for c in self.claims_dropped
            ],
            "claims_edited": [
                {
                    "before": {"n": e.before.number, "text": e.before.text},
                    "after": {"n": e.after.number, "text": e.after.text},
                    "similarity": round(e.similarity, 4),
                }
                for e in self.claims_edited
            ],
            "confidence_delta": round(self.confidence_delta, 4),
            "declared_change": self.declared_change,
            "declaration_matches_diff": self.declaration_matches_diff,
            "edit_threshold": self.edit_threshold,
            "changed": self.changed,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def diff_sheets(
    before: AnswerSheet, after: AnswerSheet, *, declared_change: bool = False
) -> SheetDiff:
    """Diff two sheets by claim *content*, not claim number.

    Matching on number would report a wholesale rewrite as "claims 1-3 edited"
    whenever a model renumbers after dropping one, which is noise. Greedy
    best-match on text similarity survives renumbering, which is the common
    case after a claim is withdrawn."""
    remaining = list(after.claims)
    edited: list[ClaimEdit] = []
    dropped: list[Claim] = []

    for old in before.claims:
        best: Claim | None = None
        best_score = 0.0
        for candidate in remaining:
            score = _similarity(old.text, candidate.text)
            if score > best_score:
                best, best_score = candidate, score
        if best is not None and best_score >= _EDIT_SIMILARITY:
            remaining.remove(best)
            if _normalize(old.text) != _normalize(best.text):
                edited.append(ClaimEdit(before=old, after=best, similarity=best_score))
        else:
            dropped.append(old)

    position_similarity = _similarity(before.position, after.position)
    position_changed = _normalize(before.position) != _normalize(after.position)

    return SheetDiff(
        position_changed=position_changed,
        position_similarity=position_similarity,
        claims_added=tuple(remaining),
        claims_dropped=tuple(dropped),
        claims_edited=tuple(edited),
        confidence_delta=after.confidence - before.confidence,
        declared_change=declared_change,
        # Compared against the *position* diff, not against any change at all.
        #
        # `changed_position` is defined for the model as "true only if your
        # one-sentence position now says something different", so comparing it
        # to a broader notion of change measures the model against a question
        # it was never asked. A real revision made that concrete: a model held
        # its position word for word, correctly declared `false`, and rewrote
        # four of its five claims in response to objections — the healthiest
        # possible outcome — and the flag called it a discrepancy.
        #
        # The flag exists to catch two specific pathologies: claiming to have
        # reconsidered while resubmitting the same sentence, and rewriting the
        # position while reporting no change. Both are about the position.
        declaration_matches_diff=(declared_change == position_changed),
    )


# --------------------------------------------------------------------------
# the arbiter's verdict
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MinorityItem:
    """A dissent that did not make the final answer, preserved in substance.

    `source` is a seat label ("Student 2"), resolved to a model by the session
    on the way into the trace. The arbiter never sees model names — see
    `prompts.build_verdict_prompt` for why."""

    source: str
    substance: str
    kind: str = "claim"


@dataclass(frozen=True)
class Verdict:
    final_answer: str
    confidence_note: str
    minority_report: tuple[MinorityItem, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "confidence_note": self.confidence_note,
            "minority_report": [
                {"source": m.source, "kind": m.kind, "substance": m.substance}
                for m in self.minority_report
            ],
        }


def parse_verdict(
    response: Any, *, allowed_sources: tuple[str, ...], actor: str = "arbiter"
) -> Verdict:
    """Parse the arbiter's output.

    An empty minority report is legal — genuine consensus exists — but an
    *unresolvable* one is not: a dissent attributed to nobody cannot be
    checked against the transcript, and an unverifiable attribution is worse
    than none.
    """
    data = response if isinstance(response, dict) else extract_json(response, actor=actor)
    if not isinstance(data, dict):
        raise SheetSchemaError("a verdict must be a JSON object", actor=actor)

    final_answer = _require_text(data.get("final_answer"), "final_answer", actor)
    confidence_note = _require_text(data.get("confidence_note"), "confidence_note", actor)

    raw = data.get("minority_report", [])
    if raw is None:
        raw = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise SheetSchemaError("minority_report must be a list", actor=actor)

    normalized_allowed = {s.strip().lower(): s for s in allowed_sources}
    items: list[MinorityItem] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise SheetSchemaError(
                f"minority_report[{index}] must be an object", actor=actor
            )
        source = _require_text(
            item.get("source", item.get("student")), f"minority_report[{index}].source", actor
        )
        key = source.strip().lower()
        if key not in normalized_allowed:
            raise SheetSchemaError(
                f"minority_report[{index}] attributed to unknown source {source!r}; "
                f"participants were {list(allowed_sources)}",
                actor=actor,
            )
        substance = _require_text(
            item.get("substance", item.get("text")),
            f"minority_report[{index}].substance",
            actor,
        )
        kind = item.get("kind", "claim")
        if not isinstance(kind, str) or kind.strip().lower() not in {"claim", "objection"}:
            raise SheetSchemaError(
                f"minority_report[{index}].kind must be 'claim' or 'objection'",
                actor=actor,
            )
        items.append(
            MinorityItem(
                source=normalized_allowed[key],
                substance=substance,
                kind=kind.strip().lower(),
            )
        )

    return Verdict(
        final_answer=final_answer,
        confidence_note=confidence_note,
        minority_report=tuple(items),
    )
