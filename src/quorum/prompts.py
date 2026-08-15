"""Every word Quorum puts on the wire, in one file.

Prompts live together for the same reason Switchboard's do: they are the part
of the system most likely to be tweaked and least likely to be tested, and
scattering them across the engine makes it impossible to answer "what exactly
did the student see?" without reading the whole call graph. The session engine
below assembles rounds; it never writes prose.

The four `*_HEADER` constants are load-bearing. They are real lines of the
real prompts, and the offline `MockProvider` keys off them to decide which
round it is being asked to play. A test-only marker would drift from the
prompt it stands in for; a line the model actually reads cannot.
"""

from __future__ import annotations

from .sheets import MAX_CLAIMS, AnswerSheet

SHEET_PROMPT_HEADER = "You are sitting a silent exam. Fill in the answer sheet below."
CRITIQUE_PROMPT_HEADER = "Below are answer sheets from other participants, anonymised."
REVISION_PROMPT_HEADER = "Objections were raised against your answer sheet."
VERDICT_PROMPT_HEADER = "You are the arbiter. You did not take part in the debate."

_JSON_RULE = (
    "Reply with a single JSON object and nothing else. No commentary before or after."
)


# --------------------------------------------------------------------------
# round 1 — the silent exam
# --------------------------------------------------------------------------

SHEET_PROMPT_TEMPLATE = f"""{SHEET_PROMPT_HEADER}

You are answering independently. You have not seen anyone else's answer and
nobody has seen yours. Do not hedge toward what you imagine a consensus would
be; state what you actually think.

QUESTION
{{task}}

Fill in exactly these fields:

- "position": one sentence stating your answer.
- "claims": at most {MAX_CLAIMS} numbered claims, one sentence each, that carry
  your position. Each must be something another participant could specifically
  disagree with.
- "assumptions": what your position depends on being true.
- "would_change_my_mind": concrete findings that would falsify your position.
- "confidence": a number between 0 and 1.
- "nuance": optional free text for anything the fields above flatten. This
  field is NOT shown to other participants and cannot be objected to, so put
  nothing load-bearing in it.

{_JSON_RULE}

{{{{"position": "...", "claims": [{{{{"n": 1, "text": "..."}}}}],
  "assumptions": ["..."], "would_change_my_mind": ["..."],
  "confidence": 0.0, "nuance": ""}}}}
"""


def build_sheet_prompt(task: str) -> str:
    return SHEET_PROMPT_TEMPLATE.format(task=task)


# --------------------------------------------------------------------------
# rendering a sheet for someone else's eyes
# --------------------------------------------------------------------------


def render_sheet(sheet: AnswerSheet, label: str, *, include_nuance: bool = False) -> str:
    """Render a sheet under an anonymous label.

    `nuance` is off by default, and the default is what the critique rounds
    use. It is the field that exists precisely because it holds the prose the
    schema could not carry — which makes it the field most likely to carry a
    model's fingerprint, and the one thing in the protocol that cannot be
    objected to. Showing it to a critic would trade the blinding away for
    material nobody is allowed to argue with.

    The arbiter is the one reader that does get it (`include_nuance=True`).
    The spec excludes nuance from the *critique* rounds; excluding it from
    grading as well would make the field write-only — a documented escape
    valve for positions the five-claim cap flattens, which then cannot reach
    the answer. The cost is real and worth stating: nuance is unblinded free
    text, so it is the one channel through which a student could signal its
    identity to the arbiter.
    """
    lines = [f"--- Sheet {label} ---", f"position: {sheet.position}", "claims:"]
    lines += [f"  {c.number}. {c.text}" for c in sheet.claims]
    lines.append("assumptions:")
    lines += [f"  - {a}" for a in sheet.assumptions]
    lines.append("would_change_my_mind:")
    lines += [f"  - {w}" for w in sheet.would_change_my_mind]
    lines.append(f"confidence: {sheet.confidence:.2f}")
    if include_nuance and sheet.nuance:
        lines.append(f"nuance (not critiqued by anyone): {sheet.nuance}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# round 2 — blind claim-level critique
# --------------------------------------------------------------------------

CRITIQUE_PROMPT_TEMPLATE = f"""{CRITIQUE_PROMPT_HEADER}

You do not know who wrote them and they do not know who you are. Judge the
arguments, not the source.

QUESTION UNDER DISCUSSION
{{task}}

{{sheets}}

Raise your strongest objection to EVERY sheet above — at least one objection
per sheet, more if you have them. Each objection must name the sheet and the
specific claim number it attacks, and must say why that claim is wrong,
unsupported, or load-bearing in a way its author has not noticed.

You may not register agreement. There is no field for it. If you think a claim
is correct, attack its weakest supporting assumption instead.

{_JSON_RULE}

{{{{"objections": [{{{{"sheet": "A", "claim_n": 1, "argument": "..."}}}}]}}}}
"""


def build_critique_prompt(task: str, blinded: dict[str, AnswerSheet]) -> str:
    sheets = "\n\n".join(
        render_sheet(sheet, label) for label, sheet in sorted(blinded.items())
    )
    return CRITIQUE_PROMPT_TEMPLATE.format(task=task, sheets=sheets)


SKEPTIC_INSTRUCTION = """
YOU HOLD THE SKEPTIC SEAT THIS ROUND.

Sheet {label} is the most confident position in front of you. Attack it
hardest, and attack it *whether or not you agree with it*. If you find you
agree, that is the signal to look harder: a confident position that nobody
pressed is a position nobody tested.

This does not excuse you from the other sheets. Every sheet still needs at
least one objection.
"""


def build_skeptic_critique_prompt(
    task: str, blinded: dict[str, AnswerSheet], target_label: str
) -> str:
    """The critique prompt plus a standing instruction to attack the leader.

    The seat exists because independence stops sycophancy between *rounds* but
    not the quieter version within one: three models can converge on the most
    confidently-stated position without any of them having tested it. One
    student assigned to press the leader regardless of its own view puts a
    floor under how much scrutiny the strongest sheet gets.

    The target is chosen by stated confidence, which is visible on every sheet
    the critic already holds — so the instruction adds pressure without adding
    a single bit of identity information. Whether it helps is an empirical
    question: the effect on position-change rate is what settles it.
    """
    base = build_critique_prompt(task, blinded)
    return base + SKEPTIC_INSTRUCTION.format(label=target_label)


def leading_sheet_label(blinded: dict[str, AnswerSheet]) -> str:
    """The most confident sheet on offer; ties broken by label for determinism."""
    return min(blinded.items(), key=lambda kv: (-kv[1].confidence, kv[0]))[0]


CRITIQUE_REPAIR_TEMPLATE = """Your critique was rejected: {reason}

Try once more. Every objection must name a sheet, name a claim number that
exists on that sheet, and argue against it in at least {min_chars} characters.
Every sheet you were shown must receive at least one objection.

{original}
"""


def build_critique_repair_prompt(original: str, reason: str, min_chars: int) -> str:
    """One re-prompt, then the critic is recorded non-compliant.

    Retrying forever would let a model that cannot follow the format burn the
    session's budget; not retrying at all would discard a good critic over a
    missing field. Once is the compromise, and the outcome is recorded either
    way."""
    return CRITIQUE_REPAIR_TEMPLATE.format(
        reason=reason, min_chars=min_chars, original=original
    )


# --------------------------------------------------------------------------
# round 3 — revision
# --------------------------------------------------------------------------

REVISION_PROMPT_TEMPLATE = f"""{REVISION_PROMPT_HEADER}

The critics are anonymous and are labelled differently from the sheets you
read earlier. You cannot tell which critic wrote which sheet, and you should
not try.

QUESTION UNDER DISCUSSION
{{task}}

YOUR CURRENT SHEET (JSON)
```json
{{sheet_json}}
```

OBJECTIONS RAISED AGAINST YOU
{{objections}}

Submit your sheet again, revised or unchanged. Changing your mind is not a
loss and holding your position under pressure is not stubbornness — but an
objection you cannot answer should move something.

Add two fields to the sheet:

- "changed_position": true only if your one-sentence position now says
  something different.
- "because": the objections that moved you, as
  [{{{{"critic": "A", "claim_n": 1}}}}]. Leave it empty if nothing moved you.

{_JSON_RULE}
"""

_NO_OBJECTIONS = (
    "(none — no critic engaged your sheet this round; your position stands "
    "untested, which is not the same as unopposed)"
)


def build_revision_prompt(
    task: str, sheet_json: str, objections: list[tuple[str, int, str]]
) -> str:
    """`objections` are (critic_label, claim_n, argument), already blinded."""
    if objections:
        blocks = [
            f"--- Critic {critic} on your claim {claim_n} ---\n{argument}"
            for critic, claim_n, argument in objections
        ]
        rendered = "\n\n".join(blocks)
    else:
        rendered = _NO_OBJECTIONS
    return REVISION_PROMPT_TEMPLATE.format(
        task=task, sheet_json=sheet_json, objections=rendered
    )


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------

VERDICT_PROMPT_TEMPLATE = f"""{VERDICT_PROMPT_HEADER}

You hold no position of your own to defend. Your job is to synthesise, and to
make sure nothing that deserved to survive was lost.

QUESTION
{{task}}

{{transcript}}

Produce:

- "final_answer": the answer the transcript supports, in prose a
  non-specialist can act on.
- "confidence_note": how settled this is, and what it turns on. Say plainly
  if the participants did not converge; manufactured consensus is worse than
  reported disagreement.
- "minority_report": every claim or objection that did NOT make your final
  answer but that a careful reader would want to know was raised. Attribute
  each to the participant label it came from and preserve its substance —
  do not soften it into agreement. An empty list is legitimate only if
  nothing was genuinely left out.

{_JSON_RULE}

{{{{"final_answer": "...", "confidence_note": "...",
  "minority_report": [{{{{"source": "Student 1", "kind": "claim",
  "substance": "..."}}}}]}}}}
"""


VERDICT_REPAIR_TEMPLATE = """Your verdict was rejected: {reason}

Try once more, in the exact shape asked for. Every minority_report entry must
be attributed to one of: {sources}.

{original}
"""


def build_verdict_repair_prompt(original: str, reason: str, sources: tuple[str, ...]) -> str:
    """One re-prompt for the arbiter, then the session closes without a verdict.

    A session that reaches the grading round has already paid for three sheets
    and six objections; discarding all of it because the arbiter returned a
    stray sentence would be wasteful. Discarding the *verdict* when it cannot
    be parsed is not — an improvised synthesis is exactly the fluent,
    unauditable output the project exists to replace."""
    return VERDICT_REPAIR_TEMPLATE.format(
        reason=reason, sources=", ".join(sources), original=original
    )


def build_verdict_prompt(task: str, transcript: str) -> str:
    """Assemble the arbiter's briefing.

    The arbiter sees seat labels ("Student 1"), never model names. It is not a
    participant, so blinding is not about sycophancy here — it is about brand
    priors. An arbiter told that one sheet came from the largest frontier model
    in the lineup has a reason to weight it that has nothing to do with the
    argument on the page. The session maps seats back to models afterwards, so
    the minority report is still attributable in the trace and the report.
    """
    return VERDICT_PROMPT_TEMPLATE.format(task=task, transcript=transcript)


# --------------------------------------------------------------------------
# the deanonymization probe
# --------------------------------------------------------------------------

PROBE_PROMPT_HEADER = "You are auditing an anonymised deliberation, not taking part in one."

PROBE_PROMPT_TEMPLATE = f"""{PROBE_PROMPT_HEADER}

Below are answer sheets written independently by different AI models. Each
model wrote exactly one sheet. Your job is to work out which model wrote which.

Use anything you can: argument style, hedging, how risk is framed, vocabulary,
how the claims are structured, what the model chose to treat as obvious. You
are being scored against chance, so guess even when you are unsure — an
abstention is not a safe answer here, it is a missing data point.

QUESTION THEY WERE ANSWERING
{{task}}

{{sheets}}

THE MODELS THAT TOOK PART (one sheet each)
{{roster}}

{_JSON_RULE}

{{{{"guesses": {{{{"A": "model-name", "B": "model-name"}}}}}}}}
"""


def build_probe_prompt(task: str, sheets: str, roster: tuple[str, ...]) -> str:
    """Brief the prober, generously.

    It gets the candidate roster and the fact that authorship is one-to-one.
    Withholding either would make the measured leak smaller than the real one,
    and this is the one number in the project that must not be flattered — its
    whole purpose is to be published even when it is bad news.
    """
    return PROBE_PROMPT_TEMPLATE.format(
        task=task, sheets=sheets, roster="\n".join(f"  - {m}" for m in roster)
    )
