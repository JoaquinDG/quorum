"""The Session Report: the artifact a non-technical reader can actually use.

This module is a *player* for the trace, in the sense `trace` sets out. It
takes a `ReplayedSession` — rebuilt from the file, not handed the live session
object — so the claim "any future UI is just a renderer" is demonstrated by
the first UI rather than asserted about later ones. Nothing here reaches into
the engine, and nothing here re-derives what the record failed to keep; if a
panel below needs a fact, that fact is in the JSONL.

Two output formats, one content model. The HTML is a single self-contained
file with no external CSS, fonts, scripts or images — a report you cannot
email is not a report — and it is theme-aware, because half of the people who
open it will be in dark mode. The Markdown fallback exists for terminals,
pull-request comments and anywhere a browser is the wrong tool.

The presentation is the theatrical one the spec files under P1 (avatars, a
vertical debate timeline, objections as speech bubbles, a visible
"changed position" beat, the minority report as a closing panel) because
building the plain version first and the theatrical version second would mean
writing the same document twice. It is framed as deliberation replay — a
transparency tool — not as a game: every visual element maps to a fact in the
trace, and nothing is dramatised that did not happen.
"""

from __future__ import annotations

import hashlib
import html
import os
from dataclasses import dataclass
from typing import Any

from .replay import ReplayedObjection, ReplayedSession, replay

# --------------------------------------------------------------------------
# picking what to show
# --------------------------------------------------------------------------

_CITED = 3
_DROPPED = 2
_EDITED = 1
_MOVED = 1


@dataclass(frozen=True)
class RankedObjection:
    objection: ReplayedObjection
    score: int
    effect: str
    """Plain-language account of what this objection actually did."""


def rank_objections(session: ReplayedSession) -> list[RankedObjection]:
    """Order objections by what they *did*, not by how they read.

    "Sharpest" has an obvious lazy proxy — length, or the model's own
    self-report — and both measure rhetoric. The trace already records the
    consequences: whether the target cited this objection as the reason it
    moved, and whether the claim under attack was later withdrawn or reworded.
    An objection that made someone drop a claim is sharp in the only sense the
    reader cares about, and it is a fact rather than an impression.
    """
    ranked: list[RankedObjection] = []
    for objection in session.objections:
        target = session.students.get(objection.target_seat)
        score = 0
        effects: list[str] = []
        if target is not None:
            cited = any(
                ref.get("critic_seat") == objection.critic_seat
                and ref.get("claim_n") == objection.claim_n
                for ref in target.because
            )
            if cited:
                score += _CITED
                effects.append("cited as the reason for a revision")
            diff = target.diff or {}
            if any(c.get("n") == objection.claim_n for c in diff.get("claims_dropped", [])):
                score += _DROPPED
                effects.append("the claim was withdrawn")
            elif any(
                e.get("before", {}).get("n") == objection.claim_n
                for e in diff.get("claims_edited", [])
            ):
                score += _EDITED
                effects.append("the claim was reworded")
            if target.changed_position and cited:
                score += _MOVED
                effects.append("the position changed")
        ranked.append(
            RankedObjection(
                objection=objection,
                score=score,
                effect="; ".join(effects) or "no recorded effect on the target's sheet",
            )
        )
    ranked.sort(key=lambda r: (-r.score, -len(r.objection.argument)))
    return ranked


def _avatar(model_id: str) -> tuple[str, int]:
    """Initials and a hue, derived from the model id. No external images."""
    letters = [c for c in model_id if c.isalnum()]
    initials = "".join(letters[:2]).upper() if letters else "??"
    hue = hashlib.blake2b(model_id.encode("utf-8"), digest_size=2).digest()[0]
    return initials, int(hue / 255 * 360)


def _headline(session: ReplayedSession) -> str:
    """One sentence a reader can take away without reading anything else."""
    if not session.verdict:
        return "This session did not reach a verdict."
    movers = sum(1 for s in session.present_students if s.changed_position)
    total = len([s for s in session.present_students if s.diff is not None])
    if session.dissent_preserved and movers == 0:
        return "The council did not converge, and nobody moved under objection."
    if session.dissent_preserved:
        return (
            f"The council reached a qualified answer; {movers} of {total} participants "
            "changed position, and dissent remains on the record."
        )
    if movers == 0:
        return (
            "The council agreed, but nobody changed position — treat the agreement "
            "as three independent priors, not as a resolved argument."
        )
    return f"The council converged after {movers} of {total} participants moved."


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def render_markdown(session: ReplayedSession, *, top_objections: int = 3) -> str:
    out: list[str] = [
        "# Quorum session report",
        "",
        f"> {_headline(session)}",
        "",
        f"**Question.** {session.task}",
        "",
    ]

    if session.reduced_council:
        out += [
            f"> ⚠️ **Reduced council.** {session.council_size} of "
            f"{len(session.students)} seated participants answered. Read the "
            "verdict as the product of a smaller council.",
            "",
        ]

    for warning in session.council_warnings:
        out += [f"> ⚠️ **Single-lab council.** {warning.capitalize()}.", ""]

    out += ["## The answer", ""]
    if session.verdict:
        out += [session.verdict.final_answer, "", f"*{session.verdict.confidence_note}*", ""]
    else:
        out += [f"**No verdict.** {session.failed_reason or 'unknown'}", ""]

    out += ["## Where they started", ""]
    for student in session.present_students:
        out.append(f"**{student.label}** ({student.model_id}) — confidence "
                   f"{student.initial.confidence:.0%}")
        out.append(f"> {student.initial.position}")
        out += [f"> {c.number}. {c.text}" for c in student.initial.claims]
        out.append("")

    out += ["## The sharpest objections", ""]
    for ranked in rank_objections(session)[:top_objections]:
        objection = ranked.objection
        critic = session.students[objection.critic_seat].label
        target = session.students[objection.target_seat].label
        out.append(f"**{critic} → {target}, claim {objection.claim_n}**")
        out.append(f"> *{objection.claim_text}*")
        out.append("")
        out.append(objection.argument)
        out.append("")
        out.append(f"*Effect: {ranked.effect}.*")
        out.append("")

    out += ["## Who changed their mind", ""]
    moved = [s for s in session.present_students if s.changed_position]
    if not moved:
        out.append("Nobody changed position. Every participant held its opening "
                   "answer through the critique round.")
        out.append("")
    for student in moved:
        out.append(f"**{student.label}** ({student.model_id})")
        out.append(f"- from: {student.position_change['from']}")
        out.append(f"- to: {student.position_change['to']}")
        for dropped in (student.diff or {}).get("claims_dropped", []):
            out.append(f"- withdrew: {dropped['text']}")
        delta = (student.diff or {}).get("confidence_delta", 0.0)
        if abs(delta) > 1e-9:
            out.append(f"- confidence {delta:+.0%}")
        out.append("")

    held = [s for s in session.present_students if s.diff is not None and not s.changed_position]
    if held and moved:
        out.append("Held position: " + ", ".join(f"{s.label} ({s.model_id})" for s in held))
        out.append("")

    out += ["## Minority report", ""]
    if session.verdict and session.verdict.minority_report:
        out.append("Raised in the session, and *not* reflected in the final answer:")
        out.append("")
        for item in session.verdict.minority_report:
            model = next(
                (m.get("source_model", "") for m in session.minority
                 if m.get("source") == item.source),
                "",
            )
            who = f"{item.source}" + (f", {model}" if model else "")
            out.append(f"- **[{who} — {item.kind}]** {item.substance}")
        out.append("")
    else:
        out += ["Nothing was recorded as left out.", ""]

    out += ["## The numbers", "", "| | |", "| --- | --- |"]
    out.append(f"| Council | {session.council_size} of {len(session.students)} answered |")
    out.append(f"| Objections | {len(session.objections)} |")
    if session.disagreement:
        out.append(f"| Opening wording spread | {session.disagreement['score']:.2f} — "
                   f"{session.disagreement['label']} |")
        out.append("| | *vocabulary only — this does **not** measure agreement* |")
    out.append(f"| Position-change rate | {session.position_change_rate:.0%} "
               "(0% = theatre, 100% = herding) |")
    out.append(f"| Dissent preserved | {'yes' if session.dissent_preserved else 'no'} |")
    out.append(f"| Claim compliance | {session.compliance_rate:.0%} |")
    out.append(f"| Session cost | ${session.cost_est:.4f}"
               + ("" if session.cost_is_complete else " (lower bound — unpriced seats)")
               + " |")
    if session.baseline_model:
        out.append(f"| One model, once ({session.baseline_model}) | "
                   f"${session.baseline_cost_est:.4f} |")
    if session.cost_multiple is not None:
        out.append(f"| Multiple | {session.cost_multiple:.1f}x a single answer |")
    if session.discarded:
        out.append(f"| Re-prompts | {len(session.discarded)} "
                   f"(${session.repair_cost_est:.4f}) |")
    out += ["", f"Session `{session.session_id}`. Rebuilt from the trace; "
            "nothing here was passed in from the engine.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a1a; --muted: #63625f;
  --line: #e3e0da; --accent: #7a5cff; --warn-bg: #fff4e0; --warn-ink: #7a4d00;
  --quote: #f4f2ee; --held: #6b7280; --moved: #16704f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14151a; --panel: #1c1e25; --ink: #ecebe8; --muted: #a3a19c;
    --line: #2e313a; --accent: #a48dff; --warn-bg: #3a2c12; --warn-ink: #f0c479;
    --quote: #23262e; --held: #9aa0ac; --moved: #5fd0a4;
  }
}
:root[data-theme="dark"] {
  --bg: #14151a; --panel: #1c1e25; --ink: #ecebe8; --muted: #a3a19c;
  --line: #2e313a; --accent: #a48dff; --warn-bg: #3a2c12; --warn-ink: #f0c479;
  --quote: #23262e; --held: #9aa0ac; --moved: #5fd0a4;
}
:root[data-theme="light"] {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a1a; --muted: #63625f;
  --line: #e3e0da; --accent: #7a5cff; --warn-bg: #fff4e0; --warn-ink: #7a4d00;
  --quote: #f4f2ee; --held: #6b7280; --moved: #16704f;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1rem 5rem; background: var(--bg); color: var(--ink);
  font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 46rem; margin: 0 auto; }
h1 { font-size: 1.05rem; letter-spacing: .09em; text-transform: uppercase;
     color: var(--muted); font-weight: 600; margin: 0 0 .4rem; }
h2 { font-size: 1.35rem; margin: 3rem 0 1rem; letter-spacing: -.01em; }
.question { font-size: 1.6rem; line-height: 1.35; font-weight: 650;
            letter-spacing: -.02em; margin: 0 0 1.2rem; }
.headline { border-left: 3px solid var(--accent); padding: .1rem 0 .1rem 1rem;
            color: var(--muted); font-size: 1.05rem; margin-bottom: 2rem; }
.banner { background: var(--warn-bg); color: var(--warn-ink); padding: .8rem 1rem;
          border-radius: .55rem; font-size: .92rem; margin-bottom: 1.5rem; }
.panel { background: var(--panel); border: 1px solid var(--line);
         border-radius: .7rem; padding: 1.15rem 1.3rem; margin-bottom: 1rem; }
.answer { border-left: 3px solid var(--accent); font-size: 1.08rem; }
.answer .note { color: var(--muted); font-size: .95rem; margin-top: .8rem;
                font-style: italic; }
.who { display: flex; align-items: center; gap: .65rem; margin-bottom: .6rem; }
.avatar { width: 2.1rem; height: 2.1rem; border-radius: 50%; flex: 0 0 auto;
          display: grid; place-items: center; font-size: .78rem; font-weight: 700;
          letter-spacing: .02em; color: #fff; }
.name { font-weight: 650; }
.model { color: var(--muted); font-size: .85rem; font-variant-numeric: tabular-nums; }
.position { margin: 0 0 .7rem; font-size: 1.02rem; }
ol.claims { margin: 0; padding-left: 1.3rem; color: var(--muted); font-size: .95rem; }
ol.claims li { margin: .22rem 0; }
.timeline { position: relative; padding-left: 1.6rem; }
.timeline::before { content: ""; position: absolute; left: .42rem; top: .4rem;
                    bottom: .4rem; width: 2px; background: var(--line); }
.beat { position: relative; margin-bottom: 1.1rem; }
.beat::before { content: ""; position: absolute; left: -1.42rem; top: .95rem;
                width: .62rem; height: .62rem; border-radius: 50%;
                background: var(--line); }
.beat.moved::before { background: var(--moved); }
.bubble { background: var(--quote); border-radius: .7rem; padding: .9rem 1.05rem;
          font-size: .97rem; }
.bubble .target { color: var(--muted); font-size: .88rem; margin-bottom: .5rem; }
.bubble .claim { font-style: italic; }
.effect { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
.change { font-size: .97rem; }
.change .from { color: var(--muted); text-decoration: line-through;
                text-decoration-color: var(--line); }
.change .to { color: var(--moved); font-weight: 600; }
.change ul { margin: .5rem 0 0; padding-left: 1.1rem; color: var(--muted);
             font-size: .9rem; }
.held { color: var(--held); font-size: .93rem; }
.minority { border: 1px solid var(--accent); border-radius: .7rem;
            padding: 1.15rem 1.3rem; }
.minority li { margin-bottom: .8rem; }
.minority .src { color: var(--accent); font-weight: 650; font-size: .85rem;
                 letter-spacing: .03em; text-transform: uppercase; }
.grid { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .94rem; }
td { padding: .5rem .2rem; border-bottom: 1px solid var(--line); }
td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
.foot { color: var(--muted); font-size: .82rem; margin-top: 2.5rem;
        border-top: 1px solid var(--line); padding-top: 1rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; }
@media (max-width: 34rem) {
  body { padding: 1.2rem .9rem 3rem; }
  .question { font-size: 1.3rem; }
}
"""


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _avatar_html(model_id: str) -> str:
    initials, hue = _avatar(model_id)
    return (
        f'<div class="avatar" style="background:hsl({hue} 52% 42%)" '
        f'aria-hidden="true">{_e(initials)}</div>'
    )


def _who_html(label: str, model_id: str, extra: str = "") -> str:
    return (
        f'<div class="who">{_avatar_html(model_id)}'
        f'<div><div class="name">{_e(label)}</div>'
        f'<div class="model">{_e(model_id)}{_e(extra)}</div></div></div>'
    )


def render_html(session: ReplayedSession, *, top_objections: int = 3) -> str:
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Quorum session — {_e(session.task[:60])}</title>",
        f"<style>{_CSS}</style>",
        "</head><body><div class=\"wrap\">",
        "<h1>Quorum session report</h1>",
        f'<p class="question">{_e(session.task)}</p>',
        f'<p class="headline">{_e(_headline(session))}</p>',
    ]

    if session.reduced_council:
        parts.append(
            f'<div class="banner"><strong>Reduced council.</strong> '
            f"{session.council_size} of {len(session.students)} seated participants "
            "answered. Read the verdict as the product of a smaller council.</div>"
        )
    for warning in session.council_warnings:
        parts.append(
            f'<div class="banner"><strong>Single-lab council.</strong> {_e(warning)}. '
            "Agreement here is weaker evidence than agreement across families.</div>"
        )
    if not session.cost_is_complete:
        parts.append(
            '<div class="banner"><strong>Incomplete cost.</strong> '
            f"Unpriced participants: {_e(', '.join(session.unpriced_seats))}. "
            "The figures below are a lower bound.</div>"
        )

    # -- the answer
    parts.append("<h2>The answer</h2>")
    if session.verdict:
        parts.append(
            f'<div class="panel answer"><div>{_e(session.verdict.final_answer)}</div>'
            f'<div class="note">{_e(session.verdict.confidence_note)}</div></div>'
        )
    else:
        parts.append(
            f'<div class="panel answer"><strong>No verdict.</strong> '
            f"{_e(session.failed_reason or 'unknown')}</div>"
        )

    # -- opening positions
    parts.append("<h2>Where they started</h2>")
    for student in session.present_students:
        claims = "".join(f"<li>{_e(c.text)}</li>" for c in student.initial.claims)
        parts.append(
            f'<div class="panel">'
            f'{_who_html(student.label, student.model_id, f" · {student.initial.confidence:.0%} confident")}'
            f'<p class="position">{_e(student.initial.position)}</p>'
            f'<ol class="claims">{claims}</ol></div>'
        )

    # -- sharpest objections, as a timeline
    parts.append("<h2>The sharpest objections</h2>")
    ranked = rank_objections(session)[:top_objections]
    if ranked:
        parts.append('<div class="timeline">')
        for item in ranked:
            objection = item.objection
            critic = session.students[objection.critic_seat]
            target = session.students[objection.target_seat]
            parts.append(
                f'<div class="beat{" moved" if item.score else ""}">'
                f'{_who_html(critic.label, critic.model_id, " objects")}'
                f'<div class="bubble">'
                f'<div class="target">on {_e(target.label)}’s claim '
                f"{objection.claim_n} — <span class=\"claim\">"
                f"{_e(objection.claim_text)}</span></div>"
                f"<div>{_e(objection.argument)}</div>"
                f'<div class="effect">Effect: {_e(item.effect)}.</div>'
                f"</div></div>"
            )
        parts.append("</div>")
    else:
        parts.append('<div class="panel">No objections were recorded.</div>')

    # -- mind changes
    parts.append("<h2>Who changed their mind</h2>")
    moved = [s for s in session.present_students if s.changed_position]
    if not moved:
        parts.append(
            '<div class="panel held">Nobody changed position. Every participant held '
            "its opening answer through the critique round — which is a finding, not "
            "a formality: it means no objection landed.</div>"
        )
    for student in moved:
        change = student.position_change or {}
        details = "".join(
            f"<li>withdrew: {_e(c['text'])}</li>"
            for c in (student.diff or {}).get("claims_dropped", [])
        )
        delta = (student.diff or {}).get("confidence_delta", 0.0)
        if abs(delta) > 1e-9:
            details += f"<li>confidence {delta:+.0%}</li>"
        parts.append(
            f'<div class="panel change">'
            f"{_who_html(student.label, student.model_id, ' changed position')}"
            f'<div class="from">{_e(change.get("from", ""))}</div>'
            f'<div class="to">{_e(change.get("to", ""))}</div>'
            + (f"<ul>{details}</ul>" if details else "")
            + "</div>"
        )
    held = [
        s for s in session.present_students if s.diff is not None and not s.changed_position
    ]
    if held and moved:
        names = ", ".join(f"{s.label} ({s.model_id})" for s in held)
        parts.append(f'<div class="panel held">Held position: {_e(names)}.</div>')

    # -- minority report as a closing panel
    parts.append("<h2>Minority report</h2>")
    if session.verdict and session.verdict.minority_report:
        items = []
        for entry in session.verdict.minority_report:
            model = next(
                (m.get("source_model", "") for m in session.minority
                 if m.get("source") == entry.source),
                "",
            )
            who = entry.source + (f" · {model}" if model else "")
            items.append(
                f'<li><div class="src">{_e(who)} — {_e(entry.kind)}</div>'
                f"<div>{_e(entry.substance)}</div></li>"
            )
        parts.append(
            '<div class="minority"><p>Raised in this session and <strong>not</strong> '
            "reflected in the final answer:</p><ul>" + "".join(items) + "</ul></div>"
        )
    else:
        parts.append(
            '<div class="panel">Nothing was recorded as left out. Treat an empty '
            "minority report as a claim the arbiter made, not as proof of consensus.</div>"
        )

    # -- the numbers
    rows = [
        ("Council", f"{session.council_size} of {len(session.students)} answered"),
        ("Objections", str(len(session.objections))),
        (
            "Opening wording spread",
            f"{session.disagreement['score']:.2f} — {session.disagreement['label']}"
            if session.disagreement
            else "not recorded",
        ),
        ("Position-change rate", f"{session.position_change_rate:.0%}"),
        ("Dissent preserved", "yes" if session.dissent_preserved else "no"),
        ("Claim compliance", f"{session.compliance_rate:.0%}"),
        (
            "Session cost",
            f"${session.cost_est:.4f}"
            + ("" if session.cost_is_complete else " (lower bound)"),
        ),
    ]
    if session.baseline_model:
        rows.append(
            (f"One model, once ({session.baseline_model})",
             f"${session.baseline_cost_est:.4f}")
        )
    if session.cost_multiple is not None:
        rows.append(("Multiple", f"{session.cost_multiple:.1f}× a single answer"))
    if session.discarded:
        rows.append(
            ("Re-prompts", f"{len(session.discarded)} (${session.repair_cost_est:.4f})")
        )

    parts.append("<h2>The numbers</h2><div class=\"panel grid\"><table><tbody>")
    for name, value in rows:
        parts.append(f"<tr><td>{_e(name)}</td><td>{_e(value)}</td></tr>")
    parts.append("</tbody></table></div>")

    parts.append(
        f'<p class="foot">Session <code>{_e(session.session_id)}</code>. '
        "Every panel above was rebuilt from the session’s JSONL trace; nothing "
        "was passed in from the engine. Position changes are computed by diffing "
        "the sheets, not taken from what the models said about themselves."
        "</p>"
    )
    parts.append("</div></body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def _as_session(source: Any) -> ReplayedSession:
    """Accept a replayed session, a live result, or a list of events.

    A live `SessionResult` is routed through `replay` rather than read
    directly. That is the point: if the report could read the result object it
    would stop noticing when the trace goes incomplete, and the completeness
    guarantee would rot the first time somebody added a field to the engine
    and forgot the event.
    """
    if isinstance(source, ReplayedSession):
        return source
    events = getattr(source, "events", source)
    return replay(list(events))


def write_report(source: Any, path: str, *, markdown: bool = True) -> str:
    """Write the HTML report (and a Markdown sibling). Returns the HTML path."""
    session = _as_session(source)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_html(session))
    if markdown:
        md_path = os.path.splitext(path)[0] + ".md"
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(session))
    return path
