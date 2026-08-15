"""The whole pipeline, offline, with no API keys.

    PYTHONPATH=src python3 examples/quickstart.py

Runs it end to end: the convening rule decides whether the question earns a
council, the four-round protocol runs against mock providers, and the Session
Report is generated *from the trace file* rather than from the session object
— the point being that the file is sufficient.

It also runs the rule against a task that should be refused, because a
convening rule that only ever says yes is not a rule.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    Session,
    Task,
    demo_council,
    mock_pool,
    render,
    replay_file,
    should_convene,
    write_report,
)

HERE = os.path.dirname(__file__)
TRACE = os.path.join(HERE, "..", "traces", "quickstart.jsonl")
REPORT = os.path.join(HERE, "..", "reports", "quickstart.html")

QUESTION = Task(
    "Our ingestion pipeline is three years old and increasingly slow. Should we rebuild "
    "it on a streaming architecture this year, or refactor it in place? The rebuild is a "
    "one-way door on the storage format and the roadmap has no slack.",
    task_type="architecture",
    complexity=0.85,
)

DECLINED = Task(
    "Extract the plan names, prices and seat limits from these five competitor pages.",
    task_type="extraction",
    complexity=0.3,
)


def rule(header: str) -> None:
    print(f"\n{header}\n{'─' * len(header)}")


def main() -> int:
    rule("1. should this question convene a council?")
    for task in (DECLINED, QUESTION):
        decision = should_convene(task)
        mark = "CONVENE" if decision.convene else "declined"
        print(f"  [{mark:>8}] {task.prompt[:64]}…")
        print(f"             {decision.reason}")

    if not should_convene(QUESTION).convene:  # pragma: no cover - demo invariant
        print("the demo question no longer convenes; nothing to show")
        return 1

    rule("2. the council")
    council = demo_council()
    for seat, student in enumerate(council.students, start=1):
        print(f"  Student {seat}: {student.model_id} ({student.provider})")
    print(f"  Arbiter:   {council.arbiter.model_id} ({council.arbiter.provider})")

    if os.path.exists(TRACE):
        os.remove(TRACE)  # the trace is append-only; start the demo clean
    result = Session(council, mock_pool(council), trace_path=TRACE).run(QUESTION.prompt)

    rule("3. the session, replayed from the trace")
    # Everything below is rebuilt from the file. If a line appears here, the
    # trace carried it; that is the whole test of the format.
    replayed = replay_file(TRACE)[0]
    print(render(replayed))

    rule("4. protocol health")
    stats = result.stats()
    print(f"  council:              {stats['council_size']} students"
          + (" (REDUCED)" if stats["reduced_council"] else ""))
    print(f"  objections raised:    {stats['objections']}")
    print(f"  position-change rate: {stats['position_change_rate']:.0%}"
          "   (0% = theatre, 100% = herding)")
    print(f"  dissent preserved:    {stats['dissent_preserved']}")
    print(f"  claim compliance:     {stats['compliance_rate']:.0%}")
    print(f"  re-prompts:           {stats['discarded_calls']}")

    rule("5. wording spread across the opening sheets")
    spread = result.disagreement
    print(f"  wording spread:        {spread.score:.2f} — {spread.label}")
    print("    (vocabulary only — this does NOT measure agreement; see README)")
    print(f"    position divergence: {spread.position_divergence:.2f}")
    print(f"    claim divergence:    {spread.claim_divergence:.2f}")
    print(f"    confidence spread:   {spread.confidence_spread:.2f}")

    rule("6. the bill")
    print(f"  this session:         ${result.cost_est:.4f}"
          + ("" if result.cost_is_complete else "  (LOWER BOUND — unpriced seats)"))
    print(f"  one model, once:      ${result.baseline.cost_est:.4f}  "
          f"({result.baseline.model_id})")
    if result.cost_multiple is not None:
        guardrail = "within" if result.cost_multiple <= 8 else "OVER"
        print(f"  multiple:             {result.cost_multiple:.1f}x  "
              f"({guardrail} the 8x guardrail)")

    write_report(result, REPORT)
    rule("7. artifacts")
    print(f"  trace:    {os.path.relpath(TRACE)}")
    print(f"  report:   {os.path.relpath(REPORT)}  (+ .md fallback)")
    print(f"  replay:   python3 replay.py {os.path.relpath(TRACE)}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
