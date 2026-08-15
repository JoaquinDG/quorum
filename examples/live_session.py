"""Run one real session against real models.

    export ANTHROPIC_API_KEY=...        # your shell, never this repo
    PYTHONPATH=src python3 examples/live_session.py "Should we rebuild or refactor?"

    # mixed-lab council, which is what the protocol is actually for:
    export OPENAI_API_KEY=...
    PYTHONPATH=src python3 examples/live_session.py --mixed "..."

This is the smallest real thing you can do with Quorum: about ten model calls,
one trace, one report. It exists so that the first real run is one command
rather than an afternoon, and so the schema meets genuine model output early —
whether real models actually return five one-sentence claims in valid JSON is
the single biggest untested assumption in the project, and it is cheap to find
out.

A single-lab council (the default here, since one key is the common case) is
labelled as such in the trace, the report and the console. It runs; it just
means less. See `Council.single_lab`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    AnthropicProvider,
    Council,
    ModelCost,
    OpenAICompatibleProvider,
    ProviderPool,
    Seat,
    Session,
    Task,
    should_convene,
    write_report,
)

HERE = os.path.dirname(__file__)

# Prices are per million tokens and are YOURS to keep current. These are
# placeholders; check the vendor's page before quoting a cost figure.
ANTHROPIC_COUNCIL = Council(
    students=(
        Seat("claude-opus-4-5", "anthropic", ModelCost(5.0, 25.0)),
        Seat("claude-sonnet-4-5", "anthropic", ModelCost(3.0, 15.0)),
        Seat("claude-haiku-4-5", "anthropic", ModelCost(1.0, 5.0)),
    ),
    arbiter=Seat("claude-opus-4-5-arbiter", "anthropic-arbiter", ModelCost(5.0, 25.0)),
)

MIXED_COUNCIL = Council(
    students=(
        Seat("claude-sonnet-4-5", "anthropic", ModelCost(3.0, 15.0)),
        Seat("gpt-5", "openai", ModelCost(2.5, 10.0)),
        Seat("claude-haiku-4-5", "anthropic-haiku", ModelCost(1.0, 5.0)),
    ),
    arbiter=Seat("claude-opus-4-5", "anthropic-arbiter", ModelCost(5.0, 25.0)),
)


def build_pool(council: Council) -> ProviderPool:
    """One adapter instance per provider name on the council.

    The arbiter is given its own provider name so the no-self-grading rule
    stays legible at the wiring level too: nothing on the council shares an
    identity with the seat that grades it.
    """
    providers = []
    for name in sorted({seat.provider for seat in council.seats()}):
        if name.startswith("openai"):
            providers.append(OpenAICompatibleProvider(name=name))
        else:
            provider = AnthropicProvider()
            provider.name = name
            providers.append(provider)
    return ProviderPool(providers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one real Quorum session.")
    parser.add_argument("question", nargs="?", default=(
        "Our ingestion pipeline is three years old and increasingly slow. Should we "
        "rebuild it on a streaming architecture this year, or refactor it in place? "
        "The rebuild is a one-way door on the storage format."
    ))
    parser.add_argument("--mixed", action="store_true",
                        help="use a mixed-lab council (needs OPENAI_API_KEY too)")
    parser.add_argument("--type", default="architecture")
    parser.add_argument("--complexity", type=float, default=0.85)
    parser.add_argument("--force", action="store_true",
                        help="run even if the convening rule says not to")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.\n\n"
              "  export ANTHROPIC_API_KEY=...\n\n"
              "Keys are read from the environment only — nothing in this repo "
              "stores one, and none is ever written to a trace or a report.",
              file=sys.stderr)
        return 2

    council = MIXED_COUNCIL if args.mixed else ANTHROPIC_COUNCIL
    if args.mixed and not os.environ.get("OPENAI_API_KEY"):
        print("--mixed needs OPENAI_API_KEY as well.", file=sys.stderr)
        return 2

    task = Task(args.question, args.type, args.complexity)
    decision = should_convene(task)
    print(f"convening rule: {'CONVENE' if decision.convene else 'declined'} — "
          f"{decision.reason}")
    if not decision.convene and not args.force:
        print("\nNot convening. Re-run with --force to override.")
        return 0

    for warning in council.warnings:
        print(f"\n  WARNING: {warning}")
        print("  A mixed-lab council is what the protocol is for; run --mixed if you can.")

    print(f"\ncouncil: {', '.join(s.model_id for s in council.students)}")
    print(f"arbiter: {council.arbiter.model_id}")
    print("\nrunning — about 10 calls…\n")

    trace = os.path.join(HERE, "..", "traces", "live.jsonl")
    result = Session(council, build_pool(council), trace_path=trace).run(task.prompt)

    if not result.ok:
        print(f"session did not complete: {result.failed_reason}")
    else:
        print(result.verdict.final_answer)
        print(f"\n{result.verdict.confidence_note}")
        if result.verdict.minority_report:
            print("\nminority report:")
            for item in result.verdict.minority_report:
                print(f"  [{item.source}] {item.substance}")

    print("\n--- how the protocol behaved on real output ---")
    stats = result.stats()
    print(f"  council:              {stats['council_size']} students"
          + (" (REDUCED)" if stats["reduced_council"] else ""))
    print(f"  single lab:           {stats['single_lab']}")
    print(f"  objections:           {stats['objections']}")
    print(f"  position-change rate: {stats['position_change_rate']:.0%}")
    print(f"  claim compliance:     {stats['compliance_rate']:.0%}"
          + ("  <- real models vs the schema" if stats["compliance_rate"] < 1 else ""))
    print(f"  re-prompts:           {stats['discarded_calls']}")
    print(f"  disagreement:         {stats['disagreement_score']} "
          f"({stats['disagreement_label']})")
    for absence in result.absences:
        print(f"  ABSENT round {absence.round} seat {absence.seat}: "
              f"{absence.reason} — {absence.detail[:120]}")

    print(f"\n  cost: ${result.cost_est:.4f}", end="")
    if result.cost_multiple:
        print(f"  ({result.cost_multiple:.1f}x one answer from "
              f"{result.baseline.model_id})")
    else:
        print()

    report = os.path.join(HERE, "..", "reports", "live.html")
    write_report(result, report)
    print(f"\n  trace:  {os.path.relpath(trace)}")
    print(f"  report: {os.path.relpath(report)}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
