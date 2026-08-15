"""Run one real session against real models.

    export ANTHROPIC_API_KEY=...        # your shell, never this repo
    PYTHONPATH=src python3 examples/live_session.py --check     # verify setup, ~4 tiny calls
    PYTHONPATH=src python3 examples/live_session.py "Should we rebuild or refactor?"

    export OPENAI_API_KEY=...
    PYTHONPATH=src python3 examples/live_session.py --mixed --check

Model ids are the thing most likely to be stale here, because vendors retire
and rename them faster than an example script gets updated. Every seat is
therefore overridable from the environment, and `--check` spends four
one-token calls confirming the ids answer before a session spends forty. A
404 on the third round of a real session is an expensive way to learn about a
rename.

`--check` is worth running first every time. It is the cheapest thing in the
repo and it distinguishes the three failures that look identical from the
outside: a missing key, a wrong model id, and a real outage.
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
    ProviderConfigError,
    ProviderError,
    ProviderPool,
    Seat,
    Session,
    Task,
    should_convene,
    write_report,
)

HERE = os.path.dirname(__file__)

# Prices are per million tokens and are YOURS to keep current. Check the
# vendor's page before quoting any cost figure these produce.
PRICES = {
    "opus": ModelCost(5.0, 25.0),
    "sonnet": ModelCost(3.0, 15.0),
    "haiku": ModelCost(1.0, 5.0),
    "fable": ModelCost(3.0, 15.0),
    "openai": ModelCost(2.5, 10.0),
}


def model(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


def anthropic_council() -> Council:
    """Three Claude tiers plus a fourth as arbiter.

    All one lab, which `Council.single_lab` flags loudly and correctly — their
    blind spots correlate, so their agreement is weak evidence. Useful when you
    only hold one key; not what the protocol is for.
    """
    return Council(
        students=(
            Seat(model("QUORUM_MODEL_1", "claude-sonnet-5"), "anthropic", PRICES["sonnet"]),
            Seat(model("QUORUM_MODEL_2", "claude-haiku-4-5-20251001"), "anthropic", PRICES["haiku"]),
            Seat(model("QUORUM_MODEL_3", "claude-fable-5"), "anthropic", PRICES["fable"]),
        ),
        arbiter=Seat(model("QUORUM_ARBITER", "claude-opus-5"), "anthropic", PRICES["opus"]),
    )


def mixed_council() -> Council:
    """Two labs across three seats, with the strongest model arbitrating.

    Two is better than one and still short of the goal. A third family —
    Kimi, DeepSeek, Gemini, anything OpenAI-compatible — goes in by setting
    QUORUM_MODEL_3 and QUORUM_PROVIDER_3 plus its base URL and env var below.
    """
    third_provider = os.environ.get("QUORUM_PROVIDER_3", "anthropic")
    return Council(
        students=(
            Seat(model("QUORUM_MODEL_1", "claude-sonnet-5"), "anthropic", PRICES["sonnet"]),
            Seat(model("QUORUM_MODEL_2", "gpt-5.1"), "openai", PRICES["openai"]),
            Seat(model("QUORUM_MODEL_3", "claude-haiku-4-5-20251001"), third_provider, PRICES["haiku"]),
        ),
        arbiter=Seat(model("QUORUM_ARBITER", "claude-opus-5"), "anthropic", PRICES["opus"]),
    )


def build_pool(council: Council) -> ProviderPool:
    """One adapter per provider name on the council.

    Anything not named `anthropic` is assumed OpenAI-compatible, which is what
    Kimi, DeepSeek, Mistral, Together, Fireworks, Groq and xAI all speak. Point
    QUORUM_BASE_URL_<provider> and QUORUM_KEY_<provider> at the vendor and the
    adapter needs no code change.
    """
    providers = []
    for name in sorted({seat.provider for seat in council.seats()}):
        if name == "anthropic":
            providers.append(AnthropicProvider())
            continue
        upper = name.upper().replace("-", "_")
        providers.append(
            OpenAICompatibleProvider(
                name=name,
                base_url=os.environ.get(f"QUORUM_BASE_URL_{upper}", "https://api.openai.com"),
                env_var=os.environ.get(f"QUORUM_KEY_{upper}", "OPENAI_API_KEY"),
            )
        )
    return ProviderPool(providers)


def preflight(council: Council, pool: ProviderPool) -> int:
    """One tiny call per seat. Cheap insurance against a stale model id."""
    print("preflight — one 1-token call per seat\n")
    failures = 0
    for seat in council.seats():
        role = "arbiter" if seat is council.arbiter else "student"
        try:
            pool.get(seat.provider).complete(seat.model_id, "Reply with: ok", 5)
            print(f"  OK        {role:<8} {seat.model_id}  ({seat.provider})")
        except ProviderConfigError as exc:
            failures += 1
            print(f"  NO KEY    {role:<8} {seat.model_id}: {exc}")
        except ProviderError as exc:
            failures += 1
            hint = ""
            if "404" in str(exc) or "not_found" in str(exc) or "does not exist" in str(exc):
                hint = "  <- model id is wrong or retired; override it, see below"
            print(f"  FAILED    {role:<8} {seat.model_id}: {str(exc)[:150]}{hint}")

    if failures:
        print(f"\n{failures} seat(s) failed. Override any model id without editing code:")
        print("  export QUORUM_MODEL_1=...   QUORUM_MODEL_2=...   QUORUM_MODEL_3=...")
        print("  export QUORUM_ARBITER=...")
        return 1
    print("\nall seats answered. the council is ready to run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one real Quorum session.")
    parser.add_argument("question", nargs="?", default=(
        "Our ingestion pipeline is three years old and increasingly slow. Should we "
        "rebuild it on a streaming architecture this year, or refactor it in place? "
        "The rebuild is a one-way door on the storage format."
    ))
    parser.add_argument("--mixed", action="store_true",
                        help="two-lab council (needs OPENAI_API_KEY too)")
    parser.add_argument("--check", action="store_true",
                        help="verify keys and model ids, then stop")
    parser.add_argument("--type", default="architecture")
    parser.add_argument("--complexity", type=float, default=0.85)
    parser.add_argument("--force", action="store_true",
                        help="run even if the convening rule says not to")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.\n\n"
              "  export ANTHROPIC_API_KEY=...\n\n"
              "Keys are read from the environment only. Nothing in this repo stores "
              "one, and no key is ever written to a trace or a report.", file=sys.stderr)
        return 2
    if args.mixed and not os.environ.get("OPENAI_API_KEY"):
        print("--mixed needs OPENAI_API_KEY as well.", file=sys.stderr)
        return 2

    council = mixed_council() if args.mixed else anthropic_council()
    pool = build_pool(council)

    if args.check:
        return preflight(council, pool)

    task = Task(args.question, args.type, args.complexity)
    decision = should_convene(task)
    print(f"convening rule: {'CONVENE' if decision.convene else 'declined'} — {decision.reason}")
    if not decision.convene and not args.force:
        print("\nNot convening. Re-run with --force to override.")
        return 0

    for warning in council.warnings:
        print(f"\n  WARNING: {warning}")

    print(f"\ncouncil: {', '.join(s.model_id for s in council.students)}")
    print(f"arbiter: {council.arbiter.model_id}")
    print("\nrunning — about 10 calls…\n")

    trace = os.path.join(HERE, "..", "traces", "live.jsonl")
    result = Session(council, pool, trace_path=trace).run(task.prompt)

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
    moved = sum(1 for s in result.students if s.changed_position)
    print(f"  position changes:     {moved} of {stats['council_size']}"
          f"  ({stats['position_change_rate']:.0%})")
    print(f"  claim compliance:     {stats['compliance_rate']:.0%}")
    print(f"  re-prompts:           {stats['discarded_calls']}")
    for absence in result.absences:
        print(f"  ABSENT round {absence.round} seat {absence.seat}: "
              f"{absence.reason} — {absence.detail[:120]}")

    print(f"\n  cost: ${result.cost_est:.4f}", end="")
    if result.cost_multiple:
        print(f"  ({result.cost_multiple:.1f}x one answer from {result.baseline.model_id})")
    else:
        print()

    report = os.path.join(HERE, "..", "reports", "live.html")
    write_report(result, report)
    print(f"\n  trace:  {os.path.relpath(trace)}")
    print(f"  report: {os.path.relpath(report)}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
