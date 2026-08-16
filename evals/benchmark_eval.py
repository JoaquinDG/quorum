"""Run the judgment benchmark: Quorum vs one model vs one model self-critiquing.

    PYTHONPATH=src python3 evals/benchmark_eval.py            # offline smoke test
    PYTHONPATH=src python3 evals/benchmark_eval.py --out evals/BENCHMARK.md

Offline this is a **smoke test, not a result**. Every arm is answered by a mock
that emits canned text and scored by a mock that hashes it, so the numbers say
only that the harness runs end to end. Anything it writes is stamped with that
warning in its first line, so a file that escapes into a README cannot be
mistaken for evidence.

Publishing real numbers is a matter of swapping the provider pool for real
adapters and rerunning. The result goes in the README whatever it says — the
project's claims (surfaced disagreement, auditable reasoning) do not depend on
Quorum winning, and a benchmark whose outcome is only published when it
flatters the author is not a benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    AnthropicProvider,
    BenchmarkMockProvider,
    Council,
    ModelCost,
    OpenAICompatibleProvider,
    ProviderPool,
    Seat,
    SessionConfig,
    demo_council,
    load_tasks,
    run_benchmark,
)
from quorum.providers.base import MockProvider  # noqa: E402

TASKS = os.path.join(os.path.dirname(__file__), "judgment_tasks.json")

# The real arms. This is the lineup the lineup experiment picked: three labs,
# one seat each, and an arbiter sharing a lab with nobody — the only council
# measured here that raises no warnings, and also the cheapest.
P = {
    "sonnet": ModelCost(3.0, 15.0), "gpt": ModelCost(2.5, 10.0),
    "deepseek": ModelCost(0.28, 1.10), "gemini_pro": ModelCost(1.25, 10.0),
    "opus": ModelCost(5.0, 25.0),
}
REAL_COUNCIL = Council(
    students=(
        Seat("claude-sonnet-5", "anthropic", P["sonnet"]),
        Seat("gpt-5.1", "openai", P["gpt"]),
        Seat("deepseek-v4-pro", "deepseek", P["deepseek"]),
    ),
    arbiter=Seat("gemini-3.1-pro-preview", "google", P["gemini_pro"]),
)
# The single-model arm is the thing a user would actually do instead, so it is
# the strongest seat available rather than the cheapest.
REAL_SINGLE = Seat("claude-opus-5", "anthropic", P["opus"])
# The judge must have taken no part in any arm, or it is scoring its own work.
REAL_JUDGE = Seat("kimi-k2.6", "moonshot", ModelCost(0.60, 2.50))

VENDORS = {
    "openai": ("https://api.openai.com", "OPENAI_API_KEY", "/v1/chat/completions"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "/v1/chat/completions"),
    "google": ("https://generativelanguage.googleapis.com", "GEMINI_API_KEY",
               "/v1beta/openai/chat/completions"),
    "moonshot": ("https://api.moonshot.ai", "MOONSHOT_API_KEY", "/v1/chat/completions"),
}


def real_pool() -> ProviderPool:
    providers = []
    labs = {s.provider for s in REAL_COUNCIL.seats()} | {REAL_SINGLE.provider,
                                                         REAL_JUDGE.provider}
    for name in sorted(labs):
        if name == "anthropic":
            providers.append(AnthropicProvider())
        else:
            base, key, path = VENDORS[name]
            providers.append(OpenAICompatibleProvider(
                name=name, base_url=base, env_var=key, chat_path=path))
    return ProviderPool(providers)


def mock_pool_with_judge(council, judge_seat: Seat) -> ProviderPool:
    """Every seat plus a judge, all offline, all wrapped for the extra arms."""
    personas = {seat.model_id: i for i, seat in enumerate(council.students)}
    providers = []
    for name in sorted({s.provider for s in council.seats()} | {judge_seat.provider}):
        inner = MockProvider(personas=personas)
        inner.name = name
        providers.append(BenchmarkMockProvider(inner, name=name))
    return ProviderPool(providers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Quorum judgment benchmark.")
    parser.add_argument("--out", default=None, help="write a Markdown report here")
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N tasks")
    parser.add_argument("--real", action="store_true",
                        help="spend real money on real models")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    if args.real:
        needed = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
                  "GEMINI_API_KEY", "MOONSHOT_API_KEY"]
        missing = [v for v in needed if not os.environ.get(v)]
        if missing:
            print(f"missing: {', '.join(missing)}", file=sys.stderr)
            return 2
        council, judge_seat, providers = REAL_COUNCIL, REAL_JUDGE, real_pool()
        single_seat = REAL_SINGLE
        print(f"benchmark — {len(tasks)} judgement tasks, 3 arms, REAL MODELS")
        print(f"  council: {', '.join(s.model_id for s in council.students)}")
        print(f"  arbiter: {council.arbiter.model_id}")
        print(f"  single:  {single_seat.model_id}")
        print(f"  judge:   {judge_seat.model_id} (took no part in any arm)")
        print(f"  warnings: {council.warnings or '(none)'}")
    else:
        council = demo_council()
        judge_seat = Seat("judge-model", "judgelab", ModelCost(5.0, 25.0))
        providers = mock_pool_with_judge(council, judge_seat)
        single_seat = None
        print(f"benchmark — {len(tasks)} judgement tasks, 3 arms, offline mocks")
        print("  arms: quorum · single (strongest seat, once) · self_critique")
    print()

    report = run_benchmark(
        tasks,
        council,
        providers,
        judge_seat,
        single_seat=single_seat,
        config=SessionConfig(baseline_seat=REAL_SINGLE) if args.real else None,
        is_mock=not args.real,
        on_task=lambda t: print(f"  running {t.key}…", flush=True),
    )

    print()
    if not args.real:
        print("  MOCK RUN — the numbers below measure the harness, not the models.")
        print()
    print(f"  {'arm':<16} {'all criteria':>13} {'neutral only':>13} {'cost':>10}")
    for arm in ("quorum", "single", "self_critique"):
        print(f"  {arm:<16} {report.mean(arm):>13.3f} "
              f"{report.mean(arm, neutral_only=True):>13.3f} "
              f"{'$' + format(report.cost(arm), '.4f'):>10}")
    print()
    print(f"  wins (all criteria):     {report.wins()}")
    print(f"  wins (neutral criteria): {report.wins(neutral_only=True)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report.render_markdown())
        print(f"\n  written to {os.path.relpath(args.out)}")

    # Harness health, not model quality: every task must have produced a
    # scored answer in every arm, or the run proves nothing either way.
    incomplete = [
        (o.task.key, arm)
        for o in report.outcomes
        for arm in ("quorum", "single", "self_critique")
        if o.arms[arm].error or not o.arms[arm].scores
    ]
    if incomplete:
        print(f"\nFAIL: {len(incomplete)} arm(s) produced no scored answer: "
              f"{incomplete[:5]}")
        return 1
    print("\nall arms produced scored answers on all tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
