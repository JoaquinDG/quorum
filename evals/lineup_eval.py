"""Which lineup earns its money? Same question, different seats.

    PYTHONPATH=src python3 evals/lineup_eval.py --check      # what it would cost
    PYTHONPATH=src python3 evals/lineup_eval.py --run        # spend real money
    PYTHONPATH=src python3 evals/lineup_eval.py --run --only arbiter

Two questions this answers with numbers instead of taste:

**Which arbiter tier?** The spec leaves it open — strongest model for the best
synthesis, or a neutral mid-tier that is cheaper and less likely to impose its
own answer. Measurement has since made it urgent rather than academic: the
arbiter is ~69% of a session's cost, because it reads the whole transcript
while every other seat reads one prompt. It is the single biggest lever on the
cost guardrail this project keeps missing.

**Which model in the weak seat?** A council is only worth its most useless
participant. In the first three-lab run one seat produced a sheet and a
revision but *zero objections*, having failed the critique round twice. That
is a seat paying rent and contributing nothing to the only thing the protocol
sells.

**On reading the output.** Every lineup runs the question ONCE. Model sampling
is stochastic and one session per arm cannot separate a real effect from
noise — a difference here is a hypothesis worth testing at n=20, not a result.
The numbers are reported anyway, because "we did not measure it" is worse than
"we measured it once and said so".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    AnthropicProvider,
    SessionConfig,
    Council,
    ModelCost,
    OpenAICompatibleProvider,
    ProviderPool,
    Seat,
    Session,
)

HERE = os.path.dirname(__file__)
TRACES = os.path.join(HERE, "..", "traces", "lineups")

BASELINE_SEAT = Seat("claude-opus-5", "anthropic", ModelCost(5.0, 25.0))
"""One ruler for every arm, seated or not — a cross-lab-arbiter lineup has no
Opus in it at all. See `costs.pick_baseline_seat`."""

QUESTION = (
    "Our ingestion pipeline is three years old and increasingly slow. Should we rebuild "
    "it on a streaming architecture this year, or refactor it in place? The rebuild is a "
    "one-way door on the storage format and the roadmap has no slack."
)

# Estimates, per million tokens. Verify against your vendor before quoting.
P = {
    "opus": ModelCost(5.0, 25.0),
    "sonnet": ModelCost(3.0, 15.0),
    "fable": ModelCost(3.0, 15.0),
    "haiku": ModelCost(1.0, 5.0),
    "gpt": ModelCost(2.5, 10.0),
    "deepseek": ModelCost(0.28, 1.10),
    "moonshot": ModelCost(0.60, 2.50),
    "gemini": ModelCost(0.30, 2.50),
}

SONNET = Seat("claude-sonnet-5", "anthropic", P["sonnet"])
GPT = Seat("gpt-5.1", "openai", P["gpt"])
DEEPSEEK = Seat("deepseek-chat", "deepseek", P["deepseek"])
DEEPSEEK_PRO = Seat("deepseek-v4-pro", "deepseek", P["deepseek"])
FABLE = Seat("claude-fable-5", "anthropic", P["fable"])
KIMI = Seat("kimi-k2.6", "moonshot", P["moonshot"])
GEMINI = Seat("gemini-3.5-flash", "google", P["gemini"])
GEMINI_PRO = Seat("gemini-3.1-pro-preview", "google", ModelCost(1.25, 10.0))

LINEUPS: dict[str, tuple[str, Council]] = {
    # -- experiment 1: which arbiter? students held fixed ------------------
    "arbiter-opus": (
        "baseline — strongest arbiter, the current default",
        Council(students=(SONNET, GPT, DEEPSEEK), arbiter=Seat("claude-opus-5", "anthropic", P["opus"])),
    ),
    "arbiter-sonnet": (
        "mid-tier arbiter — the spec's cheaper alternative",
        Council(students=(Seat("claude-sonnet-4-6", "anthropic", P["sonnet"]), GPT, DEEPSEEK),
                arbiter=Seat("claude-sonnet-5", "anthropic", P["sonnet"])),
    ),
    "arbiter-fable": (
        "a different Claude line arbitrating",
        Council(students=(SONNET, GPT, DEEPSEEK), arbiter=Seat("claude-fable-5", "anthropic", P["fable"])),
    ),
    # -- experiment 2: fix the seat that contributed nothing ---------------
    "seat-deepseek-pro": (
        "weak seat upgraded within the same lab",
        Council(students=(SONNET, GPT, DEEPSEEK_PRO), arbiter=Seat("claude-opus-5", "anthropic", P["opus"])),
    ),
    "seat-fable": (
        "weak seat replaced by a fourth Claude line — costs a lab",
        Council(students=(SONNET, GPT, FABLE), arbiter=Seat("claude-opus-5", "anthropic", P["opus"])),
    ),
    # -- experiment 3: does a fourth LAB earn its 1.4x? --------------------
    # The control is `seat-deepseek-pro`: identical students minus Kimi. Any
    # difference is attributable to the fourth lab rather than to a better
    # seat, which is the confound that made the earlier lineups hard to read.
    "four-lab": (
        "a fourth distinct lab added to the best three-lab lineup",
        Council(students=(SONNET, GPT, DEEPSEEK_PRO, KIMI),
                arbiter=Seat("claude-opus-5", "anthropic", P["opus"])),
    ),
    # -- experiment 4: an arbiter that shares no lab with any student ------
    # `arbiter_shares_lab` has fired on every real session so far, because
    # every council has had an Anthropic arbiter sitting above at least one
    # Anthropic student. This is the first lineup where the synthesis is
    # genuinely independent of every participant.
    "arbiter-cross-lab": (
        "first arbiter sharing no lab with any student",
        Council(students=(SONNET, GPT, DEEPSEEK_PRO), arbiter=GEMINI),
    ),
    "arbiter-cross-lab-pro": (
        "independent arbiter again, at the stronger tier",
        Council(students=(SONNET, GPT, DEEPSEEK_PRO), arbiter=GEMINI_PRO),
    ),
    # -- experiment 5: every lab, one seat each ---------------------------
    "five-lab": (
        "one student per lab, the widest council these keys allow",
        Council(students=(SONNET, GPT, DEEPSEEK_PRO, KIMI, GEMINI),
                arbiter=Seat("claude-opus-5", "anthropic", P["opus"])),
    ),
}

VENDORS = {
    "openai": ("https://api.openai.com", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.ai", "MOONSHOT_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com", "GEMINI_API_KEY"),
}

# Vendors whose OpenAI-compatible endpoint is not at /v1/chat/completions.
CHAT_PATHS = {"google": "/v1beta/openai/chat/completions"}


def build_pool(council: Council) -> ProviderPool:
    providers = []
    for name in sorted({s.provider for s in council.seats()}):
        if name == "anthropic":
            providers.append(AnthropicProvider())
        else:
            base, key = VENDORS[name]
            providers.append(OpenAICompatibleProvider(
                name=name, base_url=base, env_var=key,
                chat_path=CHAT_PATHS.get(name, "/v1/chat/completions"),
            ))
    return ProviderPool(providers)


def run_lineup(key: str, note: str, council: Council) -> dict:
    os.makedirs(TRACES, exist_ok=True)
    trace = os.path.join(TRACES, f"{key}.jsonl")
    if os.path.exists(trace):
        os.remove(trace)
    started = time.time()
    # Pin the baseline to one model across every arm. Left to default, the
    # baseline is the priciest seat in each council, so changing the lineup
    # moves the yardstick and the multiples stop comparing anything.
    result = Session(
        council,
        build_pool(council),
        trace_path=trace,
        config=SessionConfig(baseline_seat=BASELINE_SEAT),
    ).run(QUESTION, session_id=f"lineup-{key}")
    worst = result.worst_complier
    return {
        "lineup": key,
        "note": note,
        "students": [s.model_id for s in council.students],
        "arbiter": council.arbiter.model_id,
        "labs": len(council.labs()),
        "ok": result.ok,
        "council_size": result.council_size,
        "objections": len(result.objections),
        "position_change_rate": round(result.position_change_rate, 3),
        "movers": sum(1 for s in result.students if s.changed_position),
        "minority_items": len(result.verdict.minority_report) if result.verdict else 0,
        "compliance": round(result.compliance_rate, 3),
        "worst_complier": worst,
        "repairs": len(result.discarded_calls),
        "cost": round(result.cost_est, 4),
        "repair_cost": round(result.repair_cost_est, 4),
        "cost_multiple": round(result.cost_multiple, 1) if result.cost_multiple else None,
        "seconds": round(time.time() - started, 1),
        "failed_reason": result.failed_reason,
    }


def render(rows: list[dict]) -> str:
    out = [
        "# Lineup experiment",
        "",
        "> **n=1 per lineup.** Every arm ran the question once. Model sampling is",
        "> stochastic, so a difference below is a hypothesis worth testing at scale,",
        "> not a result. Reported anyway, because measuring once and saying so beats",
        "> not measuring.",
        "",
        f"Question held constant across all lineups. Prices are estimates.",
        "",
        "**Sorted by absolute cost, which is the only column that compares.**",
        "The `x baseline` column is within-session only: the baseline is measured",
        "from each session's own round 1, so a verbose council inflates its own",
        "yardstick. See `costs.cost_multiple`.",
        "",
        "| lineup | labs | objections | moved | minority | compliance | repairs | cost | x baseline |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: r.get("cost", 0)):
        if not r.get("ok"):
            out.append(f"| `{r['lineup']}` | {r['labs']} | — | — | — | — | — | "
                       f"${r.get('cost', 0):.4f} | FAILED |")
            continue
        out.append(
            f"| `{r['lineup']}` | {r['labs']} | {r['objections']} | "
            f"{r['movers']}/{r['council_size']} | {r['minority_items']} | "
            f"{r['compliance']:.0%} | {r['repairs']} | ${r['cost']:.4f} | "
            f"{r['cost_multiple']}x |"
        )
    out += ["", "## Lineups", ""]
    for r in rows:
        out.append(f"- **`{r['lineup']}`** — {r['note']}  ")
        out.append(f"  students: {', '.join(r['students'])} · arbiter: {r['arbiter']}  ")
        if r.get("worst_complier"):
            out.append(f"  weakest complier: `{r['worst_complier']}`  ")
        if r.get("failed_reason"):
            out.append(f"  **did not complete:** {r['failed_reason']}  ")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare council lineups on one question.")
    parser.add_argument("--run", action="store_true", help="actually spend money")
    parser.add_argument("--check", action="store_true", help="show the plan and stop")
    parser.add_argument("--only", default=None, help="substring filter on lineup name")
    parser.add_argument("--out", default=os.path.join(HERE, "LINEUPS.md"))
    args = parser.parse_args(argv)

    chosen = {k: v for k, v in LINEUPS.items() if not args.only or args.only in k}

    if args.check or not args.run:
        print(f"{len(chosen)} lineup(s), one session each, same question.\n")
        for key, (note, council) in chosen.items():
            print(f"  {key:20s} {', '.join(s.model_id for s in council.students)}")
            print(f"  {'':20s} arbiter {council.arbiter.model_id} · {len(council.labs())} lab(s)")
            for w in council.warnings:
                print(f"  {'':20s} ! {w[:90]}")
        print(f"\n  a three-lab session has been costing ~$0.40-0.55")
        print(f"  estimated total: ~${0.5 * len(chosen):.2f}")
        print("\n  re-run with --run to execute.")
        return 0

    needed = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"}
    for key, (_, council) in chosen.items():
        labs = {s.provider for s in council.seats()}
        if "moonshot" in labs:
            needed.add("MOONSHOT_API_KEY")
        if "google" in labs:
            needed.add("GEMINI_API_KEY")
    for var in sorted(needed):
        if not os.environ.get(var):
            print(f"{var} is not set", file=sys.stderr)
            return 2

    rows = []
    for key, (note, council) in chosen.items():
        print(f"running {key}…", flush=True)
        try:
            row = run_lineup(key, note, council)
        except Exception as exc:  # noqa: BLE001 - one lineup failing must not lose the rest
            print(f"  {key} raised: {exc}")
            row = {"lineup": key, "note": note, "ok": False, "cost": 0.0,
                   "labs": len(council.labs()), "failed_reason": str(exc)[:200],
                   "students": [s.model_id for s in council.students],
                   "arbiter": council.arbiter.model_id}
        rows.append(row)
        print(f"  {key}: objections={row.get('objections','-')} "
              f"cost=${row.get('cost',0):.4f} ok={row.get('ok')}")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render(rows))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"\ntotal spent: ${sum(r.get('cost', 0) for r in rows):.4f}")
    print(f"written to {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
