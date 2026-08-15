"""Run Quorum on Quorum's own open questions.

    PYTHONPATH=src python3 evals/dogfood.py --run

The project claims that for high-stakes ambiguous judgement calls, a council
surfaces disagreement a single model would hide. Quorum's own unresolved design
questions are exactly that shape — real trade-offs, no ground truth, and the
author has a stake in the answer, which is precisely when one model's confident
reply is least trustworthy.

So this is the honest test, and it can fail in a way that matters: if the
council produces four confident, interchangeable answers with an empty minority
report, the thesis is in trouble on its own home ground. The transcripts are
published either way.

The lineup is the best one measured — three labs, an arbiter sharing a lab with
nobody, and the seat that used to fail the schema replaced. It is also the
cheapest council run so far, which is a convenient accident rather than a
design goal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    Council, ModelCost, OpenAICompatibleProvider, ProviderPool, Seat,
    Session, SessionConfig, Task, should_convene, write_report,
)

HERE = os.path.dirname(__file__)
TRACES = os.path.join(HERE, "..", "traces", "dogfood")
REPORTS = os.path.join(HERE, "..", "reports", "dogfood")

COUNCIL = Council(
    students=(
        Seat("claude-sonnet-5", "anthropic", ModelCost(3.0, 15.0)),
        Seat("gpt-5.1", "openai", ModelCost(2.5, 10.0)),
        Seat("deepseek-v4-pro", "deepseek", ModelCost(0.28, 1.10)),
    ),
    arbiter=Seat("gemini-3.1-pro-preview", "google", ModelCost(1.25, 10.0)),
)
BASELINE = Seat("claude-opus-5", "anthropic", ModelCost(5.0, 25.0))

VENDORS = {
    "openai": ("https://api.openai.com", "OPENAI_API_KEY", "/v1/chat/completions"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "/v1/chat/completions"),
    "google": ("https://generativelanguage.googleapis.com", "GEMINI_API_KEY",
               "/v1beta/openai/chat/completions"),
}

QUESTIONS = {
    "minority_bar": Task(
        "Quorum's minority report currently preserves every dissent an arbiter leaves out of "
        "the final answer. A real session produced 13 items, which is thorough but works "
        "against the stated goal of a report a non-technical reader can follow in under three "
        "minutes. Should the arbiter apply a materiality bar and curate, or does curation "
        "become the synthesis-away the feature exists to prevent?", "policy", 0.8),
    "cost_guardrail": Task(
        "Quorum's spec sets a guardrail of at most 8x the cost of a single model answer. Eight "
        "measured sessions came in between 6x and 26x, and the multiple turned out not to be "
        "comparable across lineups because each session's baseline is measured from its own "
        "round one. Should the guardrail be abandoned, restated as absolute cost per session, "
        "or kept as an aspiration the project keeps missing?", "strategy", 0.8),
    "wording_metric": Task(
        "Quorum shipped a disagreement score, then demoted it after it called a unanimous "
        "council 'sharply contested' — it measures vocabulary overlap, not agreement. It is now "
        "labelled wording spread and nothing in the protocol depends on it. Should it be deleted "
        "outright, kept as a clearly labelled heuristic, or replaced with a semantic measure "
        "that costs an extra model call per session?", "architecture", 0.75),
    "council_size": Task(
        "Quorum caps a council at five students and defaults to three. Adding a fourth distinct "
        "lab measurably improved schema compliance and objection count at no extra cost in one "
        "session, contradicting the prediction that a fourth seat costs 1.4x. Should the default "
        "council size rise to four, stay at three, or become adaptive based on how contested the "
        "question appears?", "architecture", 0.8),
}


def build_pool() -> ProviderPool:
    providers = []
    from quorum import AnthropicProvider
    for name in sorted({s.provider for s in COUNCIL.seats()}):
        if name == "anthropic":
            providers.append(AnthropicProvider())
        else:
            base, key, path = VENDORS[name]
            providers.append(OpenAICompatibleProvider(
                name=name, base_url=base, env_var=key, chat_path=path))
    return ProviderPool(providers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Quorum on its own open questions.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--out", default=os.path.join(HERE, "DOGFOOD.md"))
    args = parser.parse_args(argv)

    chosen = {k: v for k, v in QUESTIONS.items() if not args.only or args.only in k}
    print(f"council: {', '.join(s.model_id for s in COUNCIL.students)}")
    print(f"arbiter: {COUNCIL.arbiter.model_id}")
    print(f"warnings: {COUNCIL.warnings or '(none)'}\n")
    for key, task in chosen.items():
        d = should_convene(task)
        print(f"  {key:16s} {'CONVENE' if d.convene else 'DECLINED'}  score {d.score:.2f}")
    if not args.run:
        print(f"\n  {len(chosen)} session(s), ~$0.20 each — re-run with --run")
        return 0

    os.makedirs(TRACES, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    pool, rows = build_pool(), []
    print()
    for key, task in chosen.items():
        print(f"running {key}…", flush=True)
        trace = os.path.join(TRACES, f"{key}.jsonl")
        if os.path.exists(trace):
            os.remove(trace)
        result = Session(COUNCIL, pool, trace_path=trace,
                         config=SessionConfig(baseline_seat=BASELINE)).run(
            task.prompt, session_id=f"dogfood-{key}")
        write_report(result, os.path.join(REPORTS, f"{key}.html"))
        rows.append({
            "key": key, "ok": result.ok, "objections": len(result.objections),
            "movers": sum(1 for s in result.students if s.changed_position),
            "council": result.council_size,
            "minority": len(result.verdict.minority_report) if result.verdict else 0,
            "compliance": round(result.compliance_rate, 3),
            "cost": round(result.cost_est, 4),
            "answer": result.verdict.final_answer if result.verdict else "",
            "note": result.verdict.confidence_note if result.verdict else "",
            "dissent": [{"source": m.source, "substance": m.substance}
                        for m in (result.verdict.minority_report if result.verdict else [])],
        })
        print(f"  {key}: {rows[-1]['objections']} objections, "
              f"{rows[-1]['minority']} dissents, ${rows[-1]['cost']:.4f}")

    lines = ["# Quorum, deliberating on Quorum", "",
             "The project claims a council beats one model on high-stakes ambiguous",
             "judgement calls. These are the project's own unresolved design questions,",
             "which are exactly that shape — and the author has a stake in the answers,",
             "which is when a single confident reply is least trustworthy.", "",
             f"Council: {', '.join(s.model_id for s in COUNCIL.students)}  ",
             f"Arbiter: {COUNCIL.arbiter.model_id} (shares a lab with no student)", "",
             "| question | objections | moved | dissents kept | cost |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| `{r['key']}` | {r['objections']} | {r['movers']}/{r['council']} | "
                     f"{r['minority']} | ${r['cost']:.4f} |")
    lines.append("")
    for r in rows:
        lines += [f"## {r['key']}", "", f"**{QUESTIONS[r['key']].prompt}**", "",
                  "### What the council decided", "", r["answer"], "",
                  f"*{r['note']}*", ""]
        if r["dissent"]:
            lines += ["### Dissent it refused to drop", ""]
            lines += [f"- **[{d['source']}]** {d['substance']}" for d in r["dissent"]]
            lines.append("")
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\ntotal ${sum(r['cost'] for r in rows):.4f} — written to {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
