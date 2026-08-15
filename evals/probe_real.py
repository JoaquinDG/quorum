"""The blinding metric, measured on real sessions.

    PYTHONPATH=src python3 evals/probe_real.py --check
    PYTHONPATH=src python3 evals/probe_real.py --run

`probe_eval.py` runs this against mocks and proves the harness works. This one
points it at every session archived on disk and produces the number the README
is supposed to publish.

It costs one call per session rather than one *session* per session, because
the probe reads a replayed trace. That is the replay-completeness rule earning
its keep: a claim about blinding can be re-measured against sessions that
finished weeks ago, with a different prober, without spending a cent on
re-running the debates.

The prober is deliberately not a participant in any archived session, and is
given every advantage — the roster, and the fact that authorship is one-to-one.
A probe that has to guess the candidate list would understate the leak, and
understating it is the one direction this measurement must not err in.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import ModelCost, OpenAICompatibleProvider, ProviderPool, Seat  # noqa: E402
from quorum import read_trace, replay  # noqa: E402
from quorum.probe import ProbeError, ProbeReport, probe_session  # noqa: E402

HERE = os.path.dirname(__file__)
SOURCES = [
    os.path.join(HERE, "..", "traces", "lineups", "*.jsonl"),
    os.path.join(HERE, "..", "tests", "fixtures", "real_session", "*.jsonl"),
]

# Google took no student seat in any archived session, so it is the one lab
# that cannot recognise its own writing here.
PROBER = Seat("gemini-3.1-pro-preview", "google", ModelCost(1.25, 10.0))


def load_sessions():
    seen, sessions = set(), []
    for pattern in SOURCES:
        for path in sorted(glob.glob(pattern)):
            try:
                session = replay(read_trace(path))
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop the run
                print(f"  skipping {os.path.basename(path)}: {str(exc)[:80]}")
                continue
            present = [s for s in session.students.values() if s.present]
            if len(present) < 2 or session.session_id in seen:
                continue
            seen.add(session.session_id)
            sessions.append((os.path.basename(path), session, present))
    return sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the blinding probe on archived sessions.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", default=os.path.join(HERE, "PROBE.md"))
    args = parser.parse_args(argv)

    sessions = load_sessions()
    print(f"{len(sessions)} archived session(s) with >=2 sheets\n")
    for name, session, present in sessions:
        roster = sorted({s.model_id for s in present})
        print(f"  {name:28s} {len(present)} sheets  chance {1/len(roster):.0%}  "
              f"{', '.join(roster)}")

    if not args.run:
        print(f"\n  prober: {PROBER.model_id} (held no seat in any of these)")
        print(f"  one call per session — re-run with --run")
        return 0

    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2

    pool = ProviderPool([OpenAICompatibleProvider(
        name="google", base_url="https://generativelanguage.googleapis.com",
        chat_path="/v1beta/openai/chat/completions", env_var="GEMINI_API_KEY")])

    report, rows = ProbeReport(), []
    print()
    for name, session, present in sessions:
        try:
            probe = probe_session(session, PROBER, pool, max_tokens=8192)
        except ProbeError as exc:
            print(f"  {name:28s} skipped: {exc}")
            continue
        report.results.append(probe)
        rows.append({"session": name, "sheets": len(present), **probe.to_dict()})
        mark = "abstained" if probe.abstained else f"{probe.hits}/{probe.attempts} correct"
        print(f"  {name:28s} {mark}")

    print(f"\n=== the blinding metric ===\n  {report.summary()}")
    excess = report.excess_over_chance
    if not report.measured:
        # With zero scored guesses accuracy and chance are both 0.0, so the
        # excess is 0.0 and the naive reading is "no leak". A run where every
        # call failed must never be reported as evidence of anything.
        verdict = ("NOT MEASURED — every prober call failed, so this is not "
                   "evidence that the blinding held")
    else:
        verdict = ("at or below chance — no leak detected from this prober" if excess <= 0
                   else "modestly above chance — a small, quantified leak" if excess <= 0.15
                   else "well above chance — the blinding leaks and the README must say so")
    print(f"  reading: {verdict}")

    lines = [
        "# Deanonymization probe — real sessions", "",
        f"**{report.summary()}**", "",
        f"Reading: {verdict}", "",
        f"Prober: `{PROBER.model_id}`, which held no seat in any session below.",
        "It was given the roster and told authorship is one-to-one — every",
        "advantage, because understating a leak is the one error this number",
        "must not make.", "",
        "Measured by replaying archived traces, so each session costs one call",
        "rather than a re-run.", "",
        "| session | sheets | correct | chance | accuracy |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['session']}` | {r['sheets']} | "
            + ("abstained" if r["abstained"] else f"{r['hits']}/{r['attempts']}")
            + f" | {r['chance']:.0%} | "
            + ("—" if r["abstained"] else f"{r['accuracy']:.0%}") + " |"
        )
    lines += [
        "", "## What this does and does not show", "",
        "It measures whether **this** prober can identify authorship from",
        f"**these** sheets. A stronger prober may do better. With",
        f"{report.attempts} guesses the error bars are wide, and a single",
        "run cannot separate a small leak from sampling noise.", "",
        "It does not measure whether the *protocol* is blind in general — only",
        "that the schema removed enough signal to defeat one capable reader on",
        "the sessions to hand.", "",
    ]
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump({"summary": report.to_dict(), "sessions": rows}, handle, indent=2)
    print(f"\n  written to {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
