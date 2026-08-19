"""Regenerate the recorded sessions the offline test suite runs against.

    PYTHONPATH=src python3 tests/fixtures/_generate.py

Synthetic on purpose. These could have been captured from real runs, but real
transcripts are large, carry whatever the labs happened to say that day, and
would have to be scrubbed by hand every time they were refreshed. Generating
them from a scripted provider makes them small, deterministic, reviewable in a
diff, and secret-free by construction — there is no key anywhere in this path
and no network call in it.

The clock is fixed so regenerating produces a byte-identical file when nothing
has changed, which is what lets a fixture refresh show up in review as an
actual behavioural diff rather than a wall of new timestamps.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from quorum import (  # noqa: E402
    Council,
    ModelCost,
    ProviderPool,
    ScriptedProvider,
    Seat,
    Session,
    SessionConfig,
)
from quorum import trace as tr  # noqa: E402

TASK = "Should we rebuild the ingestion pipeline or refactor it in place?"

POSITIONS = {
    "model-alpha": "Rebuild the pipeline behind a feature flag.",
    "model-beta": "Refactor in place and keep the current contracts.",
    "model-gamma": "Run as-is and revisit after the next capacity review.",
}


def sheet(model: str) -> str:
    position = POSITIONS[model]
    return json.dumps({
        "position": position,
        "claims": [
            {"n": 1, "text": f"The ingest path's failure modes favour: {position}"},
            {"n": 2, "text": "Three years of edge cases live in the current code."},
            {"n": 3, "text": "Capacity headroom is the binding constraint, not design."},
        ],
        "assumptions": ["traffic stays within an order of magnitude"],
        "would_change_my_mind": ["a throughput ceiling we cannot lift by tuning"],
        "confidence": 0.68,
        "nuance": "",
    }, ensure_ascii=False)


def critique(labels: tuple[str, ...]) -> str:
    return json.dumps({"objections": [
        {
            "sheet": label,
            "claim_n": 1,
            "argument": (
                f"Sheet {label}'s first claim rests on a stability record that "
                "predates the new ingest sources, so the base rate it cites no "
                "longer describes the system under discussion."
            ),
        }
        for label in labels
    ]}, ensure_ascii=False)


def revision(model: str, changed: bool) -> str:
    data = json.loads(sheet(model))
    if changed:
        data["position"] = "Refactor in place, with a rebuild gated on capacity."
        data["confidence"] = 0.55
    data["changed_position"] = changed
    data["because"] = [{"critic": "A", "claim_n": 1}] if changed else []
    return json.dumps(data, ensure_ascii=False)


VERDICT = json.dumps({
    "final_answer": (
        "Refactor in place now and hold a rebuild behind an explicit capacity "
        "trigger. The participants agreed the current code carries hard-won "
        "edge cases and disagreed only on when capacity forces the question."
    ),
    "confidence_note": (
        "Moderately settled. The council did not converge on timing, and one "
        "seat held that no change is warranted yet."
    ),
    "minority_report": [
        {"source": "Student 3", "kind": "claim",
         "substance": "Running as-is remains defensible until capacity review."},
    ],
}, ensure_ascii=False)


def council() -> Council:
    return Council(
        students=(
            Seat("model-alpha", "labs", ModelCost(3.0, 15.0, 0.3, 3.75)),
            Seat("model-beta", "labs", ModelCost(3.0, 15.0, 0.3, 3.75)),
            Seat("model-gamma", "labs", ModelCost(0.3, 1.2, 0.03, 0.375)),
        ),
        arbiter=Seat("model-arbiter", "labs", ModelCost(5.0, 25.0, 0.5, 6.25)),
    )


def fixed_clock():
    counter = itertools.count()
    return lambda: 1_786_800_000.0 + next(counter)


def write(name: str, script: dict, session_id: str, *, repair: bool = True) -> str:
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        os.remove(path)
    pool = ProviderPool([ScriptedProvider(script, name="labs")])
    writer = tr.TraceWriter(path=path, clock=fixed_clock())
    Session(
        council(), pool, config=SessionConfig(repair_json=repair), writer=writer
    ).run(TASK, session_id=session_id)
    return path


def main() -> int:
    # 1. A clean run: every seat completes every round.
    clean = {
        "model-alpha": [sheet("model-alpha"), critique(("A", "B")),
                        revision("model-alpha", True)],
        "model-beta": [sheet("model-beta"), critique(("A", "B")),
                       revision("model-beta", False)],
        "model-gamma": [sheet("model-gamma"), critique(("A", "B")),
                        revision("model-gamma", False)],
        "model-arbiter": [VERDICT],
    }
    write("clean_session.jsonl", clean, "fixture-clean")

    # 2. The 2026-08-19 incident: the cheapest seat's round-2 critique arrives
    #    cut mid-object. Before repair-before-drop this cost the seat its whole
    #    round and left its round-1 position standing unopposed.
    truncated = critique(("A", "B"))
    truncated = truncated[: int(len(truncated) * 0.80)]
    malformed = dict(clean)
    malformed["model-gamma"] = [
        sheet("model-gamma"), truncated, revision("model-gamma", False)
    ]
    # Recorded with repair OFF, so the trace captures the failure as the
    # pre-repair engine saw it: the raw truncated text, verbatim, on a
    # discard event. That is what makes it a usable regression fixture —
    # recorded with repair on, it would preserve the recovery instead of the
    # thing to recover from, and replaying it would prove nothing.
    write("malformed_seat.jsonl", malformed, "fixture-malformed", repair=False)

    # 3. An interrupted run. Written by truncating a complete trace after the
    #    last round-1 event, which is what a killed process actually leaves on
    #    disk: no session_closed, no partial line.
    full = write("interrupted_session.jsonl", clean, "fixture-interrupted")
    with open(full, encoding="utf-8") as handle:
        lines = handle.readlines()
    keep, seen_round_2 = [], False
    for line in lines:
        if json.loads(line)["round"] > 1:
            seen_round_2 = True
            break
        keep.append(line)
    assert seen_round_2, "fixture never reached round 2; nothing to interrupt"
    with open(full, "w", encoding="utf-8") as handle:
        handle.writelines(keep)

    for name in ("clean_session.jsonl", "malformed_seat.jsonl",
                 "interrupted_session.jsonl"):
        path = os.path.join(HERE, name)
        with open(path, encoding="utf-8") as handle:
            print(f"  {name:28s} {sum(1 for _ in handle):3d} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
