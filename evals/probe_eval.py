"""Run the deanonymization probe over a batch of sessions and publish the number.

    PYTHONPATH=src python3 evals/probe_eval.py

Offline, this measures the *harness*, not the blinding: `MockProvider` writes
sheets whose persona is assigned by model id and which carry no trace of that
id, so the prober genuinely cannot tell them apart and lands near chance by
construction. That is the correct offline result and it is worth almost
nothing as evidence — a mock cannot leak a fingerprint it never had.

The number that belongs in the README comes from running this against real
providers, where the sheets are written by models with real stylistic habits.
The harness is here so that run is one command away, and so the metric exists
before there is any temptation to skip it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import (  # noqa: E402
    ModelCost,
    Seat,
    convene,
    demo_council,
    mock_pool,
)
from quorum.probe import probe_all  # noqa: E402
from quorum.providers.base import MockProvider  # noqa: E402

SESSIONS = 24

QUESTIONS = [
    "Should we rebuild the ingestion pipeline or refactor it in place, given the roadmap?",
    "Should we move to usage-based pricing next quarter, given the contract risk?",
    "Do we hire two seniors or four juniors on an 18-month runway?",
    "Should we deprecate the v1 API this year, given the compliance exposure?",
    "Do we open-source the core engine, given what competitors could do with it?",
    "Should we take the single-tenant deal that is worth 20% of ARR?",
]


def main() -> int:
    council = demo_council()
    prober = Seat("probe-critic", "probelab", ModelCost(1.0, 5.0))

    pool = mock_pool(council)
    prober_provider = MockProvider()
    prober_provider.name = prober.provider
    pool._providers[prober.provider] = prober_provider  # noqa: SLF001

    results = []
    for i in range(SESSIONS):
        question = QUESTIONS[i % len(QUESTIONS)]
        results.append(
            convene(question, council, mock_pool(council), session_id=f"probe-{i}")
        )

    report = probe_all(results, prober, pool)

    print("deanonymization probe")
    print(f"  prober:   {prober.model_id} (held no seat)")
    print(f"  roster:   {', '.join(s.model_id for s in council.students)}")
    print(f"  sessions: {report.scored_sessions} scored, {report.abstentions} abstained")
    print()
    print(f"  {report.summary()}")
    print()

    excess = report.excess_over_chance
    if excess <= 0:
        verdict = "at or below chance — no leak detected from this prober"
    elif excess <= 0.15:
        verdict = "modestly above chance — a small, quantified leak"
    else:
        verdict = "well above chance — the blinding is leaking and the README must say so"
    print(f"  reading:  {verdict}")
    print()
    print("  NOTE: run against mock providers. Mock sheets carry no model-specific")
    print("  style, so a near-chance result here is a property of the mock, not")
    print("  evidence about the schema. Real numbers need real providers.")

    # The probe never "fails" — an above-chance result is a finding to publish,
    # not a build break. It fails only if it could not measure anything.
    if not report.attempts:
        print("\nFAIL: no guesses were scored")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
