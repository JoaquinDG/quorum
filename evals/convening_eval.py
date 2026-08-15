"""Scenario eval for the convening rule, on a mixed workload.

Unit tests check that the rule does what it says. This checks that what it
says is the right thing on a day's worth of real-looking traffic — the
acceptance criterion is a *rate* (<10% convened), and a rate cannot be
asserted one task at a time.

The workload is deliberately shaped like real traffic rather than like a
demo: mostly extraction, summarisation, code and Q&A, with a handful of
genuine judgement calls. Two categories exist specifically to make the eval
able to fail:

- **hard_but_verifiable** — high complexity, real difficulty, checkable
  answer. A rule that keys on difficulty alone convenes these and blows the
  budget. This is the failure mode the `verifiable` gate exists for.
- **ambiguous_but_trivial** — the surface markers of a judgement call
  ("should we…") on a question nobody needs a council for. A rule that keys
  on phrasing alone convenes these.

Each task carries the answer it *should* get, so the eval reports precision
and recall rather than just the headline rate. A rule that convenes on nothing
would pass a bare rate check and be useless.

    PYTHONPATH=src python3 evals/convening_eval.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum.convening import CONVENE_CONSERVATIVE, CONVENE_DEFAULT, Task, should_convene  # noqa: E402

# (category, expected_convene, Task)
WORKLOAD: list[tuple[str, bool, Task]] = [
    # -- routine: the bulk of any real workload ---------------------------
    ("routine", False, Task("Extract the plan names, prices and seat limits from these five competitor pricing pages.", "extraction", 0.3)),
    ("routine", False, Task("Summarise this 40-page vendor contract for a non-lawyer.", "summarization", 0.45)),
    ("routine", False, Task("Translate the onboarding emails into German and Japanese.", "translation", 0.3)),
    ("routine", False, Task("Classify these 2,000 support tickets by product area.", "classification", 0.35)),
    ("routine", False, Task("Write a SQL query returning weekly active users by cohort.", "code_generation", 0.4)),
    ("routine", False, Task("Fix the failing test in test_parser.py.", "code_generation", 0.35)),
    ("routine", False, Task("Convert this CSV export into the internal JSON schema.", "formatting", 0.25)),
    ("routine", False, Task("What is the current rate limit on the ingest endpoint?", "qa", 0.15)),
    ("routine", False, Task("Draft release notes from the last 30 merged pull requests.", "summarization", 0.3)),
    ("routine", False, Task("Rename the `usr_id` column to `user_id` across the codebase.", "code_generation", 0.2)),
    ("routine", False, Task("Count how many customers churned in Q3 by plan tier.", "arithmetic", 0.2)),
    ("routine", False, Task("Parse these server logs and list the top 20 slowest endpoints.", "extraction", 0.3)),
    ("routine", False, Task("Review this pull request for correctness and style.", "code_review", 0.45)),
    ("routine", False, Task("Write unit tests for the retry helper.", "code_generation", 0.35)),
    ("routine", False, Task("Summarise yesterday's incident thread into a timeline.", "summarization", 0.35)),
    ("routine", False, Task("Format this changelog as Markdown with consistent headings.", "formatting", 0.15)),
    ("routine", False, Task("Extract every email address mentioned in this thread.", "extraction", 0.2)),
    ("routine", False, Task("Translate the error strings into Spanish.", "translation", 0.25)),
    ("routine", False, Task("Classify these product reviews as positive, negative or mixed.", "classification", 0.25)),
    ("routine", False, Task("Retrieve the deploy history for the billing service.", "retrieval", 0.15)),
    ("routine", False, Task("Write a regex matching our internal ticket IDs.", "code_generation", 0.3)),
    ("routine", False, Task("Summarise this research paper's method section.", "summarization", 0.4)),
    ("routine", False, Task("Deserialise these protobuf payloads into readable JSON.", "formatting", 0.25)),
    ("routine", False, Task("What is the p99 latency on checkout this week?", "qa", 0.2)),
    ("routine", False, Task("Draft a reply to this customer asking about SSO support.", "summarization", 0.3)),
    ("routine", False, Task("Validate the schema on these 500 event payloads.", "extraction", 0.3)),
    ("routine", False, Task("List all TODO comments added in the last release.", "extraction", 0.2)),
    ("routine", False, Task("Generate seed data for the staging environment.", "code_generation", 0.25)),

    # -- hard, but checkable: the trap for difficulty-only rules ----------
    ("hard_but_verifiable", False, Task("Find the race condition causing intermittent duplicate charges in the payment worker and write a failing test that reproduces it.", "code_generation", 0.85)),
    ("hard_but_verifiable", False, Task("Optimise this query plan; it does a full scan on a 400M row table under concurrent writes.", "code_generation", 0.8)),
    ("hard_but_verifiable", False, Task("Derive the closed-form cost of our current sharding scheme at 10x write volume.", "arithmetic", 0.75)),
    ("hard_but_verifiable", False, Task("Reconcile these two ledgers and explain every discrepancy over $100.", "extraction", 0.7)),
    ("hard_but_verifiable", False, Task("Port the tokenizer from Rust to Go with identical output on the full test corpus.", "code_generation", 0.8)),

    # -- ambiguous phrasing, trivial stakes: the trap for phrasing rules --
    ("ambiguous_but_trivial", False, Task("Should we use tabs or spaces in the new repo?", "policy", 0.15)),
    ("ambiguous_but_trivial", False, Task("Should I name this function `fetchUser` or `getUser`?", "policy", 0.1)),
    ("ambiguous_but_trivial", False, Task("Which approach is better for this loop, map or a comprehension?", "code_review", 0.2)),
    ("ambiguous_but_trivial", False, Task("Do we want the button on the left or the right of the modal?", "policy", 0.2)),

    # -- unclassified: the rule must fail closed --------------------------
    ("unclassified", False, Task("Handle the thing we discussed.")),
    ("unclassified", False, Task("Follow up on this.")),

    # -- genuine judgement calls: what the council is for -----------------
    ("judgment", True, Task(
        "Our ingestion pipeline is three years old and increasingly slow. Should we rebuild it "
        "on a streaming architecture this year, or refactor it in place? The rebuild is a "
        "one-way door on the storage format and the roadmap has no slack.",
        "architecture", 0.85)),
    ("judgment", True, Task(
        "Should we move from seat-based pricing to usage-based pricing next quarter? Our "
        "largest accounts would pay less and our long tail would pay more, and we cannot "
        "easily reverse the contract terms once signed.",
        "pricing", 0.9)),
    ("judgment", True, Task(
        "We can either hire two senior engineers or four juniors with the same budget. Which "
        "is the better bet given an 18-month runway and a roadmap that is mostly unknowns?",
        "strategy", 0.8)),
    ("judgment", True, Task(
        "Should we deprecate the v1 API this year? Roughly 6% of revenue still depends on it, "
        "migration is a multi-quarter compliance risk for those customers, and maintaining it "
        "is costing us a third of the platform team.",
        "risk", 0.85)),
]


def evaluate(policy, label: str) -> dict:
    convened = []
    false_positive = []
    false_negative = []
    for category, expected, task in WORKLOAD:
        decision = should_convene(task, policy)
        if decision.convene:
            convened.append((category, task, decision))
        if decision.convene and not expected:
            false_positive.append((category, task, decision))
        if expected and not decision.convene:
            false_negative.append((category, task, decision))
    rate = len(convened) / len(WORKLOAD)
    return {
        "label": label,
        "rate": rate,
        "convened": convened,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> int:
    print(f"convening eval — {len(WORKLOAD)} tasks in a mixed workload\n")
    failures = 0

    for policy, label in ((CONVENE_DEFAULT, "default"), (CONVENE_CONSERVATIVE, "conservative")):
        result = evaluate(policy, label)
        print(f"policy={label}")
        print(f"  convened: {len(result['convened'])}/{len(WORKLOAD)} "
              f"({result['rate']:.1%})")
        for category, task, decision in result["convened"]:
            print(f"    [{category}] {task.prompt[:70]}…")
            print(f"        {decision.reason}")

        if result["false_positive"]:
            failures += 1
            print("  FALSE POSITIVES (convened on a task that did not need it):")
            for category, task, decision in result["false_positive"]:
                print(f"    [{category}] {task.prompt[:70]}… — {decision.reason}")
        if result["false_negative"]:
            failures += 1
            print("  FALSE NEGATIVES (declined a genuine judgement call):")
            for category, task, decision in result["false_negative"]:
                print(f"    [{category}] {task.prompt[:70]}… — {decision.reason}")
        print()

    default = evaluate(CONVENE_DEFAULT, "default")
    print("--- acceptance criteria ---")

    ac_rate = default["rate"] < 0.10
    print(f"  [{'PASS' if ac_rate else 'FAIL'}] <10% of a mixed workload convenes "
          f"({default['rate']:.1%})")
    if not ac_rate:
        failures += 1

    judgment_total = sum(1 for c, e, _ in WORKLOAD if e)
    judgment_hit = judgment_total - len(default["false_negative"])
    ac_recall = judgment_hit == judgment_total
    print(f"  [{'PASS' if ac_recall else 'FAIL'}] every genuine judgement call convenes "
          f"({judgment_hit}/{judgment_total})")
    if not ac_recall:
        failures += 1

    ac_precision = not default["false_positive"]
    print(f"  [{'PASS' if ac_precision else 'FAIL'}] no routine task convenes "
          f"({len(default['false_positive'])} false positives)")

    # A rate under 10% is trivially satisfiable by never convening. This is
    # the check that keeps the headline number honest.
    ac_nonzero = len(default["convened"]) > 0
    print(f"  [{'PASS' if ac_nonzero else 'FAIL'}] the rule is not simply 'never' "
          f"({len(default['convened'])} convened)")
    if not ac_nonzero:
        failures += 1

    print()
    print("FAILURES:" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
