# Per-seat compliance eval: specification

| | |
|---|---|
| **Status** | `SPEC`. Nothing here is implemented beyond what already runs free in mock mode |
| **Rule it serves** | no seat gets a chair without a per-seat compliance eval, and the probe re-runs on any lineup change |
| **Blocks** | the Grok seat, the weak-seat replacement, and part of D6 |

## Why this gate exists

The lineup experiment found the sharpest result in the repo: **upgrading the
weak seat fixed compliance outright.** `deepseek-chat` ran at 69–82%
compliance with 1–3 repairs every session; `deepseek-v4-pro` scored 100% with
zero repairs and raised more objections than any other three-lab lineup
(`evals/LINEUPS.md`).

That is the whole argument for this gate. A seat that cannot hold the format
does not merely score badly, it **costs money twice** (every repair is a
billed call whose output is discarded) and it **distorts the protocol**: a
sheet rejected in round 1 excludes that student from every subsequent round,
so the council silently runs a seat short.

It is also why the bar cannot be waived for an interesting model. A seat is
admitted on evidence or not at all.

## What a seat must pass

Four criteria. **All four, or no chair.**

### C1. Format-contract compliance rate ≥ 90%

The bar is the one already published in the README's metric table, and the
metric already exists: `result.compliance_rate`, computed per model response
across rounds 1 to 3.

- Denominator: responses that returned text.
- **Provider errors are excluded.** A 503 is availability, not compliance, and
  conflating them is a defect this repo already fixed once. They are reported
  in their own counter (C2).
- Multi-sentence claims count as `compliance_warnings`, not failures. The
  contract is strict about meaning and tolerant of transport.

### C2. Dropout behaviour

Compliance rate alone hides the failure that matters most, because a seat that
dies in round 1 has a small denominator and can look fine.

- **Round-1 absence rate must be 0** across the eval. A malformed opening sheet
  costs the seat every later round; this is the single most expensive way a
  seat can fail.
- No `student_absent` event that is not explained by a provider error.
- The seat must still be answering in round 3. A seat that stops participating
  after being objected to is not a council member.
- `ProviderTruncated` is recorded separately and **does not count against the
  seat**: truncation is our token budget, not the model's compliance. Budgets
  in this repo predate reasoning models and a truncated sheet has already once
  been misread as model failure.

### C3. Repair-cost share ≤ 10% of session cost

`repair_cost_est` divided by session `cost_est`, from the `attempt_discarded`
events. Repairs were invisible until those events existed, and a session that
burned extra calls reported a *lower* cost than a clean one.

Ten percent is a proposal, not an inherited number. The observed range is 0%
(`deepseek-v4-pro`) to sessions where repairs were 42% of the bill. It should
be ratified or changed before the first seat is measured against it, and the
decision recorded here.

### C4. Blinding not made materially worse

Required by the standing rule that the probe re-runs on any lineup change,
and by a finding that cuts against C1: **more capable models have more
distinctive voices**, so upgrading a seat to fix compliance may cost blinding.
The two goals pull against each other, and a seat must not be admitted on C1
while quietly failing the protocol's blinding claim.

- Re-run `probe_real.py` over the new lineup's archived traces. This replays
  traces and costs one call per session, not a re-run.
- Record the new deanonymisation accuracy against chance, and against the
  current figure (41.9% vs 32.3% chance, n=31 guesses).
- This is a **recorded observation, not a hard fail.** At these sample sizes
  the probe cannot separate a small leak from noise, and a gate that pretends
  otherwise would be inventing precision. A material regression is escalated
  to a judgement call, not auto-rejected.

## How many sessions

Each session yields roughly **three scored responses per seat** (opening
sheet, critique, revision).

**Mock first, and it is free.** Every candidate runs the full mock path before
any billed call:

- schema parses, all three rounds, no repairs from harness-shaped faults
- the seat survives a simulated round-2 dropout and a reduced council
- the model id resolves against the live config (this is a preflight gate, P2)

A candidate that fails in mock never reaches a billed call.

**Live: 10 sessions.** This number is doing real work, so here is the
arithmetic rather than an assertion. With ~3 responses per session, and a
perfect record, the one-sided 95% lower bound is:

| live sessions | responses | lower bound on true compliance |
|---|---|---|
| 1 | 3 | 37% |
| 2 | 6 | 61% |
| 5 | 15 | 82% |
| 7 | 20 | 86% |
| **10** | **30** | **90.5%** |
| 15 | 45 | 94% |

**Ten is the smallest n at which a perfect record actually certifies the 90%
bar.** Below it, a passing seat is a seat that has not yet been caught. This
is also the honest correction to `evals/LINEUPS.md`, which measured each
lineup once and says so: n=1 is a hypothesis, and a 100% compliance score at
n=1 has a lower bound of 37%.

A seat may be **failed** on far less. One round-1 absence, or two format
failures, ends the eval early: rejection needs much less evidence than
admission, and stopping early is where the budget is saved.

Questions must be held constant across candidates, drawn from the judgement
task set, and fixed before the first run so a seat cannot be helped by an
easier question.

## What it costs

Per candidate seat, using the measured $0.20 per session and the same caveat
as everywhere else: these are `PRICES` estimates, not bills.

| | |
|---|---|
| mock qualification | $0 |
| 10 live sessions | ~$2.00 |
| probe replay (C4), 10 sessions | ~$0.10 |
| **per candidate seat** | **~$2.10** |
| early rejection (fails by session 2) | ~$0.40 |

Against the three queued items:

| item | candidates | estimated |
|---|---|---|
| Grok seat | 1 | ~$2.10 |
| weak-seat replacement re-certified at n=10 | 1 | ~$2.10 |
| **total to unblock both seat decisions** | | **~$4.20** |

**D6 is only partly unblocked by this.** The residual arbiter-quality question
is not a compliance question. Cost has already stopped being a reason to pick
an arbiter (five arbiters within a 7% spread, `LINEUP_ARBITER_QUALITY.md`), and
what remains is whether the *right* dissent survives into the minority report.
That needs the benchmark's rubric machinery pointed at verdicts rather than
answers, with a judge that took no part. This spec certifies that an arbiter
can hold the format. It cannot tell you whether it arbitrates well, and it
should not be cited as if it could.

## Where results get recorded

**`evals/LINEUPS.md` is a frozen path** (DOI 10.5281/zenodo.21962850), add-only,
never rewritten. Seat evals do not touch it.

- Per-seat results: **`evals/SEATS.md`** and `evals/SEATS.json`, one row per
  candidate, appended, never edited in place. A failed candidate stays on the
  page: the record of what was rejected and why is the point.
- Traces: `traces/seats/<candidate>-<n>.jsonl`, retained. Per preflight gate
  P6, an eval that keeps no trace has bought nothing durable.
- Probe output (C4): appended to `evals/PROBE.md` under a dated heading.
- A seat that passes is recorded with its **sample size and lower bound**, not
  just its percentage. "100% compliance" without an n is the claim this spec
  exists to stop being made.
- The decision to seat or not seat is recorded in the vault decision log,
  citing the row.

## Implementation note

Most of this already exists. `lineup_eval.py` computes `compliance`, `repairs`,
`repair_cost` and absences per session, and has a `--check` mode that prints the
plan and cost without spending. The work is a candidate-seat runner over N
sessions with early-stop, an appending writer for `SEATS.md`, and trace
retention. **Nothing here should be built past the free mock path until the
benchmark question is settled**: the seat evals and the benchmark draw on the
same budget, and the benchmark is the one with a published claim resting on it.
