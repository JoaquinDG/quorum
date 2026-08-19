# Post-mortem: the 2026-08-19 live benchmark attempt

| | |
|---|---|
| **Status** | `FINAL` |
| **Subject** | the first live run of `evals/benchmark_eval.py --real` |
| **Outcome** | no result; ~$7.74 spent; 11 of 60 task/arm pairs scored |
| **Primary evidence** | `evals/BENCHMARK_REAL.md`, committed alongside this note |

The run is committed rather than discarded. It carries its own
`NOT A RESULT` banner, and a failure the project spent real money to
find is worth more on disk than in a memory.

---

## What the run reported

Three arms over 20 judgement tasks. 49 of 60 task/arm pairs produced no
scored answer. Arm costs as reported:

| arm | reported cost | per task |
|---|---|---|
| `quorum` | $4.0577 | $0.2029 |
| `single` | $0.7486 | $0.0374 |
| `self_critique` | $2.9349 | $0.1467 |
| **total** | **$7.7412** | |

The stated common cause was "a provider out of credit, or a model id that
has been retired", affecting `quorum`, `self_critique` and `single`.

## What the record actually supports

**The council arms ran. They were not the failure.**

`quorum` billed $0.2029 per task. An independently measured session with
the same arbiter (`gemini-3.1-pro-preview`) cost **$0.2001** in
`evals/LINEUP_ARBITER_QUALITY.md`. Those are 1.4% apart. A council arm
that had failed on 17 of 20 tasks could not have billed twenty full
sessions' worth of tokens. The same holds for the other two arms:
`self_critique` came in at 3.9x `single`, which is the shape of a
working two-call arm, not a broken one.

**The failure pattern points at the judge, not at the council providers.**

Sixteen of the twenty tasks show `err` in *all three arms at once*. The
three arms do not share providers: `quorum` spans anthropic, openai,
deepseek and google; `single` and `self_critique` are anthropic. For a
credit or model-id fault to produce that pattern, four labs would have
had to fail on the same sixteen tasks and recover on the same four.

One component is common to all three arms: the judge, `kimi-k2.6` on
moonshot. And the run's own completeness test (`benchmark_eval.py`)
marks a pair incomplete when it has an error **or no scores**:

    if o.arms[arm].error or not o.arms[arm].scores

A judge that fails on a task empties `.scores` for all three of that
task's arms simultaneously. That is precisely the observed pattern.

> **This is inference from the cost table and the failure shape, not a
> reading of a log.** No trace survived the run (see below), so the judge
> hypothesis is the best explanation available from the record rather
> than a confirmed root cause. Preflight gate P1 exists to make the next
> run answer this question directly instead of by arithmetic.

## The four findings that change the plan

**F1. The benchmark persists nothing.** `run_quorum` calls
`Session(...).run()` with no trace path, and `run_benchmark` accumulates
outcomes in memory and writes only the summary Markdown at the end. The
$7.74 bought roughly sixty generated answers. Eleven were scored. Zero
were retained. There is nothing to audit, nothing to re-score, and
nothing to resume from.

**F2. The engine already has the primitive the benchmark did not use.**
`quorum.resume` continues an interrupted debate from its own trace,
explicitly so that "a session that dies in round 3" does not pay twice.
The benchmark harness never wires it up. The capability gap is in the
eval harness, not in the protocol.

**F3. The `--real` guard checks the wrong thing.** It verifies that five
environment variables are *non-empty* and proceeds. A present key with no
credit, or a valid key naming a retired model, passes this gate and fails
at the first billed call. Presence is not credit, and neither is a
resolvable model id.

**F4. There is no cost ceiling anywhere in the harness.** Neither
`run_benchmark` nor the CLI accepts a per-session or total budget. The
run stopped when the tasks ran out, not when a limit was reached. A fault
that had cost more per task would have cost more in total, with no
mechanism to notice.

## What this says about the result we might get

On the four tasks where scoring worked, `quorum` scored below both other
arms every time (0.34 / 0.64 / 0.66, against 0.90 to 1.00), at 5.4x the
cost of `single`.

**This is not a finding.** Four tasks, selected by whichever ones the
judge happened to survive, is not a sample. But it is the only live
signal that exists, and it points the wrong way for the council. The
disclosure pack must therefore be written to be publishable if the valid
run reproduces this. Per the project's own rule, the result goes out
whatever it says, and the claims never depended on Quorum winning.

## Corrective actions

| # | Action | Where |
|---|---|---|
| 1 | Hard preflight gate before any billed call | `evals/PREFLIGHT.md` (gates P1 to P7) |
| 2 | Credit and model-id verification per lab, live, cheap | gate P1 |
| 3 | Full pair-completeness dry run in mock mode | gate P3 |
| 4 | Per-session and total cost ceilings | gate P5 |
| 5 | Per-task checkpointing so partial failure never orphans spend | gate P6 |
| 6 | Trace persistence for every arm of every task | gate P6 |
| 7 | Judge health treated as a first-class arm, not as infrastructure | gates P1, P3 |

Gates P1 to P7 are the checklist that would have caught this run. Every
one of them is free to execute except P1, which is bounded at a few cents.
