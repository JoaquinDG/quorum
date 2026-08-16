# Contributing

The most useful thing you can do to this project is break it.

Quorum's claims are all of the form *"this mechanism does what it says"*, and
every one of them was written by the same person who wrote the mechanism. That
is the weakness. The list below is ordered by how much a contribution would
change what the README is allowed to claim — not by how much work it is.

## What would help most

### 1. Attack the rubrics

`evals/judgment_tasks.json` decides the benchmark's outcome before a single
model is called. It has had **one author and one adversarial model reviewer**
(`evals/RUBRIC_REVIEW.md`), and that review found 44 of 120 criteria mistagged
— including one that sat in the "neutral" column of every task and handed the
council a free point.

A human who thinks the protocol is wrong is worth more than another model pass.
Specifically:

- Which criteria still favour deliberation without being tagged?
- Which are unmeasurable — a judge could not apply them twice the same way?
- Which task is missing the consideration that would actually decide it?

Seven of twenty tasks still have no neutral majority, and that may be
unfixable: what makes an answer good on an ambiguous question is largely what
a debate produces. If you can design a criterion that is genuinely reachable by
a single confident answer *and* discriminating, that is a real contribution.

### 2. Run the probe with your own models and report the number

```bash
PYTHONPATH=src python3 evals/probe_real.py --run
```

The published blinding figure is **+9.7 points over chance**, from one prober
across ten sessions — 31 guesses, wide error bars. It reads as a leak, and the
distribution hints that stronger models have more identifiable voices, but n=10
cannot settle that.

A different prober, or a council of labs not used here, is a genuinely new data
point. **Report it whatever it says.** A number that makes the blinding look
worse is more valuable than one that flatters it, because the leak is currently
a single observation.

### 3. Break a protocol invariant

Five structural guarantees, each with a test in `tests/test_invariants.py`:

| invariant | what it forbids |
| --- | --- |
| independence | a round-1 prompt seeing any peer content |
| schema blinding | model names, provider names, seat numbers or `nuance` reaching a critic |
| no self-grading | an arbiter that debated, or a student critiquing itself |
| fail-closed | a malformed reply being coerced into a valid one |
| fixed rounds | any path that loops until agreement |

Finding a route around one of these is the highest-value bug report this
project can receive. A failing test is a complete contribution — no fix needed.

### 4. Add a provider adapter

Two vendors have been wired since launch and **both surfaced a URL bug**: a
doubled `/v1`, and a vendor serving `chat/completions` under a different path
entirely. There are more of these.

`OpenAICompatibleProvider` takes a `base_url`, an `env_var` and a `chat_path`,
so most vendors need no code — just an entry and a test. If yours does need
code, that is the interesting case.

### 5. Contribute a judgment task you expect the council to lose

Tasks where deliberation should *not* help are underrepresented and are the
ones that make the benchmark honest. A task with a defensible answer that a
single strong model reaches directly, where three models would only add
hedging, is worth more than another task the council handles well.

## Ground rules

**Every claim is runnable or labelled.** A number in the README must trace to a
command in this repository or say plainly that it is an estimate. If you add a
measurement, add the command that reproduces it.

**Mock runs are never results.** Anything produced against `MockProvider` is
stamped as such and must stay stamped. `BenchmarkReport.comparable` refuses to
print a leaderboard when an arm did not answer — do not route around it.

**Report findings that hurt.** The blinding leak, the cost overrun, the rubric
tilt and the metric that got deleted are all in the README because measuring
them was the point. A pull request that quietly improves a number without
saying what changed is worse than one that reports a regression.

**Paths cited by a published paper are frozen.** The white paper
([DOI](https://doi.org/10.5281/zenodo.21962850)) references
`tests/fixtures/real_session/live_mixed_lab.jsonl`,
`tests/fixtures/real_session/live_three_lab.jsonl` and `evals/LINEUPS.md` by
path. They may be added to, never renamed or deleted. If one must move, leave a
stub at the old path pointing at the new one.

**Keys live in your environment, never in the repository.** Nothing here reads
a key from a file, and no key is ever written into a trace or a report —
`tests/test_http.py` asserts it.

## Running things

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # everything, offline, no keys
PYTHONPATH=src python3 evals/convening_eval.py         # rate over a mixed workload
PYTHONPATH=src python3 examples/quickstart.py          # a full session against mocks
python3 replay.py traces/quickstart.jsonl              # rebuild it from the trace alone
```

Anything with `--real` or `--run` spends money. Every one of those scripts has
a dry-run mode that prints the plan and the estimated cost first; use it.
