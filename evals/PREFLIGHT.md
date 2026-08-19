# PREFLIGHT: the gate before any billed benchmark call

| | |
|---|---|
| **Status** | `FINAL` |
| **Applies to** | `evals/benchmark_eval.py --real`, and any future eval that spends money |
| **Origin** | the 2026-08-19 attempt: see `evals/POSTMORTEM_2026-08-19_benchmark.md` |

Seven gates. **P2, P3, P4 and P6 are free.** P1 costs a few cents. Nothing
below authorises the full run on its own: the sequence is P1 to P6, then the
pilot, then P7, then the full run.

A gate that cannot be evidenced is a gate that failed. "It looked fine" is
not a pass.

---

## P1. Provider credit verified per lab, live

**Not** "the environment variable is set". The current guard checks only that
five variables are non-empty, and that is exactly what let the failed run
start. A present key with no credit passes it.

For each of the five labs (`anthropic`, `openai`, `deepseek`, `google`,
`moonshot`): issue one minimal real completion, `max_tokens` at its floor,
and require a 200 with a non-empty body.

- [ ] `anthropic`: billed call returned content
- [ ] `openai`: billed call returned content
- [ ] `deepseek`: billed call returned content
- [ ] `google`: billed call returned content
- [ ] `moonshot`: billed call returned content

**The judge's lab is a gate, not infrastructure.** `moonshot` carries no arm
and is easy to read as plumbing. It is the single point whose failure voids
all three arms of a task at once, which is the most likely explanation for
the 49 missing pairs. It gets the same scrutiny as a seated model.

Total cost of this gate: under $0.02.

## P2. Model ids resolved against the live config

Every id in `benchmark_eval.py` must be echoed back by the lab that serves
it. A retired id is a 404 at the first billed call, not at import.

- [ ] `claude-sonnet-5` (student, anthropic)
- [ ] `gpt-5.1` (student, openai)
- [ ] `deepseek-v4-pro` (student, deepseek)
- [ ] **`gemini-3.1-pro-preview` (arbiter, google)**. The arbiter changed:
      the default lineup's same-lab arbiter is no longer what runs here
- [ ] `claude-opus-5` (single-model arm and pinned baseline seat, anthropic)
- [ ] `kimi-k2.6` (judge, moonshot)

- [ ] The resolved ids are recorded in the run manifest, not just checked.
      A run that cannot say which model answered cannot be published.

## P3. Full pair-completeness dry run, mock mode, free

Run the entire task set offline first and require **60 of 60** task/arm pairs
to produce a scored answer:

```bash
PYTHONPATH=src python3 evals/benchmark_eval.py
```

- [ ] Exit status 0 (the harness already fails non-zero on any incomplete pair)
- [ ] `all arms produced scored answers on all tasks` printed
- [ ] 20 tasks loaded, 3 arms each, 0 errors
- [ ] The judge arm scored every task (this is what broke live; it must be
      shown working offline before it is trusted online)

This gate is free and catches every harness-shaped fault. It does not and
cannot catch a provider-shaped fault, which is what P1 and P2 are for.

*Last run 2026-08-19 on this branch: 60/60 pairs scored, exit 0, alongside
490/490 unit tests. The harness is not what is blocking the live run.*

## P4. Rubric state pinned and disclosed

The rubric has already changed underneath a published review once. See
`evals/RUBRIC_WORKSHEET.md`.

- [ ] The commit SHA of `evals/judgment_tasks.json` is recorded in the manifest
- [ ] The criterion count is recorded (currently **99**, across 20 tasks)
- [ ] The neutral-majority partition is recorded (currently **13 tasks**;
      7 lack a neutral-criteria majority)
- [ ] Any accepted worksheet decisions are merged *before* the run, not during
- [ ] `evals/RUBRIC_REVIEW.md`'s headline figures have been reconciled with the
      rubric actually being run

## P5. Cost ceilings set, both levels

Neither ceiling exists in the harness today. Until they do, the ceiling is
enforced provider-side and by the operator, and that must be stated rather
than assumed.

- [ ] **Per-session ceiling**: $0.60. Roughly 3x the measured cost of a
      single council session; a session that exceeds it is misbehaving, not
      merely expensive.
- [ ] **Per-run total ceiling**: $15 for the full run, $3 for the pilot.
- [ ] Provider-side budget limits are set at every lab, and are the real
      backstop (runbook rule 7: wrapper prices are estimates, provider caps
      are not).
- [ ] The run aborts on breach rather than warning and continuing.

## P6. Checkpoint and resume, so partial failure never orphans spend

The failed run generated roughly sixty answers and retained none. This is the
gate that turns a repeat of it from a $7.74 loss into a $0 loss.

- [ ] Every arm of every task writes its trace to disk **before** the next task
      starts
- [ ] Every arm's raw answer text is persisted, whether or not it scored
- [ ] The run is restartable from the last completed task and skips work
      already on disk
- [ ] Scoring is separable from generation: a judge failure must be re-runnable
      against retained answers **without regenerating a single arm**
- [ ] `quorum.resume` is wired in for the council arm (the engine already has
      it; the harness does not use it)

Acceptance: kill the run at task 7 and restart it. It must resume at task 8
and bill nothing for tasks 1 to 7.

## P7. Disclosure fields captured per task

Captured at run time. A field reconstructed afterwards is not a disclosure,
and anything missing here cannot be added honestly later.

**Per run:**

- [ ] engine commit SHA
- [ ] `judgment_tasks.json` SHA and criterion count
- [ ] the `PRICES` table as used, verbatim, with each entry marked verified or estimated
- [ ] resolved model ids for all six seats
- [ ] `is_mock` false, and the arbiter/judge non-participation checks passed
- [ ] wall-clock start and end

**Per task:**

- [ ] task key and partition (neutral-majority / contested)
- [ ] each arm's full answer text
- [ ] each arm's per-criterion score, with that criterion's tag at run time
- [ ] each arm's `cost_est`, tokens in and out
- [ ] trace path for the council arm
- [ ] blinding seed / shuffle seed
- [ ] absences, retries, discarded attempts
- [ ] error text where an arm failed, verbatim

**After the pilot, before the full run:**

- [ ] **Reconcile `cost_est` against the actual provider invoices.** Every
      cost figure this harness produces is an estimate derived from the
      `PRICES` table, not a bill. The $7.74 attributed to the failed run is
      itself an estimate. One entry in this table has already been caught
      understating a real bill by ~2.6x.

---

## Cost plan

All figures below are **estimates**, derived from measured per-task spend in
the failed run cross-checked against `evals/LINEUP_ARBITER_QUALITY.md`. They
are not bills. See P7.

**Measured inputs** (from `evals/BENCHMARK_REAL.md`, priced by `PRICES`):

| arm | per task |
|---|---|
| `quorum` | $0.2029 |
| `single` | $0.0374 |
| `self_critique` | $0.1467 |
| judge | ~$0.004 (estimated; the judge failed, so it is absent from the record) |
| **per task, all arms** | **~$0.391** |

**The Google entry is the exposure.** `PRICES["gemini_pro"] = 1.25 / 10.0`
per Mtok is an unverified estimate, and the arbiter is more than two thirds
of every council bill. A prior entry in this same table was caught
understating a real bill by ~2.6x. Applying that same factor to the arbiter
line is the stress case:

| | base case | Google understated 2.6x |
|---|---|---|
| per task, all arms | $0.391 | $0.608 |
| P1 credit probe | $0.02 | $0.02 |
| pilot, 3 tasks | $1.17 | $1.82 |
| full run, 20 tasks | $7.82 | $12.15 |
| re-run contingency | $2.00 | $3.00 |
| **total to a published benchmark** | **~$11** | **~$17** |

The stress case is not a prediction. It is the number that stops the Google
line being a surprise, and the reason P7 reconciles against invoices before
the full run is authorised.

**Pilot composition.** `--limit 3` takes the first three tasks, all three of
which have a neutral-criteria majority. A pilot that never touches the
contested partition does not exercise what the disclosure plan turns on.
Select the pilot set explicitly: at least one contested task
(`deprecate_v1`, `open_source_core`, `remote_office`, `free_tier`,
`monorepo_split`, `raise_now`, `migrate_off_cloud`).
