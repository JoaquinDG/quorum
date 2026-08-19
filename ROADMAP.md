# Roadmap

The working backlog, in priority order. **This file is the single source of truth**: editing it is how you steer what gets built next.

Reorder freely. Add items freely. Tick a box when the work lands on `main`.

**Claimed items.** An item marked `[~]` and **CLAIMED** is being built right now in an interactive session. Anyone picking up work skips it and takes the next unclaimed one, or the same thing gets built twice.

**Human-gated items.** An item marked **KEYS** spends real money against real API keys. It never runs unattended. The command can be prepared and the harness checked, but a person runs it.

## How to read an item

Each entry states **why it matters** (the gap it closes), **done looks like** (acceptance criteria), and where relevant a **trap**: a way of satisfying the letter of the item while making the repo worse. The traps are the important part. Most of these items have an easy wrong version.

## Ground rules for every item

These are not negotiable and apply to all work below:

- Zero runtime dependencies. The package stays stdlib-only and runs fully offline against `MockProvider`.
- The five protocol invariants hold: independence in round 1, permuted blinding, no self-grading, fail closed per round, fixed rounds. `tests/test_invariants.py` is the enforcement, not the documentation.
- `unittest discover`, `convening_eval.py`, `probe_eval.py`, `benchmark_eval.py`, the quickstart and the replay all pass before anything opens for review.
- Honest claims only. Estimated is never presented as measured; mocked is never presented as real. A metric run against mocks says so at every rendering.
- Replay completeness: anything a report or a metric reads must be in the trace. If a feature needs state the trace does not carry, the trace changes first.
- Paths cited by the published paper are frozen: `tests/fixtures/real_session/live_mixed_lab.jsonl`, `tests/fixtures/real_session/live_three_lab.jsonl`, `evals/LINEUPS.md`. Add to them, never rename or delete. If one must move, leave a stub at the old path.
- One logical change per PR.

---

## Tier 0: the measurements the claims depend on

Everything in the spec is built. What is missing is evidence, and the README is
deliberately written so that a bad number is publishable.

### M1. Run the benchmark for real, with every arm answering (**KEYS**)
- [ ] **Why it matters:** answer quality is the project's central untested proposition, and the repo says so in its own README. The harness runs end to end, but the only complete run is against `MockProvider`, and `evals/BENCHMARK.md` opens by saying so. A live attempt on 2026-08-19 did not produce a valid run: 49 task/arm pairs returned no scored answer, from a provider out of credit and a retired model id. Until every arm answers, there is no benchmark, and the honest claim set stops at surfaced disagreement and auditable reasoning.
- **Done looks like:** a run where zero pairs error, across all three arms (`quorum` / `single` / `self_critique`) on the 20 judgement tasks; the with/without-`favours_deliberation` split reported; the table committed and quoted in the README whatever it says, with `is_mock` false; the "No benchmark numbers" bullet in Known limitations rewritten to point at the real number. Budget estimated up front from the measured per-session costs.
- **Trap:** publishing a partial run. A council that degrades gracefully and a single model that fails outright are not comparable, and a missing arm flatters whoever is left. Issue #3 is the same failure mode.
- **Second trap:** re-running until the numbers look better and reporting the good one. The first complete run is the result. Publishing only if Quorum wins would make this not a benchmark; `self_critique` winning is itself a finding worth writing up.
- **Third trap:** running it before M3 and then re-running after the rubrics change, reporting only the better run. Fix the rubrics first, or version the runs.

### M2. A stronger deanonymization prober (**KEYS**)
- [ ] **Why it matters:** the published leak is +9.7 points over chance (41.9% against 32.3%) from one prober, ten sessions and 31 guesses. That is a wide-error-bar finding carrying a lot of weight in the README, and a stronger prober is the experiment that would change what the blinding claim is allowed to say. The distribution hints that stronger models have more identifiable voices, meaning better seats may cost blinding, but that is a hypothesis at n=10, not a result.
- **Done looks like:** more sessions probed (each archived session costs one call via replay), more than one prober model including at least one from a different lab, and a stated spread or confidence interval rather than a bare point estimate. `evals/PROBE.md` regenerated and the README number updated whichever direction it moves.
- **Trap:** prober shopping. Choosing the prober that returns the smallest leak inverts the design commitment that understating is the one error this measurement must not make. Pre-register the prober choice in the eval doc before running.
- **Second trap:** treating a bigger leak as a bug to fix before publishing. The measurement is the deliverable; the mitigation is a separate item.

### M3. Get the rubrics reviewed by a human who did not write them
- [ ] **Why it matters:** the rubrics decide the benchmark before a model runs. They have had one author and one *model* reviewer, which found 44 of 120 criteria mistagged and 19 unmeasurable (`evals/RUBRIC_REVIEW.md`). Issue #4 records that 7 of 20 tasks still have no neutral majority. This is the highest-value review the repo can receive.
- **Done looks like:** a named reviewer's pass recorded in `evals/RUBRIC_REVIEW.md`, with disagreements kept rather than resolved silently; mistagged criteria retagged; criteria judged unmeasurable fixed or dropped before M1 runs; the count of non-neutral tasks updated in the README.
- **Trap:** treating the model review as satisfying "reviewed by someone who did not write them". It helps, and it is already counted. The standing limitation is specifically the absence of human eyes.
- **Second trap:** re-tagging criteria until the neutral column looks healthy. The honest outcome may be that judgement questions cannot be made deliberation-neutral, which is issue #4's actual thesis.

### M4. Measure the skeptic seat (**KEYS**)
- [ ] **Why it matters:** the skeptic seat ships off by default precisely because its effect is unmeasured. A protocol that ships an untested nudge as standard has stopped measuring itself. The question it has to answer is whether it raises the position-change rate or only the bill.
- **Done looks like:** paired sessions on the same tasks and lineups with the seat on and off, position-change rate, minority items and cost reported side by side, run with real keys, and a default-on/default-off decision recorded with the numbers.
- **Trap:** measuring it on tasks that already convene strongly. A seat that only helps where disagreement was already high has not earned its cost.
- **Second trap:** n=1 theatre. The lineup evals already flag every single-run difference as hypothesis rather than result; hold this to the same bar before flipping any default.
- **Third trap, and the reason this is KEYS:** an offline arm. `MockProvider` dispatches on the round header and never reads the skeptic instruction, so paired mock sessions are byte-identical apart from +97 input tokens. That produces "just the bill", the literal answer to this item's question, arrived at by construction. Measured on 2026-08-17: four paired tasks, zero delta on change rate, objection count and objection text. Do not teach the mock to respond to the instruction either, because that manufactures the opposite answer just as cheaply. Position-change rate under pressure is a fact about models, and mocks have none.
- **Offline work that is worth doing:** one test in `test_divergence.py` that runs the paired session and asserts the mock shows *no* behavioural delta, pinning the blindness so a later run cannot mistake it for a finding. About fifteen lines. The rest of the harness check already exists in `SkepticSeatTests`.

### M5. Measure the shield against a model that is actually trying (**KEYS**)
- [ ] **Why it matters:** added 2026-08-19 with the shield itself (PR #8), because the README and `SECURITY.md` both now state in writing that this number does not exist. The structural claim is defensible on its own terms: fence markers are derived per recipient and per round, so a participant cannot close a fence around text it will never read. Everything above that line is a pattern list that catches the unsubtle, and the preamble telling a reader that an instruction attempt is itself a legitimate objection is untested. How much that sentence is worth against a competent attacker is the open question.
- **Done looks like:** a real student seat instructed to capture the verdict, manufacture consensus, deblind itself and forge a frame, run against the real arbiter; the rate at which each attack reaches the arbiter's output reported per attack class; the number published whichever direction it moves.
- **Trap:** scoring it on whether the detectors fired. Detection is the weakest leg by design and flagging is not blocking, so a run where every payload is flagged *and* still lands in the verdict is a failure that a detector-based metric reports as a success. The measurable outcome is whether the attack changed the arbiter's output, not whether the shield noticed it.
- **Second trap:** writing the attacker's payloads to match the pattern list. That measures the regex against itself. The attacking model should be told the goal and the protocol, not the detectors.
- **Why KEYS:** `MockProvider` generates no adversarial text, so an offline arm can only replay a fixed corpus. That corpus already exists in `tests/test_shield.py` and is a regression fixture, not a measurement. An attacker that adapts is the whole quantity being measured.

---

## Tier 1: harness integrity

Open issues, all of them findings against this project rather than feature
requests. Each one is a way the harness can produce a number that flatters the
council.

### H1. A provider outage flatters the council (issue #3)
- [ ] **Why it matters:** the council degrades to two students and still answers; the single-model arm just fails. Any run with outages silently compares a working arm against an absent one.
- **Done looks like:** the benchmark refuses to score a task where any arm is absent, and reports absence as absence rather than as a loss. Partly landed already ("Refuse to print a leaderboard when an arm did not answer"); the item closes when the per-task table cannot show a score next to an `err`.
- **Trap:** dropping affected tasks quietly. The count of excluded tasks is part of the result.

### H2. Completion budgets fail toward a flattering answer (issue #2)
- [ ] **Why it matters:** every budget in the repo was tuned before reasoning models, and a truncated sheet reads as non-compliance by the model rather than as a budget set too low by us.
- **Done looks like:** budgets derived per model rather than fixed, truncation distinguished from malformed output in the trace, and a test that a truncated response is never recorded as a schema violation.
- **Trap:** raising every budget until nothing truncates. That hides the failure mode and raises the bill; the goal is to see truncation, not to avoid it.

### H3. The cost multiple is not comparable across lineups (issue #5)
- [ ] **Why it matters:** the README's ≤8× guardrail is missed by 2 to 3× (23.7× on the first complete session, 6.0× to 27.4× across lineups) and the number moves with the lineup, because each session's baseline is its own round 1. A guardrail that is always missed and cannot be compared across runs is decoration. The arbiter is more than two thirds of every bill.
- **Done looks like:** a stated guardrail that measured sessions actually pass or fail meaningfully, either stated against a named lineup or normalised to be comparable; `LINEUPS.md`-style tables reporting against it; the old ≤8× figure kept in the record as the estimate it was.
- **Trap:** picking the lineup where the multiple looks best.
- **Second trap:** moving the goalposts silently. The 23.7×-against-8× miss is a published finding and part of why the README is trusted. A restatement cites it rather than burying it.

---

## Tier 2: decisions the measurements feed

### S1. A semantic disagreement score
- [ ] **Why it matters:** divergence is lexical today and says so. Two sheets that say the same thing in different words score as contested, which is the metric being wrong in the direction that makes the protocol look useful.
- **Done looks like:** a semantic score behind the existing interface, with the lexical one kept as the offline default so the zero-dependency promise survives; both reported when the semantic path is available.
- **Trap:** making an embedding model or a judge a hard dependency. Stdlib-only and offline-first are the house style, and this is the item most likely to break them.
- **Second trap:** shipping it as a free-looking number. It costs a call per session, and the bill has to show it.

### S2. Decide the wording-spread metric's fate
- [ ] **Why it matters:** the metric is demoted and inert but still rendered, labelled "Opening wording spread" with *does not measure agreement* printed underneath. The dogfood council reversed every initial position and recommended outright deletion unanimously (`evals/DOGFOOD.md`), on the grounds that a precise-looking number will be read as contention however it is labelled. The README argues that keeping it visible preserves the finding. Someone has to decide; both answers are honest and limbo is not.
- **Done looks like:** either deleted from all surfaces with the finding preserved in prose, or kept with a recorded reason that overrides the council.
- **Trap:** deleting the *story* along with the number. A metric that scored a unanimous council as sharply contested is one of this repo's most useful teaching findings, and it has to survive the metric.

### S3. Arbiter tier: strongest or neutral mid-tier
- [ ] **Why it matters:** an open question in the README and the biggest single lever on the cost multiple in H3. Measurement has already reframed it: arbiter choice moved one set of otherwise-comparable bills by only about 7% (`evals/LINEUP_ARBITER_QUALITY.md`), so the cost framing is largely settled and the live question is verdict *quality*, which needs M1's judge machinery.
- **Done looks like:** arbiter variants scored on verdict and minority-report quality by a blinded judge; position-change rate and minority-report survival compared; the default chosen with a number attached and the answer written into the README.
- **Trap:** letting a same-lab judge grade its sibling's syntheses. `arbiter_shares_lab` exists because that prior is real.

### S4. Minority report threshold
- [ ] **Why it matters:** preserve every dissent (current) or apply a materiality bar? A real session produced 13 items, which is thorough but hostile to the three-minute non-technical reader the report targets. The dogfood council converged on tiering and compression with an explicit auditable bar, and specifically rejected silent curation as the synthesis-away the feature exists to prevent.
- **Done looks like:** a decision recorded with its reasoning, and if a bar is added, the report renders tiered dissent with the bar's criteria visible, every item still present, and a test that a dissent below the bar is still in the trace even when compressed in the report.
- **Trap:** the arbiter quietly deciding what is material. The arbiter authored the majority synthesis and is not neutral about dissent; that is the design constraint, not a footnote.

### S5. Council size: three, four, or adaptive
- [ ] **Why it matters:** the default is three and the cap is five, but a four-lab session measured 100% schema compliance and 8.1× at no extra cost over three-lab lineups (`evals/LINEUP_FOURLAB.md`), contradicting the 1.4x-per-seat prediction. The dogfood council split three ways and reached no consensus. It is also entangled with the weak-seat finding: the cheapest, most distinct seat is the one that fails the schema.
- **Done looks like:** enough sessions at three and four seats to compare compliance, objections, minority items and cost beyond n=1; the default confirmed or changed with the data cited; one council-size number documented rather than two.
- **Trap:** adaptive sizing triggered by signals endogenous to council size. The council's own preserved dissent named this circularity. Prove the fixed-size comparison first.

### S6. Claim granularity
- [ ] **Why it matters:** "max 5 claims, one sentence each" may force lossy compression on genuinely complex positions. `nuance` is the current answer and whether it is enough is open, not settled.
- **Done looks like:** evidence from real sessions on how often claims hit the cap and how much ends up in `nuance`, before any schema change.

---

## Tier 3: rendering

### V1. The replay world
- [ ] **Why it matters:** designed and deliberately not built. A world where avatars act out a session by replaying the trace, as a rendering layer with no engine changes by construction. The Session Report is the first proof the constraint holds: it already renders entirely from the file. This would be the second.
- **Trap:** any change to `session.py` or `trace.py` to make the rendering easier. That would falsify the claim the feature exists to demonstrate. If the renderer needs something the trace lacks, the trace format is what changes, and the frozen fixture traces must still replay.

---

## When this list is empty

Do not invent work. Run the maintenance pass instead:

- Re-verify captured eval output, probe numbers and cost figures against the current files. The README quotes numbers from `evals/`, and quoted numbers drift.
- Re-check seat prices against vendor pages. One was found understated by about 2.6×, in the reassuring direction, which is the direction to hunt.
- Confirm the frozen fixture paths still exist under their published names and still replay.
- If everything is accurate and nothing above is unchecked, open an issue describing the repo's state and stop. An empty run is a good outcome; padding the log is not.
