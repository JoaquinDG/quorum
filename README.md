# Quorum

[![ci](https://github.com/JoaquinDG/quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/JoaquinDG/quorum/actions/workflows/ci.yml)

**A council of AI models that answers your question independently, critiques each other claim by claim, and hands you the argument — not just the answer.**

Zero dependencies. Runs fully offline out of the box. `git clone`, run the tests, watch a debate in under a minute.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # 318 tests
PYTHONPATH=src python3 evals/convening_eval.py         # when is a council worth it?
PYTHONPATH=src python3 evals/probe_eval.py             # is the blinding actually blind?
PYTHONPATH=src python3 evals/benchmark_eval.py         # quorum vs one model vs self-critique
PYTHONPATH=src python3 examples/quickstart.py          # full pipeline, no API keys
python3 replay.py traces/quickstart.jsonl              # rebuild it from the trace alone
```

The quickstart writes `reports/quickstart.html` — a single self-contained file you can open, email, or attach to a decision doc.

Sibling to [Switchboard](https://github.com/JoaquinDG/switchboard): Switchboard picks the one right model for a task; Quorum convenes several when one isn't enough. Quorum imports the pattern — the `Provider` protocol, the policy object, the JSONL trace philosophy — rather than the code, so the two install independently.

> **Status: every phase built; no results published yet.** The session engine, answer-sheet schema, blinding, replay-complete traces, the Session Report, the convening rule, cost accounting, the skeptic seat, the disagreement score, the deanonymization probe and the benchmark harness all exist and are tested. **What does not exist is a single number produced by a real model.** Every eval here runs against mocks, which proves the harnesses work and proves nothing about the models. The blinding metric and the benchmark table go into this README the day someone runs them with real keys — whatever they say.

## The problem

For high-stakes, ambiguous questions — strategy, pricing, architecture, risk — a single model gives you a confident answer with no visibility into its blind spots. You cannot tell a robust conclusion from a fluent guess. Asking three models by hand means juggling tabs, re-pasting context, and eyeballing differences with no structure.

The cost of not solving it: overconfident single-model answers get shipped into real decisions, and the disagreement between models — the best available signal of genuine uncertainty — is thrown away.

## The bet

**The value is not a better final answer.** The evidence for multi-model deliberation improving answer quality is modest and task-dependent, and this README does not claim it.

The value is **surfaced disagreement and auditable reasoning**. Where independent models diverge is where the real uncertainty lives, and a decision-maker who can see the argument — the objections, the mind-changes, the dissent that didn't make the final answer — is better calibrated than one handed a single fluent paragraph.

The transcript is the product.

## The protocol

Three students (distinct model families — diversity of priors is the point) and one arbiter that never debates.

```mermaid
flowchart TB
    Q[Question] --> CR{should_convene?}
    CR -->|no, ~91% of a mixed workload| ONE[One model + a verifier<br/>that's Switchboard's job]
    CR -->|yes| R1
    subgraph R1["Round 1 — silent exam"]
        S1[Student 1<br/>answer sheet] ~~~ S2[Student 2<br/>answer sheet] ~~~ S3[Student 3<br/>answer sheet]
    end
    R1 -->|sheets relabelled A/B<br/>per recipient| R2
    subgraph R2["Round 2 — blind claim-level critique"]
        O["objection(sheet, claim_n, argument)<br/>≥1 per foreign sheet, no agreement field"]
    end
    R2 -->|critics relabelled<br/>under a fresh mapping| R3
    subgraph R3["Round 3 — revision"]
        D["revised sheet<br/>+ structural diff computed by the engine"]
    end
    R3 --> R4
    subgraph R4["Round 4 — grading"]
        V[Arbiter: final answer<br/>+ confidence note<br/>+ minority report]
    end
    R1 -.-> T[(replay-complete<br/>JSONL trace)]
    R2 -.-> T
    R3 -.-> T
    R4 -.-> T
    T --> RP[Session Report · benchmark ·<br/>replay world — all just players]
```

**The medium of exchange is a schema, not a voice.** Students never send each other prose. They fill in the same rigid template and exchange *that*:

```json
{"position": "one sentence",
 "claims": [{"n": 1, "text": "one sentence, specifically disagreeable"}],
 "assumptions": ["what this depends on"],
 "would_change_my_mind": ["explicit falsifiers"],
 "confidence": 0.7,
 "nuance": "free text, never shown to another student"}
```

This is the anonymization mechanism, not a side effect of one. Most of the style signal that would identify a model family is never generated in the first place, so there is nothing to detect and nothing to paraphrase away. It is also what makes position tracking possible: a numbered claim can be objected to, dropped, or edited, and all three are mechanically detectable.

## Design decisions & tradeoffs

**Independence is round 1's entire job.** A round-1 prompt contains the question and nothing else — no peer answers, no hints about the council. It costs nothing to preserve and everything to lose: one leaked peer answer converts three independent priors into one prior and two echoes. `test_invariants.py` asserts the prompt is byte-identical to `build_sheet_prompt(task)`.

**You cannot register agreement, because there is no field for it.** Round 2 accepts `objection(sheet, claim_n, argument)` and nothing else, with at least one objection required per foreign sheet. A critique that skips a sheet is non-compliant even if its other objections are excellent. Free-form debate reads nicer, but it lets a model endorse a vibe, leaks identity through style, and makes position tracking impossible.

**Blinding is permuted, not positional.** The naive implementation — "the other sheets, in seat order, labelled A then B" — leaks the seating chart after one session: a critic that recognises one sheet knows the other by elimination, and anyone comparing two sessions can align every label. Labels are derived from a hash of `(session, recipient, round)`, deterministic (tests and replay reproduce it exactly) without being positional. Round 3 relabels the critics under a different salt, so a student cannot align "the sheet I critiqued" with "the critic who attacked me".

**`nuance` reaches the arbiter and no student.** The escape valve for positions the five-claim cap would flatten is also the field most likely to carry a fingerprint, and the one thing nobody is allowed to object to — so no critic sees it. Excluding it from *grading* too, though, made the field write-only: a documented escape valve that could never reach the answer. The arbiter gets it, clearly marked as un-critiqued. The cost is stated rather than hidden: nuance is unblinded free text, so it is the one channel through which a student could signal its identity to the grader.

**Diffs are computed, never taken on the model's word.** The engine matches claims by *content* similarity (so a withdrawal that triggers renumbering doesn't read as "everything changed"), and compares the result against what the student *declared*. `declaration_matches_diff` is the interesting field: a model that says it reconsidered while resubmitting the same sentence is a sycophancy signal, and one that quietly rewrites its position while reporting no change is the opposite. The published position-change rate uses the computed value, because taking the model's word would make the metric agree with the pathology it exists to measure.

**No self-grading, enforced in the type.** A `Council` whose arbiter also holds a student seat cannot be constructed. Switchboard learned this in its auditor: independence that lives in a code path is independence a later refactor can quietly remove. The arbiter is also called exactly once, after the debate — asserted in the invariant tests.

**The arbiter sees seats, not model names.** It isn't a participant, so this isn't about sycophancy — it's about brand priors. An arbiter told that one sheet came from the largest frontier model in the lineup has a reason to weight it that has nothing to do with the argument on the page. The session maps seats back to models afterwards, so the minority report is still attributable in the trace.

**Fail closed, per round.** A student that errors or returns an unparseable sheet is recorded absent for that round and the session says so everywhere the verdict appears — a two-student session is labelled a two-student session, never presented as a full council. Malformed output is never coerced into a valid shape, and the raw text is kept in the trace because "returned six claims" and "returned an apology" are different bugs. Absence is scoped: fail round 1 and you're out; fail round 2 and you raise no objections but still answer those raised against you; fail round 3 and your opening sheet stands as final. Below two sheets there is no peer review, so the session closes with no verdict rather than presenting one model's opinion as a council's.

**Strict about meaning, tolerant of format.** Fenced JSON, prose preambles, `"65%"` for `0.65`, and `"3. foo"` instead of `{"n": 3, "text": "foo"}` all parse — those are transport accidents. Six claims, a confidence of `1.4`, or claims numbered `1, 2, 4` are rejected: clamping a confidence would launder a schema violation into a number the report quotes, and a gap in claim numbering makes every objection against that sheet ambiguous. The one rule deliberately *not* enforced as an error is "one sentence each" — sentence counting is a heuristic, and hard-failing a substantive claim over a semicolon throws away real content to satisfy a regex. Multi-sentence claims are recorded as `compliance_warnings`, which is what the claim-compliance metric actually wants: a rate, not a crash.

**One re-prompt, then it's recorded — and billed either way.** A non-compliant critique gets exactly one repair attempt with the reason attached. Retrying forever lets a model that can't follow the format burn the session budget; not retrying discards a good critic over a missing field.

**Fixed rounds.** There is no "continue until they agree" loop. Unbounded deliberation burns tokens manufacturing the consensus the protocol exists to avoid manufacturing, and a session that can't converge in one revision has found real disagreement — which is the finding, not a failure.

## The trace

Every session emits append-only JSONL where each event is self-describing:

```json
{"ts": 1.0, "session_id": "q-…", "round": 2, "actor": "student:1",
 "event_type": "objection_raised", "payload": {…},
 "tokens_in": 412, "tokens_out": 96, "cost_est": 0.0021}
```

Event types: `task_posed`, `sheet_submitted`, `sheets_blinded`, `objection_raised`, `sheet_revised`, `position_changed`, `verdict_delivered`, `minority_recorded`, `student_absent`, `probe_result`, plus three additions:

| Added | Why |
| --- | --- |
| `arbiter_absent` | Fail-closed has to cover the grader. Folding it into `student_absent` would make "how many students dropped out?" — a published number — quietly wrong. |
| `session_closed` | Replay needs a terminator to tell a finished session from a truncated file. It also carries the bill, the baseline and the council size. |
| `attempt_discarded` | A re-prompt is a real model call whose output is thrown away. Without an event it is invisible, and a session with repairs reports a *lower* cost than a clean one. |

**The chess-PGN principle: any renderer is a player for this file.** A PGN doesn't ship with a board; it carries enough that any board can reconstruct the game. If the Session Report, the benchmark harness, or the v2 replay world would need something the trace doesn't carry, *the trace format is wrong* — not the renderer.

`replay.py` keeps that honest. It imports no session state and rebuilds every position, objection, diff, absence and verdict from the file alone:

```bash
python3 replay.py traces/quickstart.jsonl
```

The test suite runs a live session and a replay of its trace side by side and compares them field by field — including recomputing each structural diff from the replayed sheets and asserting it matches what the engine recorded. If the trace ever starts storing a summary instead of the sheets themselves, that test goes red. The Session Report is held to the same rule: it takes a `ReplayedSession`, and a live result handed to it is routed through `replay` first, so the report physically cannot start depending on state the record doesn't keep.

Two consequences the format is built around: events carry *full* content rather than references (a trace of pointers into a database is a log, not a record), and the trace is **not** blinded. It records seats, models and every label mapping, because the auditor's view and the participant's view are different views on purpose. No participant is ever shown the trace.

## The Session Report

One self-contained HTML file — no external CSS, fonts, scripts or images, theme-aware, responsive — plus a Markdown fallback for terminals and PR comments.

```python
from quorum import convene, demo_council, mock_pool, write_report

result = convene(question, demo_council(), mock_pool(demo_council()))
write_report(result, "reports/session.html")   # writes session.md alongside
```

It carries the opening positions, the sharpest objections, who changed their mind and why, the final answer and confidence note, the minority report as a closing panel, and the bill against a single-model baseline.

**"Sharpest" is measured, not guessed.** The obvious proxies — argument length, or the model's own self-report — measure rhetoric. The trace already records consequences: whether the target *cited* this objection as its reason for revising, and whether the claim under attack was later withdrawn or reworded. An objection that made someone drop a claim is sharp in the only sense a reader cares about, and it's a fact rather than an impression. Each one is shown with what it actually did.

**The report is built not to flatter the session.** A reduced council and an incomplete cost are banner-level news. A session where nobody moved says so plainly — *"no objection landed"* — instead of presenting agreement as resolution. An empty minority report is labelled as a claim the arbiter made, not as proof of consensus. Each of those has a test in `test_report.py::HonestyTests`.

## When to convene

A session costs several times a single answer, so the interesting problem isn't "can three models debate" but "how often should they".

```python
from quorum import Task, should_convene

decision = should_convene(Task(question, "architecture", 0.85))
print(decision.convene, decision.reason)
```

**The gates fail closed — the opposite of Switchboard's router, on purpose.** Switchboard's gates degrade *upward*: when nothing qualifies it escalates to the most capable tier, because the cost of an unqualified model answering is a bad answer. Here the cost of a wrong call is a 7x bill, so an unrecognised task type or a missing complexity score means *don't convene*. The default answer is no.

**A council cannot help a question with a checkable answer.** If the output can be verified — extracted fields, a translation, a passing test — the right tool is one model plus a verifier, which is what Switchboard already does. Deliberation is for questions where the disagreement *is* the information.

`evals/convening_eval.py` runs the rule over a 43-task mixed workload with two deliberate traps: **hard but verifiable** tasks (a rule keying on difficulty alone convenes these and blows the budget) and **ambiguous phrasing, trivial stakes** (a rule keying on phrasing alone convenes these). Current results:

```
[PASS] <10% of a mixed workload convenes (9.3%)
[PASS] every genuine judgement call convenes (4/4)
[PASS] no routine task convenes (0 false positives)
[PASS] the rule is not simply 'never' (4 convened)
```

That last check exists because a rate under 10% is trivially satisfiable by never convening at all.

Two honest caveats. The 9.3% has little headroom, and the rate is a property of the *traffic* as much as the rule — a workload with more genuine judgement calls in it should and would exceed 10%, and that would be correct behaviour rather than a regression. And the signal extraction is keyword heuristics, which is the weakest part of the module: it exists so the rule is testable offline and so a caller who supplies nothing still gets a conservative answer. Real deployments should classify with a small model (Switchboard's triage does exactly this) and pass `task_type` and `complexity` in, which the API treats as authoritative.

## The bill

Every session computes its own baseline and carries it in the trace, so the report can print it without re-deriving prices the record doesn't contain:

```
this session:         $0.0379
one model, once:      $0.0057  (delphi-frontier)
multiple:             6.7x  (within the 8x guardrail)
```

The baseline is deliberately conservative: **the same question, answered once, by the priciest seat in the room**, measured from the session's own round 1 rather than guessed. It isn't "the cheapest thing that would have worked" (unknowable) and isn't "three sheets from one model" (a different protocol). Price is a rough proxy for capability, so `SessionConfig.baseline_model` overrides the choice, and the chosen model is printed alongside the number — a reader should be able to disagree with the comparison rather than having to guess it.

Unpriced seats are named rather than costed at zero. A report showing `$0.0031` for a session where the arbiter carried no price isn't approximately right, it's wrong in the reassuring direction.

## Is the blinding actually blind?

Every anonymization scheme claims to work. This one publishes a number.

A critic model that took no part is shown the round-1 sheets under fresh labels, told which model families were in the room and that each wrote exactly one sheet, and asked to say which wrote which. Accuracy is scored against chance.

```bash
PYTHONPATH=src python3 evals/probe_eval.py
```

Three commitments make the number worth reading:

- **The probe never runs on a participant.** A student recognises its own sheet instantly and scores a free hit, which inflates the leak without measuring one. `probe_session` refuses a prober that holds a seat, the same way `Council` refuses an arbiter that debated.
- **The prober is given every advantage.** It gets the roster and the one-sheet-each constraint. A probe that also had to guess the candidate list would *understate* the leak, and understating it is the one direction this measurement must not err in.
- **A refusal is not a pass.** An abstention or an unparseable reply is excluded from the denominator, not counted as a miss. Counting abstentions as wrong answers lets a cautious prober manufacture a reassuring number.

**The offline number is worthless as evidence and the eval says so.** Mock sheets carry no model-specific style, so a near-chance result there is a property of the mock. The metric exists now so that running it for real is one command, and so the number exists before there is any temptation to skip it. Whatever it says goes in this README — near chance means the schema held; above chance is a known, quantified limitation stated with its size.

## The benchmark

Three arms on 20 judgement tasks: **quorum**, **single** (the strongest seat, answering once — the thing you'd actually do instead), and **self_critique** (that same model attacking its own answer and revising). That third arm is the sceptic's question, and the most useful one in the harness: is the value in *multiple models*, or just in a second pass?

**Writing rubrics that don't hand the council the win is the hard part**, and pretending otherwise would make the exercise theatre. Score "acknowledges uncertainty" and "presents multiple perspectives" and a deliberation cannot lose — it has demonstrated only that it was scored on its own format. Two defences, neither complete:

1. Every criterion is tagged `favours_deliberation`, and the harness reports the total **with and without** those criteria. If Quorum leads on the full set but not the neutral set, it is winning on format, and that shows up in its own column rather than in the headline.
2. Criteria are about the *answer*, not the process. "Names a specific course of action" and "states what the recommendation depends on" are properties of a good answer whoever wrote it. "Shows its deliberation" is not a criterion at all.

The judge took no part, scores all three answers in one call on a common scale, and sees them shuffled and unlabelled. `evals/judgment_tasks.json` says plainly that the rubrics were drafted in one pass and reviewed by nobody — a second pair of eyes on that wording is the highest-value review this repo can get.

**Mock runs are not results.** Offline every arm emits canned text scored by a hash. `BenchmarkReport.is_mock` is set and every rendering says so in its first line, so a file that escapes into a README cannot be mistaken for evidence.

## Wording spread (a metric that got demoted)

This shipped as a **disagreement score**: a 0–1 divergence measure over the round-1 sheets, computed before anyone argues, because that is the only moment the positions are independent. The lexical blind spot was documented from the start — word overlap tracks vocabulary, not meaning.

Then it met real models and failed in the *common* case rather than the exotic one.

Three Claude tiers answered one question independently. All three said **refactor in place** — same conclusion, similar reasoning. The metric scored it **0.68, "sharply contested"**, driven by 0.79 claim divergence between sheets that agreed. Different vocabulary for the same idea reads as conflict.

A number that can call unanimity "contested" is worse than no number, because it misleads in exactly the direction this project exists to prevent. So it was demoted rather than quietly kept:

- `label` now describes wording (`"high lexical variety"`), never agreement. A test forbids the words *contested*, *unanimous*, *aligned*, *agree* from ever appearing in a label again.
- `measures_agreement` is a property that returns `False`, so any surface showing the number has to acknowledge what it is.
- The report calls it **"Opening wording spread"** and prints *does not measure agreement* underneath.
- Both failure directions are pinned by tests — the constructed one (opposite conclusions, near-identical words, scored as agreeing) and the observed one.

A real disagreement score needs semantics: an embedding model or a judge scoring stance agreement. That stops being free, which makes it honest P2 work rather than a weekend regex. Nothing in the protocol reads this number.

## The skeptic seat

Optional: one student instructed to attack the most confident sheet regardless of whether it agrees.

```python
Session(council, providers, config=SessionConfig(skeptic_seat=2))
```

Independence stops sycophancy *between* rounds but not the quieter version within one — three models can converge on the most confidently-stated position without any of them having tested it. The target is chosen by stated confidence, which is already visible on every sheet the critic holds, so the instruction adds pressure without adding a single bit of identity information.

It is **off by default**, because whether it helps is an empirical question and a protocol that ships an untested nudge as standard has stopped measuring itself. The effect on position-change rate is what settles it.

## What a session looks like

```
=== ROUND 2 — blind claim-level critique ===
  Student 3 -> Student 1 claim 2 (seen as Sheet A):
      On claim 2: Deferring to measurement is not a neutral act; the two weeks
      spent measuring are themselves a decision with a cost, and this claim
      prices that cost at zero without arguing for it

=== ROUND 3 — revision ===
  Student 1 (atlas-3): CHANGED POSITION
      from: Yes, and the main risk is manageable with staged rollout
      to:   On reflection, yes … but only under the narrower reading below
      withdrew claim: Staging the rollout converts an all-or-nothing bet into
                      three cheap experiments
      confidence -0.15
  Student 2 (borealis-2): held position
  Student 3 (cinder-1): held position

=== ROUND 4 — grading ===
  minority report:
    - [Student 3, objection] The dissent that the question is unanswerable without
      measuring where the constraint actually binds was not resolved, only outvoted

position-change rate: 33%
```

The offline mocks *disagree with each other* — three distinct personas, real objections, a real withdrawal. That's deliberate: a mock that echoed its prompt would reach unanimous agreement in round 1 and prove nothing about a protocol whose entire purpose is surfacing disagreement.

## Protocol health numbers

Every session reports these. They are diagnostics for the protocol, not scores for the answer:

| Metric | What it means | Healthy |
| --- | --- | --- |
| `position_change_rate` | Share of students whose *computed* position diff moved | 15–60%. 0% is theatre; ~100% is herding |
| `dissent_preserved` | Did the minority report carry anything? | ≥30% of sessions |
| `compliance_rate` | Share of model responses that parsed into valid, claim-referencing structure | ≥90% |
| `provider_errors` | Calls that never returned text | tracked separately — availability is not compliance |
| `reduced_council` | Did fewer students answer than were seated? | shown wherever the verdict is |
| `cost_multiple` | Session cost ÷ one single-model answer | median ≤ 8x |

## What the adversarial review caught

Phase 1 shipped green — 142 tests, every acceptance criterion met — and a review pass afterwards found six real defects that the tests had not been written to catch. Recorded because the *class* of bug is the interesting part, and because `test_accounting.py` now exists to keep each one fixed.

1. **Repairs were free.** A re-prompted critique is a real model call whose output is discarded, and it produced no trace event — so a session that burned extra calls reported a *lower* cost than a clean one. A cost guardrail that errs downward on exactly the paths that waste money is worse than no guardrail. Fixed with `attempt_discarded`, which also answers "what are repairs costing us?".
2. **Unpriced seats were costed at zero,** and the total looked precise. Now `unpriced_seats` names them and every surface labels the figure a lower bound.
3. **Claim-compliance was measuring uptime.** A provider 503 counted as a schema failure — the exact conflation Switchboard's README warns about in its own routing. Provider errors now sit in their own counter, outside the compliance denominator.
4. **`nuance` was write-only.** Excluded from the critique rounds *and* from grading, it could never influence the answer it exists to inform.
5. **Session ids could collide** inside one clock tick. Not a cosmetic problem: `replay` refuses a file holding two sessions under one id, so a collision doesn't corrupt one report — it makes both unreadable.
6. **A round-1 sheet could declare `changed_position`** and have it silently ignored, in a parser whose whole stated posture is rejecting unexpected fields.

Then the evals caught three more, each from a different direction:

7. **The convening rule was measuring dialect, not ambiguity.** A genuine judgement call ("we can either hire two seniors or four juniors — which is the better bet?") scored *zero* ambiguity because the heuristics only recognised "should we".
8. **The same bug, at scale, from the benchmark.** A test asserting that every benchmark task would actually convene found that **16 of 20 were declined** — hand-written judgement calls, explicitly typed, scored 0.75–0.9, refused because they open "Do we rebuild…" rather than "Should we rebuild…". Patching in one more verb would have been the same bug rescheduled a third time. The fix was structural in two places: ambiguity is now detected as two *families* (a question that asks what to do, or one that puts alternatives on the table) rather than a list of phrasings, and a **supplied** classification is now weighted far above the keyword markers that exist to approximate it. The module had always claimed supplied values were authoritative; the scoring quietly overrode them.
9. **The disagreement score called a unanimous council "sharply contested."** Found by pointing the schema at real models for the first time. Demoted to a wording-spread measure rather than deleted, because deleting it would hide the finding. Full write-up above.
10. **The mock prober scored 3 standard deviations below chance.** Its permutation was hashed from the prompt's tail — the roster and the JSON rule, byte-identical in every session — so one fixed permutation was reused throughout. A probe reading "better than random" is not a thing, and an artifact in the *blinding metric* is the worst place to leave one.

The meta-lesson matches Switchboard's: unit tests confirm the code does what it says; scenario evals and adversarial probes are what catch the code saying the wrong thing. Finding 8 is the sharpest version — it surfaced only because two independently-built components (the benchmark's task set and the convening rule) were asserted against each other, which no test of either alone would have done. All of it runs in CI.

## Using real models

`AnthropicProvider` and `OpenAICompatibleProvider` ship with the package — stdlib only, with retries, jittered backoff, `retry-after` support and typed errors. One real session is one command:

```bash
export ANTHROPIC_API_KEY=...        # your shell, never this repo
PYTHONPATH=src python3 examples/live_session.py "Should we rebuild or refactor?"
export OPENAI_API_KEY=...
PYTHONPATH=src python3 examples/live_session.py --mixed "..."   # what the protocol is for
```

**Keys are read from the environment and nowhere else.** No constructor default, no config file, no lookup path. Nothing in this repo stores one, and no key is ever written into a trace or a report — `test_http.py` asserts that.

A missing key raises `ProviderConfigError`, which names the variable. That is deliberately *not* the same as an outage: marking a student absent for a missing key would run a two-model session, label it as one, and hide a deployment bug behind a protocol feature built for a different problem.

Writing your own transport is one method:

```python
class MyProvider:
    name = "anthropic"

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        ...  # raise ProviderError on failure so the session fails closed
        return Completion(text=..., model_id=model_id, input_tokens=..., output_tokens=...)
```

Wiring a council directly:

```python
from quorum import Council, ModelCost, ProviderPool, Seat, Task, convene, should_convene

council = Council(
    students=(
        Seat("claude-…",  "anthropic", ModelCost(3.0, 15.0)),
        Seat("gpt-…",     "openai",    ModelCost(2.5, 10.0)),
        Seat("gemini-…",  "google",    ModelCost(1.0,  5.0)),
    ),
    arbiter=Seat("…-frontier", "anthropic", ModelCost(5.0, 25.0)),
)

task = Task(question, task_type="architecture", complexity=0.85)
if should_convene(task).convene:
    result = convene(task.prompt, council, ProviderPool([...]), trace_path="traces/prod.jsonl")
    write_report(result, "reports/latest.html")
```

Prices are per million tokens and yours to maintain, exactly as Switchboard's catalog is. The demo council's numbers are synthetic and say so — a cost figure copied from a README is a cost figure that was true once.

## Testing your own protocol changes

`ScriptedProvider` puts exact text on the wire, which is how the ugly cases get tested — a six-claim sheet, a critique that is polite agreement, a model that declares it changed its mind while resubmitting a byte-identical position:

```python
providers = ProviderPool([ScriptedProvider({
    "model-1": [sheet_json, '{"objections": [...]}', revision_json],
    "model-2": [ProviderUnavailable("503")],       # inject an outage
}, name="lab-1")])
```

Queue entries are strings (returned) or exceptions (raised); the last entry repeats once exhausted.

## One lab is not a council

The protocol buys **diversity of priors**, and that comes from distinct model *families* — not from headcount, and not from tiers of one family. A council of three Claude models (or three GPTs) runs fine and means considerably less:

- Models from one lab share training data and alignment, so their **blind spots correlate**. They tend to miss the same things, and the agreement they reach is much weaker evidence than the same agreement across families. Switchboard hit this in its auditor, where a same-lab pass was being counted as full independence.
- The failure mode is worse here, because a single-lab council can look **healthier** than a mixed one. Three similar models converge faster, which reads as consensus on the report and is actually correlation.
- It changes what the **deanonymization probe** measures. Telling Opus from Haiku is a different and easier question than telling Claude from GPT from Gemini, so a probe number from a single-lab council does not transfer to the mixed case in either direction.

So it is allowed and it is never silent. `Council.single_lab` is surfaced on the session, in the `task_posed` and `session_closed` events, and as a banner on the report in both formats. `Council.arbiter_shares_lab` flags the weaker case separately — an arbiter from a student's lab never debated, which is the rule that matters, but it is disposed to find its sibling's reasoning natural.

**There is now evidence for this, not just an argument.** Three Claude tiers were given the real round-1 prompt independently. All three reached the same conclusion, and all three independently reached for the *same* load-bearing consideration — the one-way door under no slack. That is what correlated priors look like from the outside: not three views converging, but one prior expressed three times. `test_real_output.py` pins it.

If you only have one API key, run it — just read the verdict knowing what it is, and use `--mixed` the moment you have a second.

## What real model output changed

Mocks emit what their author decided they would emit, so a schema that has only met mock output has met its own reflection. Three Claude tiers were given the real round-1 prompt by hand and their sheets kept as fixtures (`tests/fixtures/real_sheets/`).

**The tolerant/strict split held.** All three parsed. Three different transport quirks arrived — a fenced block, a stray blank line inside the JSON object, escaped quotes inside claim text — and all three were absorbed as format noise. Zero compliance warnings: every claim was one sentence, and all three landed *exactly* on the five-claim cap. The strictness that rejects six claims and a confidence of `1.4` rejected nothing a real model actually produced.

**All three used the `nuance` field.** That is mild evidence the five-claim cap does flatten something real, which is the open question the field exists to answer — and a reason the arbiter gets to read it.

**The disagreement score broke** (above), and **single-lab correlation showed up immediately** (above).

Rounds 2 and 3 were then run the same way, and this is where it earned its keep.

**The critique schema held under the hardest available test.** All three models had reached the *same* conclusion in round 1 — the worst case for a protocol built to surface disagreement. The format gave them nowhere to say so: no agreement field, one objection per foreign sheet minimum, name a claim number. They produced **23 objections against an acceptance minimum of 6**, none of them agreement in disguise, none needing a repair. Sonnet caught Haiku invoking *sunk cost* as a reason to preserve the status quo and called it "the sunk cost fallacy pointed at itself". Opus, unprompted, noticed what the single-lab warning predicts: *"Both sheets converge on the same answer via this same one-sided accounting, which should reduce rather than increase confidence."*

**Round 3 found three engine defects.**

1. *The revision round had no repair budget.* Opus's revision was excellent — genuine position change, six cited objections, confidence 0.72 → 0.61 — and carried six claims against a cap of five, so the engine discarded it and let the opening sheet stand. The failure is structural: round 3 asks a student to answer objections, answering adds material, the cap forbids growth, so **the students who engage hardest are the ones pushed over the line**. It was also the only round with zero repairs, while critique and verdict each had one. `SessionConfig.revision_repairs` now exists; the cap did not move. Opus's repaired attempt came back at five claims and still changed position.
2. *`declaration_matches_diff` fired on healthy behaviour.* Haiku held its position word for word, correctly declared `changed_position: false`, and rewrote four of its five claims under objection — the ideal outcome. The flag compared that declaration against *any* change rather than the position diff, so it registered a discrepancy. It is sold in this README as a sycophancy detector; it was firing on exactly what the protocol is trying to produce.
3. *The position-change rate was presented as a per-session verdict.* It is a population statistic. A 3-student council can only score 0/33/67/100%, so **three of its four reachable values fall outside the 15–60% band by arithmetic**. Every surface now shows the raw count alongside the rate and says the band applies across sessions.

The session scored **67% — outside the band**, reported as found. One hypothesis of mine was also wrong: I suspected the claim-similarity threshold would misread rewordings as drop-and-add. It didn't — real revisions came in at 0.06–0.33 similarity (genuine rewrites) while an untouched claim matched at exactly 1.00.

To be exact about what this was: one lab, three tiers, one question, orchestrated by hand. Not a benchmark, not a blinding measurement, and no substitute for either. Everything is kept in `tests/fixtures/real_session/`, with a regression test per finding.

### Then the adapters met a live API

Wiring up a real key found two more, before a single session ran.

**A fabricated model id.** The runner's arbiter seat was `claude-opus-4-5-arbiter` — not a model. I had invented the suffix to satisfy the no-duplicate-models rule, and it would have 404'd on the first arbiter call, forty calls into a session. There is now a `--check` preflight that spends one 1-token call per seat and distinguishes the three failures that look identical from outside: missing key, wrong model id, real outage. Every seat is overridable by environment variable, because vendors rename models faster than an example script gets updated.

**Truncation looked exactly like a malformed sheet.** Reasoning models spend the completion budget on reasoning *first*, invisibly. Given 16 tokens, a live model spent all 16 reasoning and returned `finish_reason: length` with empty content. The adapter handed back `text=""`, the parser called it an empty response, and the session would have recorded a healthy model as producing a malformed sheet — blaming the model for our configuration, and corrupting the claim-compliance metric, which exists precisely to measure whether models can follow the format. Now raises `ProviderTruncated`, which says which budget was hit and how much went to reasoning.

Measured usage then set the default: a real round-1 sheet cost **756 visible tokens**, and critiques run larger, so `max_tokens` moved from 2048 to 4096. That is close to free — the budget is a cap, not a reservation.

### One claim of mine got weaker

I attributed the three Claude models' unanimity to single-lab correlated priors. A **GPT-5.1** sheet then parsed clean on the first try — 5 claims, no compliance warnings — and reached *the same conclusion*: refactor in place.

Four models, two labs, one answer. That does not refute correlated priors (all four are trained on overlapping text), but it does weaken the specific claim that the agreement was a Claude-family artifact. The plainer reading is that this question has a defensible answer most capable models find. The single-lab warning stands on its argument; this particular session is no longer evidence for it.

## Known limitations

- **The blinding is structurally sound and empirically unmeasured.** The probe exists and runs; it has never been pointed at a real model. Until it is, "blind" here means "structurally blinded", not "measured to be blind".
- **No benchmark numbers.** The harness runs; every arm has so far been answered by a mock. There is no result, and the README will not imply one.
- **The rubrics have had one author and no reviewer.** They are the artifact most able to decide the benchmark's outcome before a model is called.
- **Vague-agreement detection is coarse.** The structural defence is the schema — you must name a sheet and a claim number. The text filter only stops the laziest evasion; a model determined to be agreeable in forty characters will get past it. The position-change rate is what actually reveals whether critique is happening.
- **No answer-quality claim.** The claims are surfaced disagreement, auditable reasoning, and calibrated confidence — nothing about better answers.
- **Convening runs on keyword heuristics** unless you supply `task_type` and `complexity`. See the caveats above.
- **The cost baseline is a modelled comparison, not a measured one.** It prices a hypothetical single call using the session's own round-1 token counts. It does not run that call.

## Repo map

```
src/quorum/
  sheets.py      the answer sheet: schema, tolerant parsing, strict validation, structural diffs
  blinding.py    per-(session, recipient, round) label permutation; not positional
  prompts.py     every word sent on the wire, in one file
  council.py     seats, prices, and the no-self-grading constraint (enforced in the type)
  session.py     the four-round engine: independence, fail-closed, fixed rounds
  trace.py       append-only JSONL, replay-complete by design
  replay.py      rebuilds a session from the trace alone — the format's proof obligation
  report.py      Session Report: self-contained HTML + Markdown, rendered from the trace
  convening.py   should_convene: gates that fail closed, because the default answer is no
  costs.py       what the council cost, against what one model would have
  divergence.py  how contested the opening sheets were — lexical, and says so
  probe.py       the deanonymization probe: measuring the blinding, not assuming it
  benchmark.py   quorum vs one model vs one model self-critiquing, rubric-scored
  providers/     one-method Provider protocol; offline mocks that actually play the protocol
tests/           318 tests; protocol invariants, review regressions, real-session fixtures
evals/           convening rate, the blinding probe, the judgment benchmark + its 20 tasks
examples/        offline quickstart: convene → session → report → replay
replay.py        front door for the replay utility
```

## Roadmap

Everything in the spec is built. What remains is measurement, and it needs API keys rather than code:

- **Run the probe for real** and put the number in this README, whatever it is.
- **Run the benchmark for real** and put the table here, whatever it says. The project's claims don't depend on Quorum winning, and a benchmark published only when it flatters its author is not a benchmark.
- **Get the rubrics reviewed** by someone who did not write them.
- **Measure the skeptic seat** — does it raise the position-change rate, or just the bill?
- **A semantic disagreement score** to replace the lexical one, which needs an embedding model or a judge and stops being free.
- **v2, designed for and deliberately not built:** a Gather-style replay world where avatars act out the session by replaying the trace — a rendering layer, no engine changes by construction. The Session Report is the first proof that this works; it already renders entirely from the file.

## The trilogy

Three small repos on the same problem from three angles, sharing a house
style: stdlib-only, offline-first, explicit policy objects, JSONL traces,
scenario evals with a scorecard, and claims that are demonstrated by a
command or labelled as estimates.

| | Thesis |
|---|---|
| **[Switchboard](https://github.com/JoaquinDG/switchboard)**, *route* | Which model should this task go to, and what did that choice cost? Routing and economics as an explicit, auditable policy. |
| **[Quorum](https://github.com/JoaquinDG/quorum)**, *deliberate* | When one model's answer isn't enough, how do several disagree productively? Multi-model deliberation with a recorded transcript. |
| **[Governor](https://github.com/JoaquinDG/governor)**, *supervise* | When a run goes wrong, who notices and what do they do about it? Behavioural supervision with a graduated response. |

Quorum's contribution to the shared trace philosophy is the **replay-completeness
rule**: the record has to be sufficient on its own, and `replay.py` plus a
report that reads only the file are how that claim is kept honest rather than
asserted.

## Open questions

- **Arbiter tier.** Strongest model (best synthesis) or neutral mid-tier (cheaper, less likely to impose its own answer)? Currently strongest, one line to change, and a question for the benchmark rather than the README.
- **Minority report threshold.** Preserve every dissent (current behaviour) or apply an arbiter-scored materiality bar? Curation risks becoming the synthesis-away this feature exists to prevent.
- **Claim granularity.** Does "max 5 claims, one sentence each" fit genuinely complex positions, or force lossy compression? The `nuance` field is the current answer; whether it's enough is open, not settled.

MIT licensed.
