# Arbiter quality: the question cost could not answer

Six arbiters have now been measured across otherwise-comparable sessions, and
the finding is consistent: **arbiter choice barely moves the bill.**

| arbiter | lab | shares a lab with a student | objections | cost |
| --- | --- | --- | --- | --- |
| `claude-opus-5` | anthropic | yes | 22 | $0.3177 |
| `claude-sonnet-5` | anthropic | yes | 18 | $0.2981 |
| `claude-fable-5` | anthropic | yes | 20 | $0.2957 |
| `gemini-3.5-flash` | google | **no** | 30 | **$0.2015** |
| `gemini-3.1-pro-preview` | google | **no** | 29 | **$0.2001** |

Flash and Pro land within 0.7% of each other on cost and within one objection
on volume. The spec's open question — "strongest model for the best synthesis,
or a neutral mid-tier that is cheaper and less likely to impose its own answer"
— was framed as a cost/quality trade. **The cost half is now answered: there
isn't one.**

## What the two independent arbiters actually produced

Both reached the same shape of answer: diagnose before committing. They differ
in how they handled the dissent they were sitting on.

| | `gemini-3.5-flash` | `gemini-3.1-pro-preview` |
| --- | --- | --- |
| final answer | 1491 chars | 1071 chars |
| confidence note | 519 chars | 359 chars |
| minority items | 4 | **6** |
| dissent attributed to | all three students | all three students |

The stronger tier wrote a **shorter answer and preserved more dissent** — 6
minority items against 4, from one fewer objection. That is the direction the
protocol wants: the minority report is the feature, and compressing the answer
while keeping more of what was left out is exactly the trade the spec asks the
arbiter to make.

## What this is not

One session per arbiter. Length and item counts are proxies, not quality, and
"preserved more dissent" is a count rather than a judgement about whether the
*right* dissent survived. Scoring that needs the benchmark's rubric machinery
pointed at verdicts rather than answers, with a judge that took no part — which
is built but has never been run on real models.

The honest summary: **cost has stopped being a reason to pick an arbiter, and
quality is now the only axis left — which this harness cannot yet measure.**
