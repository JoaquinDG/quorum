# A real session, rounds 1-3

Claude Opus, Sonnet and Haiku answering one architecture question through the
actual prompts in `quorum.prompts`, under the actual blinding from
`quorum.blinding`, captured 2026-08-15. Round-1 sheets live in
`../real_sheets/`; this directory holds the critiques and revisions.

Every defect listed below was invisible to `MockProvider`, because a mock
emits what its author decided it would emit. These files are the regression
fixtures for each fix.

## What they caught

**Round 3 had no repair budget.** Opus's first revision was excellent — a
genuine position change citing six objections, confidence dropped 0.72 to 0.61
— and carried six claims against a cap of five, so the engine discarded it and
let the opening sheet stand. The failure is structural: round 3 asks a student
to answer objections, answering adds material, and the cap forbids growth, so
the students who engage hardest are the ones pushed over. Round 3 was also the
only round with zero repairs. Fixed by `SessionConfig.revision_repairs`; the
repair prompt is in this directory's `opus_revision.json` result, which came
back compliant and still changed position.

**`declaration_matches_diff` fired on healthy behaviour.** Haiku held its
position word for word, correctly declared `changed_position: false`, and
rewrote four of its five claims under objection. The flag compared that
declaration against *any* change rather than the position diff, so the ideal
outcome registered as a discrepancy. Now compares like with like.

**The position-change rate was presented as a per-session verdict.** It is a
population statistic. A 3-student council can only score 0/33/67/100%, so
three of its four possible outcomes fall outside the spec's 15-60% band by
arithmetic alone.

## What held

The critique schema, under the hardest available test. All three models had
reached the *same* conclusion in round 1, and the format — no agreement field,
one objection per foreign sheet minimum, name a claim number — still produced
23 substantive objections with zero agreement language.

## What this is not

One lab, three tiers, one question, orchestrated by hand. Not a benchmark, not
a blinding measurement, and no substitute for either.
