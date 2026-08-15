# Real model output, kept as a fixture

Three round-1 answer sheets produced by Claude Opus, Sonnet and Haiku against
the exact prompt in `prompts.build_sheet_prompt`, captured 2026-08-15.

They are here because mocks cannot test the thing that most needed testing:
`MockProvider` emits what its author decided it would emit, so a schema that
only ever meets mock output has never met a model. These three arrived with
three different transport quirks — a fenced block, a stray blank line inside
the JSON object, and escaped quotes inside claim text — and all three parsed.

They also carry the finding that demoted the disagreement score: all three
models reached the *same* conclusion, and the lexical metric called the
question "sharply contested". See `tests/test_real_output.py`.

Not a benchmark, not a blinding measurement: one lab, three tiers, one
question, orchestrated by hand.
