# Deanonymization probe — real sessions

**41.9% accuracy vs 32.3% chance (+9.7% points) over 31 guesses in 10 session(s)**

Reading: modestly above chance — a small, quantified leak

Prober: `gemini-3.1-pro-preview`, which held no seat in any session below.
It was given the roster and told authorship is one-to-one — every
advantage, because understating a leak is the one error this number
must not make.

Measured by replaying archived traces, so each session costs one call
rather than a re-run.

| session | sheets | correct | chance | accuracy |
| --- | --- | --- | --- | --- |
| `arbiter-cross-lab-pro.jsonl` | 3 | 0/3 | 33% | 0% |
| `arbiter-cross-lab.jsonl` | 3 | 3/3 | 33% | 100% |
| `arbiter-fable.jsonl` | 3 | 1/3 | 33% | 33% |
| `arbiter-opus.jsonl` | 3 | 0/3 | 33% | 0% |
| `arbiter-sonnet.jsonl` | 3 | 1/3 | 33% | 33% |
| `four-lab.jsonl` | 4 | 4/4 | 25% | 100% |
| `seat-deepseek-pro.jsonl` | 3 | 3/3 | 33% | 100% |
| `seat-fable.jsonl` | 3 | 1/3 | 33% | 33% |
| `live_mixed_lab.jsonl` | 3 | 0/3 | 33% | 0% |
| `live_three_lab.jsonl` | 3 | 0/3 | 33% | 0% |

## What this does and does not show

It measures whether **this** prober can identify authorship from
**these** sheets. A stronger prober may do better. With
31 guesses the error bars are wide, and a single
run cannot separate a small leak from sampling noise.

It does not measure whether the *protocol* is blind in general — only
that the schema removed enough signal to defeat one capable reader on
the sessions to hand.
