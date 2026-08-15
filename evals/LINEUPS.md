# Lineup experiment

> **n=1 per lineup.** Every arm ran the question once. Model sampling is
> stochastic, so a difference below is a hypothesis worth testing at scale,
> not a result. Reported anyway, because measuring once and saying so beats
> not measuring.

Question held constant across all lineups. Prices are estimates.

> **The `x baseline` column below is not comparable across rows.** The baseline
> defaults to the priciest seat *in that council*, so changing the lineup moves
> the yardstick too. `arbiter-sonnet` reads 27.4x against `arbiter-opus`'s 13.3x
> while costing **less** in absolute terms. Recomputed against one fixed ruler
> (Opus pricing on each session's own round-1 tokens) the order is:
> opus 13.3x · sonnet 16.4x · fable 12.3x · deepseek-pro 15.0x · fable-seat 15.8x.
> The eval now pins `baseline_model` so future runs do not repeat this.

| lineup | labs | objections | moved | minority | compliance | repairs | cost | x baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `arbiter-opus` | 3 | 22 | 2/3 | 11 | 82% | 1 | $0.3177 | 13.3x |
| `arbiter-sonnet` | 3 | 18 | 2/3 | 10 | 69% | 3 | $0.2981 | 27.4x |
| `arbiter-fable` | 3 | 20 | 2/3 | 12 | 69% | 3 | $0.2957 | 19.1x |
| `seat-deepseek-pro` | 3 | 27 | 1/3 | 13 | 100% | 0 | $0.4016 | 15.0x |
| `seat-fable` | 2 | 31 | 2/3 | 15 | 91% | 1 | $0.4920 | 16.7x |

## What it found

**The arbiter tier barely moves the bill.** $0.2957–$0.3177 across three
arbiters — a 7% spread, not the 40% saving predicted from "the arbiter is 69%
of cost". That prediction came from one earlier session with an unusually
long transcript; on these, arbiter choice is close to free. The spec's open
question stays open, and it is now a *quality* question rather than a cost one.

**Upgrading the weak seat fixed compliance outright.** `deepseek-chat` failed
the critique round repeatedly (69–82% compliance, 1–3 repairs every run).
`deepseek-v4-pro` scored **100% with zero repairs**, and raised more objections
than any three-lab lineup. An earlier round-1-only test had suggested tier did
not matter; round 2 is where it discriminates, and that test was too easy.

**A prediction of mine did not survive.** I argued from arithmetic that fewer
labs should mean less disagreement, and that `seat-fable` — three students but
only two labs — would show it. It produced the **most** objections (31) and the
most minority items (15) of any lineup. Objection count is evidently tracking
model capability, not diversity of priors, so it is the wrong proxy for the
thing the protocol sells. Whether the *content* of those objections is more
redundant is a real question this harness cannot yet answer.

## Lineups

- **`arbiter-opus`** — baseline — strongest arbiter, the current default  
  students: claude-sonnet-5, gpt-5.1, deepseek-chat · arbiter: claude-opus-5  
  weakest complier: `deepseek-chat`  

- **`arbiter-sonnet`** — mid-tier arbiter — the spec's cheaper alternative  
  students: claude-sonnet-4-6, gpt-5.1, deepseek-chat · arbiter: claude-sonnet-5  
  weakest complier: `deepseek-chat`  

- **`arbiter-fable`** — a different Claude line arbitrating  
  students: claude-sonnet-5, gpt-5.1, deepseek-chat · arbiter: claude-fable-5  
  weakest complier: `deepseek-chat`  

- **`seat-deepseek-pro`** — weak seat upgraded within the same lab  
  students: claude-sonnet-5, gpt-5.1, deepseek-v4-pro · arbiter: claude-opus-5  

- **`seat-fable`** — weak seat replaced by a fourth Claude line — costs a lab  
  students: claude-sonnet-5, gpt-5.1, claude-fable-5 · arbiter: claude-opus-5  
  weakest complier: `claude-fable-5`  
