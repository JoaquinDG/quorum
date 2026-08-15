# Quorum, deliberating on Quorum

The project claims a council beats one model on high-stakes ambiguous
judgement calls. These are the project's own unresolved design questions,
which are exactly that shape — and the author has a stake in the answers,
which is when a single confident reply is least trustworthy.

Council: claude-sonnet-5, gpt-5.1, deepseek-v4-pro  
Arbiter: gemini-3.1-pro-preview (shares a lab with no student)

| question | objections | moved | dissents kept | cost |
| --- | --- | --- | --- | --- |
| `minority_bar` | 28 | 3/3 | 4 | $0.2040 |
| `cost_guardrail` | 29 | 3/3 | 6 | $0.2255 |
| `wording_metric` | 25 | 3/3 | 3 | $0.1907 |
| `council_size` | 28 | 3/3 | 9 | $0.1895 |

## minority_bar

**Quorum's minority report currently preserves every dissent an arbiter leaves out of the final answer. A real session produced 13 items, which is thorough but works against the stated goal of a report a non-technical reader can follow in under three minutes. Should the arbiter apply a materiality bar and curate, or does curation become the synthesis-away the feature exists to prevent?**

### What the council decided

The arbiter should not resolve the tension by silently dropping minor dissents into an appendix. Participants agreed that relegating items to a secondary record functionally erases them for time-constrained, non-technical readers, thereby recreating the 'synthesis-away' problem the minority report was built to prevent. Instead, the arbiter should manage the 13 items through structured tiering and compression. By establishing an explicit, auditable materiality bar, the arbiter can give full explanations to the most critical dissents while heavily compressing the rest. Dissents falling below the bar are not omitted from the primary document, but are presented as grouped, one-line gists (e.g., '8 additional items deemed less material: X, Y, Z'). This approach leverages formatting and progressive disclosure to respect the three-minute reading budget without allowing invisible, unaccountable suppression.

*The participants reached a strong practical compromise (tiering and summarizing instead of binary inclusion/omission), but remain philosophically divided on whether an arbiter can be trusted to filter dissent at all. Furthermore, convergence depends entirely on the untested empirical assumption that 13 compressed items can actually be processed in three minutes without causing cognitive overload.*

### Dissent it refused to drop

- **[Student 1]** When it is ambiguous whether a dissent is material, the arbiter should default to inclusion in the main summary; if this threatens the time budget, it means the criteria are mis-specified and must be tightened, not that the default-to-inclusion rule should be abandoned.
- **[Student 2]** Because the arbiter is the author of the majority synthesis, they are inherently non-neutral and should not be permitted to act as the gatekeeper for what is 'material' enough to include in the minority report.
- **[Student 2]** The defining purpose of the minority report is the exhaustiveness of dissent; three-minute readability is strictly a subordinate constraint that requires better formatting, not a primary goal that justifies any form of curation.
- **[Student 3]** Curation differs procedurally and conceptually from synthesis-away because it formally omits rather than dilutes or absorbs minority reasoning, even if both result in a loss of information for the end reader.

## cost_guardrail

**Quorum's spec sets a guardrail of at most 8x the cost of a single model answer. Eight measured sessions came in between 6x and 26x, and the multiple turned out not to be comparable across lineups because each session's baseline is measured from its own round one. Should the guardrail be abandoned, restated as absolute cost per session, or kept as an aspiration the project keeps missing?**

### What the council decided

Quorum should adopt a dual-pronged approach to its cost guardrails. The primary project-wide guardrail should be restated as an absolute cost per session, categorized into budget tiers (or normalized by task class) to account for legitimate differences in model mix, scope, and difficulty across different lineups. At the same time, the current 'at most 8x' metric should not be abandoned; rather, it should be retained as a within-session diagnostic tool and soft heuristic. Used this way, the multiplier effectively bounds iterative overhead relative to a session's own starting point, acknowledging that while it is useful locally, it cannot serve as a rigid, cross-session comparative budget.

*The participants reached a strong consensus. They initially debated choosing between a rigid absolute cap and fixing the multiple, but ultimately converged on using both mechanisms for different purposes. The resolution turns on the shared recognition that a single absolute cap would unfairly penalize complex tasks, while a single multiplier fails as a predictable global budget constraint.*

### Dissent it refused to drop

- **[Student 1]** Creating separate absolute budgets for different experiment types silently reintroduces a classification problem that could be just as complex to manage as the varying baselines the project is trying to escape.
- **[Student 1]** There is no actual evidence that teams will obey an absolute cost limit any more faithfully than they followed the 8x relative guardrail.
- **[Student 2]** The 6x-26x spread in costs might reflect measurement noise, inconsistent enforcement, or justified overrides for complex tasks, rather than proving the rule is purely decorative.
- **[Student 2]** Without perfect normalization, absolute budgets risk being arbitrarily restrictive for difficult lineups while remaining trivially loose for simple ones.
- **[Student 3]** Violating a cap does not prove the cap had no effect; without a counterfactual, it is entirely possible these sessions would have been far more expensive if the 8x guardrail had not existed.
- **[Student 3]** A denominator that fluctuates depending on the session's initial baseline may actually be a feature, not a defect, because it automatically scales the allowable budget to match local task difficulty.

## wording_metric

**Quorum shipped a disagreement score, then demoted it after it called a unanimous council 'sharply contested' — it measures vocabulary overlap, not agreement. It is now labelled wording spread and nothing in the protocol depends on it. Should it be deleted outright, kept as a clearly labelled heuristic, or replaced with a semantic measure that costs an extra model call per session?**

### What the council decided

The consensus is to **delete the metric outright** rather than keeping it or immediately replacing it.

All three participants initially argued for keeping the metric as a relabeled 'wording spread' heuristic. However, throughout the debate, every participant completely reversed their position. They concluded that the core issue was a construct-validity failure, not merely a labeling problem: vocabulary overlap is fundamentally distinct from semantic disagreement. Because the metric produces a precise-looking number, users will likely continue to intuitively misinterpret it as substantive contention regardless of how carefully it is named or documented (e.g., a unanimous council using varied phrasing will still look 'contested').

Furthermore, the participants recognized that keeping an exposed but unused metric is not costless. It imposes UI clutter, cognitive overhead, and a standing risk of downstream misuse. Since nothing in the protocol currently depends on this metric, deletion is the safest and least complex option.

The participants also agreed to reject the immediate introduction of a costly semantic measure. An extra model call per session is not justified without a concrete use case. A proper semantic metric should only be built, benchmarked, and validated when a specific protocol component genuinely requires a true measure of agreement.

*The participants reached a strong, unanimous consensus. This is a highly settled outcome where all three participants independently abandoned their initial positions to converge on deletion. The agreement turns on the realization that relabeling does not neutralize the intuitive pull of a misleading numerical score, and that retaining 'inert' metrics carries hidden human-factor and maintenance costs.*

### Dissent it refused to drop

- **[Student 2]** Deferring the development of a principled semantic agreement measure until a specific consumer requires it assumes that sufficient coordination and refactoring capacity will exist at that time. Designing and building it early could proactively shape better APIs and data contracts before the pressure to ship a new feature hits.
- **[Student 2]** If a cheap semantic approximation becomes available that does not require an extra model call—such as reusing existing embeddings or logits—it could justify introducing a true disagreement metric under stricter evidentiary standards without the prohibitive compute cost.
- **[Student 1]** Deleting the exposed metric does not truly sacrifice optionality. As long as the internal code and raw session data are retained, any needed signal can be cheaply recomputed retroactively if a valid, concrete use case ever emerges.

## council_size

**Quorum caps a council at five students and defaults to three. Adding a fourth distinct lab measurably improved schema compliance and objection count at no extra cost in one session, contradicting the prediction that a fourth seat costs 1.4x. Should the default council size rise to four, stay at three, or become adaptive based on how contested the question appears?**

### What the council decided

The debate resulted in a three-way split, with each participant ultimately advocating for a different policy option. The core disagreement centers on how heavily to weight a single successful test session against a prior model predicting a 1.4x cost increase, and how practical an adaptive system would be.

1. **Raise the default to four (Student 1's final position):** The prior 1.4x cost model was empirically contradicted by the only test available, which showed improved proxy metrics at no extra cost. Because institutional defaults are easily revisable and building an adaptive system requires unvalidated triggers, provisionally moving to four is the most evidence-aligned choice.

2. **Become adaptive (Student 2's final position):** Neither a strict three- nor four-seat default has robust, large-scale empirical validation across different query types. Designing a system that uses three seats as a baseline but escalates to four when signals of contestation are detected provides the best operational hedge, provided the triggers can be refined over time.

3. **Stay at three (Student 3's final position):** Institutional defaults require strong evidence to change. A single, potentially noisy data point does not statistically invalidate a probabilistic 1.4x cost prediction. Furthermore, it is not yet proven that the observed proxy improvements (schema compliance and objection counts) actually result in better final decisions, making it premature to adopt either a four-seat default or a complex adaptive scheme.

*The participants completely diverged, failing to reach any consensus. Each student ultimately adopted a different one of the three available options. The decision turns heavily on the evidentiary standard required to change an established default, whether one contradictory test session is viewed as a definitive falsification or a statistical anomaly, and the technical feasibility of building non-circular, low-latency adaptive triggers.*

### Dissent it refused to drop

- **[Student 1]** Candidate signals for adaptive sizing (e.g., early disagreement, initial schema violations) are plausibly endogenous to council size and composition, creating circularity and gaming risks.
- **[Student 1]** The hypothesis that contested questions benefit more from added perspectives is purely speculative, as the test session did not measure or vary contestation levels.
- **[Student 1]** It is an inconsistent standard to demand strong evidence for the cost-effectiveness of adaptive sizing while accepting a single session as sufficient to permanently raise the fixed default.
- **[Student 2]** A single session showing no extra cost is perfectly compatible with a probabilistic 1.4x average cost prediction, due to variance, measurement noise, or uncaptured costs like cognitive load and coordination time.
- **[Student 2]** Higher objection counts and schema compliance might indicate noise, confusion, overcautiousness, or overfitting rather than genuinely better substantive reasoning.
- **[Student 2]** Keeping the previously modeled 1.4x cost prediction 'in the background' without quantifying its remaining credence after a conflicting test provides no operational guidance and unfairly biases analysis toward retaining three seats.
- **[Student 3]** If contestation is only detected after a three-seat council has already produced a defective output, an adaptive scheme could incur rework costs or missed benefits greater than simply starting with four seats.
- **[Student 3]** Defenders of the three-seat default impose an asymmetric evidence burden by demanding large-scale validation to move to four seats, without providing equivalent empirical validation that three seats is mathematically optimal.
- **[Student 3]** Framing a default change as a 'permanent structural change' falsely raises the stakes; defaults are easily revisable tools used while collecting more evidence.
