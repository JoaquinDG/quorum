# Quorum benchmark

> **THIS IS A MOCK RUN. THESE ARE NOT RESULTS.**
>
> Every arm was answered by `MockProvider`, which emits canned text. The numbers below demonstrate that the harness runs end to end and measure nothing else. Real numbers require real providers and keys.

- tasks: **20**
- council: atlas-3, borealis-2, cinder-1
- single-model arm: delphi-frontier
- judge: judge-model

## Mean rubric score

| arm | all criteria | neutral criteria only | total cost |
| --- | --- | --- | --- |
| `quorum` | 0.673 | 0.673 | $0.7357 |
| `single` | 0.672 | 0.670 | $0.0238 |
| `self_critique` | 0.704 | 0.708 | $0.0533 |

*Neutral criteria* exclude the ones tagged `favours_deliberation` — the ones a deliberative process gets more easily. If an arm leads on all criteria but not on neutral ones, it is winning on format rather than on substance, and that column is where you would see it.

## Wins per task

| arm | all criteria | neutral criteria only |
| --- | --- | --- |
| `quorum` | 6 | 7 |
| `single` | 4 | 3 |
| `self_critique` | 9 | 9 |
| `tie` | 1 | 1 |
| `none` | 0 | 0 |

## Per task

| task | quorum | single | self-critique | winner |
| --- | --- | --- | --- | --- |
| `pipeline_rebuild` | 0.59 | 0.60 | 0.51 | single |
| `usage_pricing` | 0.71 | 0.66 | 0.72 | self_critique |
| `seniors_vs_juniors` | 0.73 | 0.59 | 0.73 | tie |
| `deprecate_v1` | 0.65 | 0.65 | 0.66 | self_critique |
| `single_tenant` | 0.62 | 0.75 | 0.79 | self_critique |
| `build_vs_buy_auth` | 0.53 | 0.57 | 0.73 | self_critique |
| `open_source_core` | 0.68 | 0.67 | 0.69 | self_critique |
| `remote_office` | 0.71 | 0.70 | 0.69 | quorum |
| `tech_debt_quarter` | 0.75 | 0.63 | 0.72 | quorum |
| `enterprise_pivot` | 0.66 | 0.71 | 0.80 | self_critique |
| `incident_disclosure` | 0.58 | 0.80 | 0.84 | self_critique |
| `free_tier` | 0.78 | 0.64 | 0.75 | quorum |
| `monorepo_split` | 0.63 | 0.67 | 0.74 | self_critique |
| `acquire_competitor` | 0.75 | 0.77 | 0.73 | single |
| `founder_role` | 0.70 | 0.78 | 0.66 | single |
| `eu_region` | 0.72 | 0.62 | 0.50 | quorum |
| `ai_feature_bet` | 0.71 | 0.53 | 0.69 | quorum |
| `support_outsource` | 0.63 | 0.75 | 0.63 | single |
| `raise_now` | 0.65 | 0.67 | 0.86 | self_critique |
| `migrate_off_cloud` | 0.68 | 0.68 | 0.64 | quorum |
