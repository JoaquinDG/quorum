# Quorum benchmark

> **NOT A RESULT — arms are missing.**
>
> 49 task/arm pair(s) produced no scored answer, affecting: `quorum`, `self_critique`, `single`. Any score below is a comparison against an opponent that did not answer, and must not be read as a finding. Common cause: a provider out of credit, or a model id that has been retired.

- tasks: **20**
- council: claude-sonnet-5, gpt-5.1, deepseek-v4-pro
- single-model arm: claude-opus-5
- judge: kimi-k2.6

## Mean rubric score

| arm | all criteria | neutral criteria only | total cost |
| --- | --- | --- | --- |
| `quorum` | 0.547 | 0.644 | $4.0577 |
| `single` | 0.960 | 0.958 | $0.7486 |
| `self_critique` | 0.980 | 0.983 | $2.9349 |

*Neutral criteria* exclude the ones tagged `favours_deliberation` — the ones a deliberative process gets more easily. If an arm leads on all criteria but not on neutral ones, it is winning on format rather than on substance, and that column is where you would see it.

## Wins per task

| arm | all criteria | neutral criteria only |
| --- | --- | --- |
| `quorum` | 0 | 0 |
| `single` | 0 | 0 |
| `self_critique` | 2 | 2 |
| `tie` | 2 | 2 |
| `none` | 16 | 16 |

## Per task

| task | quorum | single | self-critique | winner |
| --- | --- | --- | --- | --- |
| `pipeline_rebuild` | err | err | err | none |
| `usage_pricing` | err | err | err | none |
| `seniors_vs_juniors` | err | 1.00 | 1.00 | tie |
| `deprecate_v1` | err | err | err | none |
| `single_tenant` | err | err | err | none |
| `build_vs_buy_auth` | err | err | err | none |
| `open_source_core` | err | err | err | none |
| `remote_office` | err | err | err | none |
| `tech_debt_quarter` | 0.34 | 0.94 | 1.00 | self_critique |
| `enterprise_pivot` | err | err | err | none |
| `incident_disclosure` | err | err | err | none |
| `free_tier` | err | err | err | none |
| `monorepo_split` | err | err | err | none |
| `acquire_competitor` | err | err | err | none |
| `founder_role` | 0.64 | 1.00 | 1.00 | tie |
| `eu_region` | err | err | err | none |
| `ai_feature_bet` | 0.66 | 0.90 | 0.92 | self_critique |
| `support_outsource` | err | err | err | none |
| `raise_now` | err | err | err | none |
| `migrate_off_cloud` | err | err | err | none |
