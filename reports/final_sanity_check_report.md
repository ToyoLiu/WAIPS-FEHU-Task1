# Final Sanity Check Report

## Checked Runs

Primary candidate:

- `dev_deepseek_ensemble_label_aware_targeted_balanced`

Backup / lower-DRR candidate:

- `dev_deepseek_ensemble_label_aware_small_low_drr`

## Primary Candidate Result

Full dev evaluation:

| Metric | Score |
|---|---:|
| Task-1a Micro F1 | 0.6555 |
| Task-1a Macro F1 | 0.5044 |
| Task-1b Micro F1 | 0.5657 |
| Task-1b Macro F1 | 0.3870 |
| Task-1b Precision | 0.6149 |
| Task-1b Recall | 0.5238 |
| DRR | 0.0279 |

## Schema / ID Check

Primary candidate:

| Check item | Result |
|---|---|
| `pred_task1a.json` exists | PASS |
| `pred_task1b.json` exists | PASS |
| JSON parse | PASS |
| Dev guid coverage | 177 / 177 |
| Task-1a duplicate predictions | 0 |
| Task-1b duplicate predictions | 0 |
| Missing dev guids | 0 |
| Invalid actor IDs | 0 |
| Invalid labels / directions | 0 |
| Task-1b L1 values whose mapped L2 is absent from Task-1a | 0 |

Backup candidate `small_low_drr` also passed the same checks.

## Notes

- The current primary candidate is structurally safe on the dev set.
- The lower-DRR backup is also structurally safe.
- The repository's `evaluation.py` currently has `DEBUG = True`, which overrides command-line paths when run directly. Existing comparison scripts import its functions and are not affected.

## Current Submission Interpretation

If the final decision is based mainly on Task-1b Micro F1, the primary candidate remains:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

If direction stability is prioritized more strongly, the backup candidate is:

`dev_deepseek_ensemble_label_aware_small_low_drr`
