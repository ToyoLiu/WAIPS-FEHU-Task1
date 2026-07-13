# Smaller Label-aware Ensemble Experiment

## Purpose

The previous label-aware ensemble used 10 selected Task-1b labels.  
This experiment checks whether a smaller selected-label set can reduce DRR while keeping most of the Task-1b gain.

No new LLM calls were made. This is a post-processing experiment using existing runs:

- Base run: `dev_deepseek_ensemble_majority_vote_full`
- Target run: `dev_deepseek_deepseek-v4-flash_targeted_more_shot_balanced_temp0p1_max8000`

## Candidate Sets Tested

| Run tag | Selected labels | Added Task-1b predictions |
|---|---|---:|
| `dev_deepseek_ensemble_label_aware_targeted_balanced` | original 10 labels | 291 |
| `dev_deepseek_ensemble_label_aware_small_best9` | original labels except `1:35` | 289 |
| `dev_deepseek_ensemble_label_aware_small_low_drr` | original labels except `1:27` | 219 |
| `dev_deepseek_ensemble_label_aware_small_core` | `0:27, 0:48, 1:13, 1:22` | 162 |
| `dev_deepseek_ensemble_label_aware_small_health_society` | `0:22, 0:27, 1:22, 1:27` | 225 |
| `dev_deepseek_ensemble_label_aware_small_selected6` | `0:0, 0:27, 0:44, 0:48, 1:13, 1:22` | 174 |

## Main Results

### Compared with original label-aware ensemble

| Run | Task-1b Micro F1 | Macro F1 | Precision | Recall | DRR | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Original label-aware | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 | best score |
| small_best9 | 0.5656 | 0.3851 | 0.6150 | 0.5234 | 0.0279 | almost identical to original |
| small_low_drr | 0.5613 | 0.3858 | **0.6222** | 0.5113 | **0.0230** | better stability / lower DRR |

### Compared with majority vote

`small_low_drr` still clearly improves over majority vote:

| Metric | Majority vote | small_low_drr | Diff |
|---|---:|---:|---:|
| Task-1b Micro F1 | 0.5359 | 0.5613 | +0.0254 |
| Task-1b Macro F1 | 0.3532 | 0.3858 | +0.0325 |
| Precision | 0.6350 | 0.6222 | -0.0128 |
| Recall | 0.4636 | 0.5113 | +0.0477 |
| DRR | 0.0180 | 0.0230 | +0.0050 |

## Search Finding

I also searched all subsets of the original 10 selected labels.

- The original 10-label set still gives the best Task-1b Micro F1.
- The best smaller set with DRR <= 0.0230 is `small_low_drr`.
- `small_low_drr` removes `1:27` (`aligned: Have a stable society`) from the selected-label set.

## Current Interpretation

The smaller experiment did not beat the original label-aware ensemble on Micro F1.  
However, it provides a useful alternative submission candidate:

- Original label-aware ensemble: best score-oriented candidate.
- small_low_drr: more conservative candidate with lower DRR and slightly higher precision.

If only one run can be submitted and the official priority is Micro F1, the original label-aware ensemble is still the strongest candidate.  
If direction stability is considered important, `small_low_drr` is a reasonable backup.
