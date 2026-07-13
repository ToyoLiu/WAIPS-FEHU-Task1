# Why Choose the Label-aware Ensemble

## Final Candidate

Recommended final candidate:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

This run should be selected if the final submission prioritizes Task-1b Micro F1 and overall Task-1 performance.

## 1. It Gives the Best Overall Task-1b Score

Full dev set comparison:

| Method | Task-1a Micro F1 | Task-1b Micro F1 | Task-1b Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek baseline | 0.6490 | 0.5245 | 0.3599 | 0.5716 | 0.4845 | 0.0243 |
| Majority vote | 0.6507 | 0.5359 | 0.3532 | **0.6350** | 0.4636 | **0.0180** |
| Targeted balanced | **0.6555** | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |
| Label-aware ensemble | **0.6555** | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |

Compared with majority vote, the label-aware ensemble improves:

- Task-1b Micro F1: `0.5359 -> 0.5657` (`+0.0298`)
- Task-1b Macro F1: `0.3532 -> 0.3870` (`+0.0338`)
- Task-1b Recall: `0.4636 -> 0.5238` (`+0.0602`)

The main cost is that DRR increases from `0.0180` to `0.0279`, but the score gain is large enough to make it the best current candidate.

## 2. It Combines Two Complementary Runs

The two source runs have different strengths:

| Run | Strength | Weakness |
|---|---|---|
| Majority vote | High precision, low DRR, stable common-label predictions | Misses difficult and infrequent labels |
| Targeted balanced | Higher recall, better macro F1, recovers difficult labels | Adds more false positives and direction errors |

The label-aware ensemble uses majority vote as the base, then adds targeted predictions only for selected labels where the targeted prompt clearly improved label-level performance.

This is different from a blind union. It is a selective combination guided by label-level evidence.

## 3. It Improves Difficult and Infrequent Labels

Compared with majority vote, the label-aware ensemble improves average F1 by label frequency group:

| Label group | Majority Avg F1 | Label-aware Avg F1 | Change |
|---|---:|---:|---:|
| Frequent labels | 0.4817 | 0.5109 | +0.0292 |
| Medium labels | 0.2866 | 0.2866 | +0.0000 |
| Infrequent labels | 0.2168 | 0.3025 | +0.0857 |

This is important because Professor Iwaihara suggested checking each label separately, especially difficult and infrequent labels.

The strongest effect appears on infrequent labels, where average F1 improves from `0.2168` to `0.3025`.

## 4. The Improvement Is Label-specific, Not Random

Examples of improved labels:

| Label | Majority F1 | Label-aware F1 | Main effect |
|---|---:|---:|---|
| contradictory: Have a stable society | 0.2545 | 0.5611 | large recall recovery |
| aligned: Have good health | 0.3188 | 0.5496 | recovered many missed labels |
| aligned: Be capable | 0.5070 | 0.6549 | higher recall with acceptable FP |
| contradictory: Have harmony with nature | 0.0000 | 0.8000 | infrequent label recovered |

These labels match the known weaknesses of majority vote: society, health, capability, and rare labels.

## 5. Why Not Choose the Other Runs?

### DeepSeek baseline

It is a strong prompt-only baseline, but it is weaker than the later ensemble methods:

- Task-1b Micro F1: `0.5245`
- Task-1b Macro F1: `0.3599`

It is useful as a baseline, but not the best submission candidate.

### Majority vote

It has the best precision and DRR, but it loses too much recall:

- Recall drops to `0.4636`
- Macro F1 is only `0.3532`

This makes it a good precision-oriented backup, but not the best single-run submission if Micro F1 is prioritized.

### Targeted balanced

It improves recall, but it is less stable:

- DRR increases to `0.0302`
- Precision drops to `0.5750`

It is useful as a high-recall run, but weaker than the label-aware ensemble.

### small_low_drr

This is a useful backup because it lowers DRR:

| Run | Task-1b Micro F1 | Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|
| Label-aware ensemble | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |
| small_low_drr | 0.5613 | 0.3858 | **0.6222** | 0.5113 | **0.0230** |

However, it slightly sacrifices Task-1b Micro F1 and Recall, so it should be treated as a lower-DRR backup rather than the main candidate.

## Final Decision

If only one run can be submitted, the current best choice is:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

Reason:

- It achieves the best Task-1b Micro F1 and Macro F1.
- It keeps Task-1a Micro F1 tied for best.
- It improves recall without dropping precision as much as targeted balanced.
- It directly follows the professor's advice by using label-level analysis.
- Its improvement is explainable: it recovers selected difficult labels rather than blindly adding all predictions.

## Short Slide Message

The label-aware ensemble is selected because it gives the best Task-1b score while remaining explainable. It keeps the stable majority-vote predictions, then selectively adds targeted predictions only for labels where label-level analysis showed clear improvement. This especially helps difficult and infrequent labels, which were the main weakness of the conservative ensemble.
