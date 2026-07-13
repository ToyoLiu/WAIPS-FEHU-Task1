# Label-aware Ensemble Experiment Summary

## 1. Experiment Matrix

### Full Dev Set Results

| Method | Main idea | Task-1a Micro F1 | Task-1b Micro F1 | Task-1b Macro F1 | Precision | Recall | DRR |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4o A0 | Previous prompt-only baseline | 0.4810 | 0.3424 | 0.1496 | 0.5696 | 0.2448 | 0.0212 |
| DeepSeek baseline | DeepSeek V4 Flash with original A0-style prompt | 0.6490 | 0.5245 | 0.3599 | 0.5716 | 0.4845 | 0.0243 |
| compact | Remove reasoning field, shorter JSON output | 0.6441 | 0.5261 | 0.3511 | 0.5698 | 0.4887 | 0.0261 |
| compact_strict | Conservative prompt, stronger evidence requirement | 0.6125 | 0.5062 | 0.3234 | 0.6247 | 0.4255 | 0.0180 |
| majority vote | Keep predictions supported by prompt variants | 0.6507 | 0.5359 | 0.3532 | 0.6350 | 0.4636 | 0.0180 |
| targeted_more_shot_balanced | Add targeted examples for difficult labels | 0.6555 | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |
| label-aware ensemble | Majority vote + selected targeted labels | 0.6555 | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |

### Main Observations

- DeepSeek V4 Flash produced a much stronger prompt-only baseline than the previous GPT-4o A0 result.
- `majority vote` improved precision and reduced direction errors, but lost recall.
- `targeted_more_shot_balanced` improved recall and macro F1, especially for difficult labels.
- `label-aware ensemble` combined both strengths and achieved the best Task-1b Micro F1 and Macro F1.

## 2. Label-level Behavior

### Frequency Group Comparison

Compared with `majority vote`:

| Label group | Majority F1 | Label-aware F1 | Change |
|---|---:|---:|---:|
| Frequent labels | 0.4817 | 0.5109 | +0.0292 |
| Medium labels | 0.2866 | 0.2866 | +0.0000 |
| Infrequent labels | 0.2168 | 0.3025 | +0.0857 |

This shows that the improvement is not only from frequent labels. The label-aware ensemble especially improves infrequent labels, which matches the professor's suggestion to evaluate individual labels separately.

### Labels Improved by Label-aware Ensemble

| Label | Majority F1 | Label-aware F1 | Main effect |
|---|---:|---:|---|
| contradictory: Have a stable society | 0.2545 | 0.5611 | large recall recovery |
| aligned: Have good health | 0.3188 | 0.5496 | many missed labels recovered |
| aligned: Be capable | 0.5070 | 0.6549 | better recall with acceptable FP |
| aligned: Have a stable society | 0.4965 | 0.6103 | stable improvement |
| contradictory: Have harmony with nature | 0.0000 | 0.8000 | infrequent label recovered |

## 3. Slide Draft: Why Label-aware Ensemble Works

### Slide Title

Why Label-aware Ensemble Works

### Subtitle

Combining stable precision with targeted recall recovery

### Main Layout

#### Left: Two Models Have Different Strengths

**Majority vote**

- Higher precision
- Lower DRR
- Stable on common labels
- But misses some difficult / infrequent labels

**Targeted more-shot balanced**

- Higher recall
- Better macro F1
- Recovers health / society labels
- But introduces more false positives and direction errors

#### Center: Label-aware Rule

Start from `majority vote`, then add predictions from `targeted_more_shot_balanced` only for labels where targeted prompting showed clear improvement.

Selected labels:

- `Have a stable society`
- `Have good health`
- `Be capable`
- `Have harmony with nature`
- `Be creative`
- `Have equality`

#### Right: Result

| Metric | Majority vote | Label-aware |
|---|---:|---:|
| Task-1b Micro F1 | 0.5359 | **0.5657** |
| Task-1b Macro F1 | 0.3532 | **0.3870** |
| Precision | 0.6350 | 0.6149 |
| Recall | 0.4636 | **0.5238** |
| DRR | 0.0180 | 0.0279 |

### Takeaway

The ensemble works because it does not combine runs blindly. It uses label-level analysis to keep majority vote's stable predictions while selectively recovering labels that targeted prompting handles better.

## 4. Speaker Notes for This Slide

This slide explains why the label-aware ensemble improved the result.

The majority-vote run was useful because it was more conservative. It had higher precision and a lower direction reverse rate, but it also removed many true labels, especially difficult or infrequent ones.

On the other hand, the targeted more-shot balanced prompt had higher recall and better macro F1. It recovered labels such as "Have good health" and "Have a stable society", but it also introduced more false positives.

So I did not simply take the union of all predictions. Instead, I used majority vote as the base, and only added targeted predictions for labels where the targeted prompt clearly performed better.

As a result, Task-1b Micro F1 improved from 0.5359 to 0.5657, and Macro F1 improved from 0.3532 to 0.3870. This suggests that label-level analysis can guide a more effective ensemble strategy.

## 5. Next Analysis Direction

- Check whether the selected label list is too broad or too narrow.
- Try a stricter label-aware ensemble with fewer labels to reduce DRR.
- Compare label-aware ensemble with Peiyang's hybrid method under the same evaluation setting.
- Prepare a formal submission strategy: one high-recall run, one precision-oriented run, and one label-aware ensemble run.
