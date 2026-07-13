# FEHU Task-1 Submission Strategy - Slide Draft

Presenter: Liu Dongyang  
Meeting: Group Seminar, July 2026  
Topic: Final candidate selection and submission-oriented analysis for FEHU Task-1

---

## Slide 1. Title

### Toward Final Submission for FEHU Task-1

- Liu Dongyang
- Data Engineering Lab, IPS, Waseda University
- NTCIR-19 FEHU Task-1 progress report

**Layout Suggestion**

Use a simple title slide. Add one short subtitle:

> From label-level analysis to final candidate selection

**Speaker Notes**

Good afternoon everyone. Today I will report my recent progress on FEHU Task-1. In the previous presentation, I introduced label-level analysis and the label-aware ensemble. This time, I focus more on the submission stage: checking the final candidate, comparing smaller variants, and explaining why I choose the current run.

---

## Slide 2. Outline

### Outline

- Previous best run: label-aware ensemble
- Final sanity check
- Smaller label-aware experiment
- Why choose the current candidate
- Remaining risks and submission plan

**Layout Suggestion**

Use large bullets, similar to your previous outline slide. Keep each line short.

**Speaker Notes**

The structure of this presentation is as follows. First, I briefly review the previous best run. Then I show the final sanity check. After that, I explain a small experiment on smaller label-aware ensembles, and finally discuss why the current candidate should be selected for submission.

---

## Slide 3. Previous Best Run

### Previous Best: Label-aware Ensemble

#### Main idea

`label-aware ensemble = majority vote base + selected targeted labels`

#### Why it was proposed

- `majority vote`: high precision and low DRR, but conservative
- `targeted_more_shot_balanced`: higher recall, but less stable
- Label-level analysis showed which labels benefit from targeted prompting
- The ensemble adds targeted predictions only for selected labels

#### Previous result

| Method | Task-1b Micro F1 | Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|
| Majority vote | 0.5359 | 0.3532 | **0.6350** | 0.4636 | **0.0180** |
| Targeted balanced | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |
| Label-aware ensemble | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |

**Layout Suggestion**

Top: formula.  
Middle: two source-run boxes, `majority vote` and `targeted balanced`.  
Bottom: compact result table.

**Speaker Notes**

The previous best run was the label-aware ensemble. It uses majority vote as the base because majority vote is precise and stable. Then it adds predictions from the targeted balanced prompt only for selected labels where label-level analysis showed clear improvement. This gave the best Task-1b Micro F1 and Macro F1 so far.

---

## Slide 4. Final Sanity Check

### Final Sanity Check: Is the Candidate Submit-ready?

Checked run:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

#### Structural checks

| Check item | Result |
|---|---|
| `pred_task1a.json` exists | PASS |
| `pred_task1b.json` exists | PASS |
| JSON parse | PASS |
| Dev guid coverage | 177 / 177 |
| Duplicate predictions | 0 |
| Missing dev guids | 0 |
| Invalid actor IDs | 0 |
| Invalid labels / directions | 0 |
| Task-1b mapped Level-2 absent from Task-1a | 0 |

#### Result

> The current candidate is structurally safe on the full dev set.

**Layout Suggestion**

Use a checklist table. Put a large `PASS` mark or blue box on the right.

**Speaker Notes**

Before deciding the final candidate, I checked whether the prediction files are structurally safe. I verified JSON parsing, coverage of all 177 dev articles, duplicate predictions, actor IDs, labels, directions, and consistency between Task-1a and Task-1b. The current candidate passed these checks.

---

## Slide 5. Main Candidate Result

### Current Best Candidate

Full dev result:

| Metric | Score |
|---|---:|
| Task-1a Micro F1 | 0.6555 |
| Task-1a Macro F1 | 0.5044 |
| Task-1b Micro F1 | **0.5657** |
| Task-1b Macro F1 | **0.3870** |
| Task-1b Precision | 0.6149 |
| Task-1b Recall | **0.5238** |
| DRR | 0.0279 |

#### Main gain over majority vote

- Task-1b Micro F1: `+0.0298`
- Task-1b Macro F1: `+0.0338`
- Recall: `+0.0602`

**Layout Suggestion**

Use three large number cards:

- `T1b Micro F1 0.5657`
- `T1b Macro F1 0.3870`
- `Recall 0.5238`

Put the full metric table below or on the right.

**Speaker Notes**

This is the current best candidate. Compared with majority vote, Task-1b Micro F1 improves by about 0.03, Macro F1 improves by about 0.034, and Recall improves by about 0.06. The cost is that DRR is higher than majority vote, but the overall Task-1b score is clearly better.

---

## Slide 6. Smaller Label-aware Experiment

### Smaller Label-aware Ensemble: Can We Reduce DRR?

#### What is smaller label-aware ensemble?

The original label-aware ensemble uses:

`majority vote base + targeted predictions for 10 selected labels`

The smaller version keeps the same base run, but reduces the selected-label list:

`majority vote base + targeted predictions for fewer selected labels`

This means the model predictions are not regenerated.  
Only the post-processing rule is changed.

#### Why test smaller selected-label sets?

- The original label-aware ensemble gives the best Task-1b score
- However, its DRR is higher than majority vote
- Some selected labels may add useful recall but also introduce direction errors
- A smaller label set may keep most F1 gain while reducing DRR

#### Candidate variants

| Variant | Selected-label idea | Purpose |
|---|---|---|
| Original label-aware | 10 selected labels | Best score-oriented candidate |
| small_best9 | Remove one low-impact label | Check whether one label is unnecessary |
| small_low_drr | Remove label that increases direction errors | Lower-DRR backup |

#### Experiment setting

- Base run: `majority vote`
- Added source: `targeted_more_shot_balanced`
- Operation: post-processing only
- New LLM calls: none

#### Main comparison

| Run | Task-1b Micro F1 | Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|
| Original label-aware | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |
| small_best9 | 0.5656 | 0.3851 | 0.6150 | 0.5234 | 0.0279 |
| small_low_drr | 0.5613 | 0.3858 | **0.6222** | 0.5113 | **0.0230** |

**Layout Suggestion**

Use a three-part layout:

Top: simple formula comparison

- `Original = majority vote + 10 selected labels`
- `Smaller = majority vote + fewer selected labels`

Middle: candidate variant table.

Bottom: result table plus two callouts:

- `Original: best score`
- `small_low_drr: lower DRR backup`

**Speaker Notes**

The smaller label-aware experiment is a refinement of the previous label-aware ensemble. The original version adds targeted predictions for 10 selected labels. In this experiment, I kept majority vote as the base, but reduced the selected-label list. This is only post-processing, so I did not need new LLM calls.

The purpose was to check whether some selected labels were helping recall but also increasing direction errors. The result shows that the original label-aware ensemble still has the best Micro F1, but small_low_drr reduces DRR from 0.0279 to 0.0230 with only a small score loss. Therefore, the original version remains the main candidate, while small_low_drr is useful as a lower-DRR backup.

---

## Slide 7. Why Not Choose the Smaller One?

### Score-oriented vs Stability-oriented Candidate

#### If the priority is highest Task-1b Micro F1

Choose:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

- Best Task-1b Micro F1: `0.5657`
- Best Task-1b Macro F1: `0.3870`
- Best recall among stable candidates: `0.5238`

#### If the priority is lower DRR

Backup:

`dev_deepseek_ensemble_label_aware_small_low_drr`

- Lower DRR: `0.0230`
- Higher precision: `0.6222`
- Slightly lower Task-1b Micro F1: `0.5613`

**Layout Suggestion**

Use a two-column layout:

Left: `Score-oriented final candidate`  
Right: `Lower-DRR backup`

Add one bottom takeaway:

> Since only one run can be submitted, the score-oriented label-aware ensemble remains the current choice.

**Speaker Notes**

The smaller experiment gives a useful backup. If direction stability is strongly prioritized, small_low_drr is attractive. However, if only one run can be submitted and the main evaluation is based on Task-1b Micro F1, the original label-aware ensemble is still the best current choice.

---

## Slide 8. Why Choose It?

### Why the Label-aware Ensemble Is Explainable

#### It is not only a score improvement

- It follows label-level analysis
- It keeps stable majority-vote predictions
- It selectively recovers labels that targeted prompting handles better
- It avoids blind union of all predictions

#### Label-frequency effect

Compared with majority vote:

| Label group | Majority Avg F1 | Label-aware Avg F1 | Change |
|---|---:|---:|---:|
| Frequent labels | 0.4817 | 0.5109 | +0.0292 |
| Medium labels | 0.2866 | 0.2866 | +0.0000 |
| Infrequent labels | 0.2168 | 0.3025 | +0.0857 |

**Layout Suggestion**

Left: short bullet explanation.  
Right: frequency-group table.  
Bottom: strong takeaway.

> The main improvement comes from label-level selection, especially for difficult and infrequent labels.

**Speaker Notes**

The reason I choose this run is not only that it has the highest score. The improvement is explainable. It uses the conservative majority-vote run as the base, and only adds targeted predictions for labels where the targeted prompt clearly worked better. The strongest effect appears on infrequent labels, which matches Professor Iwaihara's suggestion to analyze labels separately.

---

## Slide 9. Label Examples

### Which Labels Were Recovered?

Examples improved by the label-aware ensemble:

| Label | Majority F1 | Label-aware F1 | Main effect |
|---|---:|---:|---|
| contradictory: Have a stable society | 0.2545 | 0.5611 | large recall recovery |
| aligned: Have good health | 0.3188 | 0.5496 | recovered missed labels |
| aligned: Be capable | 0.5070 | 0.6549 | higher recall |
| contradictory: Have harmony with nature | 0.0000 | 0.8000 | rare label recovered |

#### Interpretation

- The selected labels are not random
- Many are difficult, rare, or semantically important
- This supports the label-aware selection rule

**Layout Suggestion**

Use the table on the left.  
On the right, add a small diagram:

`Label-level analysis -> selected labels -> final ensemble`

**Speaker Notes**

Here are examples of labels recovered by the label-aware ensemble. For example, contradictory "Have a stable society" improved from 0.2545 to 0.5611, and aligned "Have good health" improved from 0.3188 to 0.5496. These are exactly the kinds of difficult labels that were identified by error analysis.

---

## Slide 10. Remaining Risks

### Remaining Risks Before Submission

#### Current risk

- DRR is not as low as majority vote
- Some added labels may still introduce false positives
- The selected label list may still be slightly broad

#### How I handle it

- Keep `small_low_drr` as a backup
- Do not continue broad prompt search
- Focus on submit-ready files and clear explanation
- Discuss final choice with Peiyang and align with the hybrid track

**Layout Suggestion**

Use two boxes:

1. `Risks`
2. `Mitigation`

Bottom message:

> The current goal is not only to improve the score, but to make the submission decision explainable.

**Speaker Notes**

There are still some risks. The DRR is higher than majority vote, and some added labels may introduce false positives. However, I now have a lower-DRR backup, and the main candidate has passed structural checks. So at this stage, I think the priority should be preparing a stable and explainable submission rather than continuing broad prompt search.

---

## Slide 11. Submission Plan

### Submission Plan

#### Current decision

Final candidate:

`dev_deepseek_ensemble_label_aware_targeted_balanced`

Backup candidate:

`dev_deepseek_ensemble_label_aware_small_low_drr`

#### Remaining work before final submission

1. Confirm the exact formal/test-set input files
2. Run the fixed final pipeline once
3. Apply the same sanity check to formal outputs
4. Compare with Peiyang's hybrid result
5. Decide the final single run for submission

**Layout Suggestion**

Use a vertical checklist. Mark completed items:

- Dev result fixed
- Sanity check script ready
- Candidate decision prepared

Leave formal/test run items unchecked.

**Speaker Notes**

For submission, my current candidate is the original label-aware ensemble. The lower-DRR version is a backup. The remaining work is to confirm the formal input files, run the final pipeline once, apply the same sanity check, and discuss with Peiyang how this prompt-only result compares with his hybrid approach.

---

## Backup Slide. Full Experiment Matrix

### Full Dev Set Results

| Method | Main idea | Task-1a Micro F1 | Task-1b Micro F1 | Task-1b Macro F1 | Precision | Recall | DRR |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4o A0 | Previous prompt-only baseline | 0.4810 | 0.3424 | 0.1496 | 0.5696 | 0.2448 | 0.0212 |
| DeepSeek baseline | DeepSeek V4 Flash with A0-style prompt | 0.6490 | 0.5245 | 0.3599 | 0.5716 | 0.4845 | 0.0243 |
| compact | Short JSON output, no reasoning field | 0.6441 | 0.5261 | 0.3511 | 0.5698 | 0.4887 | 0.0261 |
| compact_strict | Conservative evidence rule | 0.6125 | 0.5062 | 0.3234 | 0.6247 | 0.4255 | 0.0180 |
| majority vote | Keep labels supported by variants | 0.6507 | 0.5359 | 0.3532 | 0.6350 | 0.4636 | 0.0180 |
| targeted_more_shot_balanced | Targeted examples for difficult labels | 0.6555 | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |
| label-aware ensemble | Majority vote + selected targeted labels | 0.6555 | 0.5657 | 0.3870 | 0.6149 | 0.5238 | 0.0279 |
| small_low_drr | Smaller selected-label set | 0.6555 | 0.5613 | 0.3858 | 0.6222 | 0.5113 | 0.0230 |

**Speaker Notes**

This backup slide summarizes all major full-dev results. The current submission candidate is the label-aware ensemble because it gives the best Task-1b Micro F1 and Macro F1. The small_low_drr run is a backup if direction stability becomes more important.

---

## Backup Slide. Sanity Check Details

### Final Sanity Check Details

Checked output files:

- `pred_task1a.json`
- `pred_task1b.json`

Checked items:

- JSON parse
- Required keys
- Valid `guid`
- Valid `actor_id`
- Valid Level-2 and Level-1 labels
- Valid direction values
- Duplicate predictions
- Missing dev articles
- Task-1a / Task-1b consistency

**Speaker Notes**

This backup slide lists the details of the sanity check. The goal was to make sure that the final candidate is not only high-scoring, but also structurally safe as a submission file.
