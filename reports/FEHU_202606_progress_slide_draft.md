# FEHU Task-1 Progress Report - Slide Draft

Presenter: Liu Dongyang  
Meeting: Group Seminar, June 2026  
Topic: Label-level analysis and label-aware ensemble for FEHU Task-1

---

## Slide 1. Title

### Label-level Analysis for FEHU Task-1

- Liu Dongyang
- Data Engineering Lab, IPS, Waseda University
- Progress report for NTCIR-19 FEHU Task-1

**Layout Suggestion**

Use a simple title slide. Add one short subtitle:

> From overall prompt tuning to label-aware ensemble

**Speaker Notes**

Good afternoon everyone. Today I will report my recent progress on FEHU Task-1. In my previous presentation, I mainly reported the reproduction of the A0 baseline, DeepSeek prompt-based experiments, and majority-vote ensemble. After that, following Professor Iwaihara's advice, I focused more on label-level analysis and tried to improve difficult labels more systematically.

---

## Slide 2. Outline

### Outline

- Previous status: prompt-based baseline and majority vote
- Professor's suggestion: evaluate each label separately
- Label-level analysis: frequent and infrequent labels
- Targeted more-shot prompting
- Label-aware ensemble
- Current result and next plan

**Layout Suggestion**

Use large bullets, similar to your previous outline slide. Keep each line short.

**Speaker Notes**

The main flow of this presentation is as follows. First, I briefly review the previous status. Then I explain how I followed Professor Iwaihara's suggestion and analyzed each label separately. Based on this analysis, I tested targeted more-shot prompts and finally designed a label-aware ensemble strategy.

---

## Slide 3. Previous Status

### Previous Status: Strong Prompt-only Baseline

#### Main completed work before the last presentation

- Reproduced the A0-style prompt pipeline
- Built local workflow:
  - data loading
  - LLM API call
  - prediction file generation
  - official evaluation
- Tested DeepSeek V4 Flash as a low-cost prompt-only baseline
- Tried prompt variants and majority-vote ensemble

#### Best result at that time

| Method | Task-1b Micro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|
| DeepSeek baseline | 0.5245 | 0.5716 | 0.4845 | 0.0243 |
| Majority vote | 0.5359 | 0.6350 | 0.4636 | 0.0180 |

**Layout Suggestion**

Left side: short bullet list.  
Right side: small comparison table.  
Bottom takeaway:

> Majority vote was stable, but recall for difficult labels was still limited.

**Speaker Notes**

Before the last presentation, I had already reproduced the A0-style pipeline and built a local experimental workflow. I also tested DeepSeek V4 Flash and several prompt variants. The best result at that time was majority vote. It improved precision and reduced the direction reverse rate, but it also became more conservative and lost some recall.

---

## Slide 4. Motivation

### Motivation: Overall Score Is Not Enough

#### Professor's suggestion

- Do not only report Micro F1 and Macro F1
- Check the performance of each label separately
- Compare frequent and infrequent labels
- Identify difficult labels
- Use more-shot / generated samples / reasoning examples for difficult labels

#### My research question

> Which labels improve, which labels degrade, and can this information guide better prompt or ensemble design?

**Layout Suggestion**

Make this page text-based but strong.  
Use two boxes:

1. `Professor's suggestion`
2. `Research question`

**Speaker Notes**

Professor Iwaihara suggested that I should not only show total Micro F1 and Macro F1. For this task, each value label may behave very differently. Some labels are frequent, while some are very rare. Therefore, I started to analyze the performance of each label separately, and tried to use this information to design better prompts and ensembles.

---

## Slide 5. Label-level Analysis Method

### Label-level Analysis Method

#### Analysis unit

- Task-1b label = `direction + Level-1 value`
- Example:
  - `aligned: Have good health`
  - `contradictory: Have a stable society`

#### Procedure

1. Compare prediction files with gold labels
2. Calculate TP / FP / FN for each label
3. Compute label-level Precision / Recall / F1
4. Divide labels into frequency groups
5. Visualize top improved and degraded labels

**Layout Suggestion**

Use a horizontal workflow:

`Prediction files -> Gold labels -> Label-level metrics -> Frequency groups -> Bar charts`

Add a small note:

> This makes prompt behavior observable at the label level.

**Speaker Notes**

For Task-1b, I treated each direction and Level-1 value pair as one label. Then I compared prediction files with the gold labels and calculated TP, FP, FN, precision, recall, and F1 for each label. I also divided labels into frequent, medium, and infrequent groups, and visualized the top improved and degraded labels with bar charts.

---

## Slide 6. First Finding: Majority Vote Is Conservative

### Majority Vote: Stable but Loses Infrequent Labels

Compared with DeepSeek baseline:

| Label group | Baseline Avg F1 | Majority Avg F1 | Change |
|---|---:|---:|---:|
| Frequent labels | 0.4796 | 0.4817 | +0.0021 |
| Medium labels | 0.2815 | 0.2866 | +0.0051 |
| Infrequent labels | 0.2537 | 0.2168 | -0.0370 |

#### Interpretation

- Majority vote removes unstable false positives
- But it also removes true labels predicted by only one high-recall prompt
- Infrequent labels are especially vulnerable

**Layout Suggestion**

Use a table on the left and a simple small bar chart or arrow diagram on the right:

`Precision up -> Recall risk`

**Speaker Notes**

The first finding was that majority vote is useful but conservative. It slightly improved frequent and medium labels, but hurt infrequent labels. This means that majority vote can remove unstable false positives, but it may also remove true labels that are predicted by only one high-recall prompt.

---

## Slide 7. Error Pattern

### Error Pattern: What Labels Need Help?

#### Labels often lost by conservative ensemble

- `Have good health`
- `Have a stable society`
- `Have social recognition`
- `Be capable`
- `Have a good reputation`

#### Important observation

- Some labels are semantically important but difficult
- Health / society / safety labels are often missed
- Generic moral labels may be over-predicted

#### Design implication

> Recover important false negatives without reintroducing too many generic false positives.

**Layout Suggestion**

Use two columns:

Left: `Missed important labels`  
Right: `Risky over-predicted labels`

If you have room, put a small screenshot or simplified bar chart of top degraded labels.

**Speaker Notes**

From the label-level error analysis, I found that some important labels were often lost, especially health-related and society-related labels. At the same time, the model may over-predict generic moral values. Therefore, the next goal was to recover important false negatives without bringing back too many false positives.

---

## Slide 8. Targeted More-shot Prompting

### Targeted More-shot Prompting

#### Motivation

- Professor suggested more-shot or reasoning examples for difficult labels
- Random examples may not help the target labels
- I focused examples on labels found in error analysis

#### Target label groups

- Health:
  - `Have good health`
- Society / security:
  - `Have a stable society`
  - `Have a safe country`
- Recognition / reputation:
  - `Have social recognition`
  - `Have a good reputation`
- Capability:
  - `Be capable`

#### Prompt variants

| Variant | Purpose |
|---|---|
| targeted_more_shot | recover target labels |
| targeted_more_shot_strict | reduce false positives |
| targeted_more_shot_balanced | balance recall and precision |

**Layout Suggestion**

Top: motivation sentence.  
Middle: target labels grouped by meaning.  
Bottom: variant table.

**Speaker Notes**

Based on this analysis, I designed targeted more-shot prompt variants. Instead of adding random examples, I focused on difficult labels such as health, stable society, safe country, social recognition, reputation, and capability. I tested three versions: a recall-oriented version, a stricter version, and a balanced version.

---

## Slide 9. Targeted More-shot Result

### Targeted More-shot Balanced: Higher Recall

Full dev comparison:

| Method | Task-1b Micro F1 | Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|
| Majority vote | 0.5359 | 0.3532 | 0.6350 | 0.4636 | 0.0180 |
| Targeted balanced | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |

#### Main effect

- Recall increased: `0.4636 -> 0.5067`
- Macro F1 increased: `0.3532 -> 0.3679`
- But precision and DRR became worse

**Layout Suggestion**

Use three number cards:

- Recall `+0.0431`
- Macro F1 `+0.0146`
- DRR `+0.0122`

Then put the table below.

**Speaker Notes**

The targeted balanced prompt improved recall and macro F1 compared with majority vote. This suggests that targeted examples helped recover some difficult labels. However, precision decreased and the direction reverse rate increased. So this prompt was useful, but not stable enough by itself.

---

## Slide 10. Complementary Behavior

### Majority Vote and Targeted Prompt Are Complementary

#### Majority vote

- Higher precision
- Lower DRR
- Stable on common labels
- But misses difficult / infrequent labels

#### Targeted more-shot balanced

- Higher recall
- Better macro F1
- Recovers health / society labels
- But introduces more false positives

#### Key idea

> Use majority vote as the base, and add targeted predictions only for labels where targeted prompting clearly works better.

**Layout Suggestion**

Three-panel layout:

Left: majority vote strengths  
Middle: targeted balanced strengths  
Right: label-aware rule

**Speaker Notes**

The two methods showed complementary behavior. Majority vote was more precise and stable, while targeted balanced recovered more difficult labels. Therefore, I did not simply choose one of them. Instead, I designed a label-aware ensemble strategy.

---

## Slide 11. Label-aware Ensemble Design

### Label-aware Ensemble

#### Rule

Start from `majority vote`, then add predictions from `targeted_more_shot_balanced` only for selected labels.

#### Selected labels

- `Have a stable society`
- `Have good health`
- `Be capable`
- `Have harmony with nature`
- `Be creative`
- `Have equality`

#### Why these labels?

- Targeted prompt clearly improved them
- Many of them were difficult or infrequent
- They were important for recall and macro F1

**Layout Suggestion**

Use a simple formula:

`Label-aware ensemble = Majority vote base + selected targeted labels`

Then show selected labels as colored tags.

**Speaker Notes**

The label-aware ensemble uses majority vote as the base prediction. Then it selectively adds predictions from the targeted balanced prompt only for labels where the targeted prompt showed clear improvement. This makes the ensemble different from a blind union.

---

## Slide 12. Main Result

### Main Result: Best Current Task-1b Score

Full dev comparison:

| Method | Task-1a Micro F1 | Task-1b Micro F1 | Task-1b Macro F1 | Precision | Recall | DRR |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek baseline | 0.6490 | 0.5245 | 0.3599 | 0.5716 | 0.4845 | 0.0243 |
| Majority vote | 0.6507 | 0.5359 | 0.3532 | 0.6350 | 0.4636 | 0.0180 |
| Targeted balanced | 0.6555 | 0.5387 | 0.3679 | 0.5750 | 0.5067 | 0.0302 |
| Label-aware ensemble | **0.6555** | **0.5657** | **0.3870** | 0.6149 | **0.5238** | 0.0279 |

#### Main gain over majority vote

- Task-1b Micro F1: `+0.0298`
- Task-1b Macro F1: `+0.0337`
- Recall: `+0.0603`

**Layout Suggestion**

This should be one of the strongest slides.  
Use a table plus three large number callouts:

- `T1b Micro F1 0.5657`
- `T1b Macro F1 0.3870`
- `Recall 0.5238`

**Speaker Notes**

This is the current best result in my experiments. Compared with majority vote, the label-aware ensemble improved Task-1b Micro F1 from 0.5359 to 0.5657, and Macro F1 from 0.3532 to 0.3870. Recall also improved clearly while precision stayed higher than the targeted balanced prompt.

---

## Slide 13. Why It Works

### Why Label-aware Ensemble Works

Compared with majority vote:

| Label group | Majority F1 | Label-aware F1 | Change |
|---|---:|---:|---:|
| Frequent labels | 0.4817 | 0.5109 | +0.0292 |
| Medium labels | 0.2866 | 0.2866 | +0.0000 |
| Infrequent labels | 0.2168 | 0.3025 | +0.0857 |

#### Improved labels

| Label | Majority F1 | Label-aware F1 |
|---|---:|---:|
| contradictory: Have a stable society | 0.2545 | 0.5611 |
| aligned: Have good health | 0.3188 | 0.5496 |
| aligned: Be capable | 0.5070 | 0.6549 |
| contradictory: Have harmony with nature | 0.0000 | 0.8000 |

#### Takeaway

> The improvement comes from label-level selection, not blind combination.

**Layout Suggestion**

Use the frequency group table on the left.  
Use improved-label examples on the right.  
Bottom: one strong takeaway sentence.

**Speaker Notes**

The reason this ensemble works is that it improves the labels that majority vote missed, especially infrequent labels. Compared with majority vote, the average F1 of infrequent labels increased from 0.2168 to 0.3025. This supports the idea that label-level analysis can guide a more effective ensemble.

---

## Slide 14. Current Understanding

### Current Understanding

#### What I learned

- Overall F1 alone hides important label behavior
- Conservative ensemble improves precision but can hurt rare labels
- Targeted more-shot examples can recover difficult labels
- Label-aware ensemble can combine precision and recall more effectively

#### Remaining issues

- Selected label list may still be too broad
- DRR is higher than majority vote
- Some labels still do not improve:
  - `Have social recognition`
  - `Have a good reputation`
  - some safety-related directions

**Layout Suggestion**

Use two columns:

Left: `Findings`  
Right: `Remaining issues`

**Speaker Notes**

From these experiments, I learned that overall F1 is not enough to understand model behavior. Label-level analysis helped me find complementary patterns between prompt variants. However, the current label-aware ensemble is not perfect. The selected label list may still need refinement, and the direction reverse rate is higher than majority vote.

---

## Slide 15. Next Plan

### Next Plan

#### 1. Refine label-aware ensemble

- Test smaller selected-label sets
- Try to reduce DRR
- Check label-wise FP increase

#### 2. Prepare candidate runs

- High-recall run: targeted balanced
- Precision-oriented run: majority vote
- Best current run: label-aware ensemble

#### 3. Discuss team strategy

- Compare with Peiyang's hybrid approach
- Decide which runs to submit
- Align prompt-only and hybrid tracks

**Layout Suggestion**

Use three cards:

1. `Refine`
2. `Prepare`
3. `Discuss`

Bottom goal:

> Move from prompt optimization to explainable submission strategy.

**Speaker Notes**

For the next step, I want to refine the label-aware ensemble by testing smaller selected-label sets and checking whether DRR can be reduced. I also plan to prepare candidate runs for submission, including a high-recall run, a precision-oriented run, and the current label-aware ensemble. Finally, I will discuss with Peiyang how to combine this prompt-only track with the hybrid approach.

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

**Speaker Notes**

This backup slide shows the full experiment matrix. The important point is that the label-aware ensemble achieves the best Task-1b Micro F1 and Macro F1, while keeping precision higher than the targeted balanced prompt.
