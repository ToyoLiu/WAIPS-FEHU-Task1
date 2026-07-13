# FEHU Task-1 Prompt Pipeline

This repository contains Liu Dongyang's prompt-based pipeline and post-processing
experiments for NTCIR-19 FEHU Task-1.

The main focus is:

- reproducing an A0-style prompt baseline,
- running DeepSeek V4 Flash prompt-only experiments,
- analyzing Task-1b labels at the label level,
- improving difficult labels with targeted more-shot prompts,
- combining prompt variants with majority vote and label-aware ensemble.

## What to Read First

For understanding the current best method, read these files in order:

1. `run_pipeline.py`
   - Main Task-1 prompt pipeline.
   - Supports OpenAI, Gemini, and DeepSeek providers.
   - Includes prompt variants such as `baseline`, `compact`,
     `compact_strict`, `targeted_more_shot`, and
     `targeted_more_shot_balanced`.

2. `label_aware_ensemble.py`
   - Creates the current best label-aware ensemble.
   - Starts from majority vote and selectively adds targeted prompt
     predictions for selected labels.

3. `label_level_compare.py`
   - Computes label-level Task-1b performance.
   - Used to identify improved / degraded labels.

4. `final_sanity_check.py`
   - Checks whether a candidate run is structurally safe before submission.

## Current Best Run

The current best dev-set candidate is:

```text
dev_deepseek_ensemble_label_aware_targeted_balanced
```

Full dev result:

| Metric | Score |
|---|---:|
| Task-1a Micro F1 | 0.6555 |
| Task-1a Macro F1 | 0.5044 |
| Task-1b Micro F1 | 0.5657 |
| Task-1b Macro F1 | 0.3870 |
| Task-1b Precision | 0.6149 |
| Task-1b Recall | 0.5238 |
| DRR | 0.0279 |

Lower-DRR backup candidate:

```text
dev_deepseek_ensemble_label_aware_small_low_drr
```

## Folder Structure

```text
.
├── run_pipeline.py
├── label_aware_ensemble.py
├── label_level_compare.py
├── final_sanity_check.py
├── reports/
│   ├── label_aware_ensemble_report.md
│   ├── why_choose_label_aware_ensemble.md
│   └── final_sanity_check_report.md
├── figures/
│   ├── task1b_top_improved.png
│   └── task1b_top_degraded.png
└── scripts/
    └── backup copies of the same scripts
```

## Data and Official Repository

This repository does not include the official FEHU dataset or API cache.

To run the pipeline, put the official project folder next to these scripts:

```text
ntcir19_fehu-master/
  ntcir19_fehu-master/
    dataset/
    evaluation.py
```

The scripts automatically search for this folder structure.

## Environment Variables

Set API keys outside the code.

PowerShell examples:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
$env:GEMINI_API_KEY="your_key_here"
$env:OPENAI_API_KEY="your_key_here"
```

Do not commit API keys, `.env` files, cache files, or generated outputs.

## Example Commands

Run a small DeepSeek dev test:

```powershell
python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --limit 20
```

Run the targeted more-shot balanced prompt:

```powershell
python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant targeted_more_shot_balanced
```

Create the label-aware ensemble:

```powershell
python label_aware_ensemble.py
```

Run final sanity check:

```powershell
python final_sanity_check.py --run dev_deepseek_ensemble_label_aware_targeted_balanced
```

## Notes for Collaboration

This code is intended for comparison with Huang Peiyang's hybrid and data
augmentation approaches. The most useful comparison point is label-level
behavior, especially difficult or infrequent Task-1b labels.
