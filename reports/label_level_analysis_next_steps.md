# Label-level Analysis: Next Steps

## Comparison

Current comparison:

- Left: DeepSeek V4 Flash baseline full
- Right: majority_vote_full
- Task: Task-1b, so each label means `Level-1 value + direction`

Main frequency-group result:

| Group | Labels | Baseline Avg F1 | Majority Avg F1 | Change |
|---|---:|---:|---:|---:|
| frequent (>=20 gold) | 40 | 0.4796 | 0.4817 | +0.0021 |
| medium (5-19 gold) | 30 | 0.2815 | 0.2866 | +0.0051 |
| infrequent (1-4 gold) | 23 | 0.2537 | 0.2168 | -0.0370 |

Interpretation:

Majority vote improves stability for frequent and medium labels, but hurts infrequent labels. This matches Professor Iwaihara's suggestion that we should not only show total Micro-F1 / Macro-F1, but also inspect individual labels and frequency groups.

## What Improved

Labels improved mostly because false positives were removed or recall was kept while precision improved.

Important improved labels:

| Label | Gold | F1 Change | Main Reason |
|---|---:|---:|---|
| contradictory: Be capable | 5 | +0.407 | recall increased and FP decreased |
| aligned: Have equality | 45 | +0.167 | recall increased, FP decreased |
| contradictory: Have privacy | 10 | +0.143 | recall increased, FP decreased |
| contradictory: Be protecting the environment | 12 | +0.118 | recall increased, FP decreased |
| aligned: Be ambitious | 35 | +0.075 | FP decreased without recall loss |
| aligned: Be honest | 44 | +0.057 | slight recall gain and FP reduction |
| aligned: Be helpful | 97 | +0.045 | FP reduction while keeping high recall |

Takeaway:

Majority vote is effective when a label is over-predicted by only one unstable prompt variant. It keeps labels that are supported by multiple prompts.

## What Degraded

Some labels degraded because majority vote filtered out predictions that appeared only in the high-recall baseline/compact run.

High-priority degraded labels:

| Label | Gold | F1 Change | Why Important |
|---|---:|---:|---|
| aligned: Have good health | 50 | -0.091 | health-related, frequent, recall dropped |
| contradictory: Have good health | 72 | -0.089 | health-related, frequent, recall dropped |
| contradictory: Have a stable society | 139 | -0.067 | security/society-related, frequent |
| aligned: Have social recognition | 29 | -0.065 | frequent, already low baseline F1 |
| aligned: Be capable | 50 | -0.128 | frequent, recall dropped strongly |

Lower-frequency degraded labels:

| Label | Gold | F1 Change | Note |
|---|---:|---:|---|
| contradictory: Have harmony with nature | 3 | -0.500 | one missed case causes large F1 drop |
| contradictory: Be loving | 4 | -0.400 | rare label, majority vote removed all TP |
| aligned: Have a good reputation | 5 | -0.286 | medium-frequency, recall became 0 |
| contradictory: Have a good reputation | 6 | -0.250 | medium-frequency, recall became 0 |
| aligned: Have loyalty towards friends | 11 | -0.133 | medium-frequency, recall became 0 |

Takeaway:

For the next improvement, the most urgent labels are not only the largest F1 drops. We should prioritize labels that are frequent enough and semantically important, especially health / security / society labels.

## Proposed Target Labels

Priority 1: recover important labels lost by majority vote

1. aligned: Have good health
2. contradictory: Have good health
3. contradictory: Have a stable society
4. aligned: Have social recognition
5. aligned: Be capable

Priority 2: low/medium-frequency labels for Macro-F1

1. aligned: Have a good reputation
2. contradictory: Have a good reputation
3. aligned: Have loyalty towards friends
4. contradictory: Be loving
5. contradictory: Have harmony with nature

## More-shot / Reasoning Example Design

The next prompt experiment should not simply add random few-shot examples. It should add label-specific examples for the target labels.

Recommended example format:

```text
Article situation:
  Short summary of the article context.

Actor:
  actor name / role

Correct label:
  direction + Level-1 value

Reasoning:
  Why this actor expresses or violates this value.

Common confusion:
  Which similar label should NOT be selected, and why.
```

For health-related labels:

```text
Target:
  Have good health

Use when:
  The actor's action protects, improves, harms, or threatens physical/mental health.

Aligned:
  public health warning, medical support, safety measures, treatment, prevention

Contradictory:
  disease, injury, exposure to danger, harmful policy/action, neglect of health risks

Avoid confusion:
  Do not use only because the article is generally about a crisis.
  The actor must be connected to health impact or health protection.
```

For stable society / safe country:

```text
Target:
  Have a stable society / Have a safe country

Use when:
  The actor affects public order, national/community safety, crime, violence, conflict, or institutional stability.

Aligned:
  maintaining order, preventing crime, protecting citizens, stabilizing institutions

Contradictory:
  violence, crime, disorder, destabilizing actions, public safety threats

Avoid confusion:
  Stable society is broader social order.
  Safe country is more directly about public/national safety.
```

For social recognition / reputation:

```text
Target:
  Have social recognition / Have a good reputation

Use when:
  The article explicitly discusses public image, honor, status, recognition, criticism, or reputation.

Avoid confusion:
  Do not select only because the actor is mentioned in news.
  There must be evidence about social evaluation or public recognition.
```

## Suggested Next Experiment

Create a new prompt variant:

```text
targeted_more_shot
```

Idea:

- Keep the stable JSON format of compact.
- Add 4-6 short reasoning-style examples focused on:
  - health labels
  - stable society / safe country labels
  - social recognition / reputation labels
- Do not require final output to include reasoning.
- Use reasoning only inside examples and guidelines.

Expected effect:

- Recover FN for important labels.
- Avoid fully returning to the high-FP behavior of the original baseline.
- Improve Macro-F1 or at least reduce damage on infrequent / difficult labels.

## Evaluation Plan

After running `targeted_more_shot`, compare:

1. overall Task-1b Micro-F1 / Macro-F1
2. frequent / medium / infrequent label average F1
3. target-label recall and F1
4. false-positive increase for generic labels such as:
   - Be helpful
   - Be just
   - Be honest
   - Be logical

Success condition:

The new prompt should recover target labels without reintroducing too many generic false positives.

