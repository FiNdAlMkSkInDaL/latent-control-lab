# Prompt Augmentation

Milestone 3 showed that a probe trained on one router prompt did not transfer to
minimal, instruction, or noisy prompt wrappers. Milestone 4 trained one probe on
the union of five prompt-rendered versions of the v2 train split:

1. Default router prompt
2. Minimal: `{text}`
3. Instruction: `Classify the controller intent represented by this request: {text}`
4. Controller: `Request from user: {text}\nInternal controller representation:`
5. Terse: `Intent: {text}`

Artifacts:

- `artifacts/prompt_augmented_metrics.json`
- `artifacts/prompt_augmented_metrics.csv`
- `artifacts/prompt_augmented_hard_eval.csv`

## Results

| Prompt | V2 test accuracy | V2 test macro F1 | Hard accuracy | Hard macro F1 | Hard ABSTAIN recall |
|---|---:|---:|---:|---:|---:|
| Default | 1.000 | 1.000 | 0.651 | 0.516 | 0.847 |
| Minimal | 0.986 | 0.988 | 0.669 | 0.524 | 0.867 |
| Instruction | 0.986 | 0.988 | 0.651 | 0.494 | 0.873 |
| Controller | 1.000 | 1.000 | 0.658 | 0.536 | 0.827 |
| Terse | 0.986 | 0.988 | 0.676 | 0.553 | 0.860 |

## Did It Improve Robustness?

Yes. Unlike the Milestone 3 single-prompt probe, the prompt-augmented probe
keeps high v2 test accuracy across all prompt templates and no longer collapses
alternate templates into all-ABSTAIN behavior.

## Did It Hurt Default Prompt Performance?

No on v2 test: default prompt accuracy and macro-F1 remain `1.000`. On hard eval,
the prompt-augmented default prompt reaches accuracy `0.651` and macro-F1
`0.516`, above the default V2 router macro-F1 of `0.349`.

## Which Prompts Remain Weak?

Instruction is the weakest hard-eval prompt in this run with macro-F1 `0.494`.
All prompts still struggle with executable hard paraphrases compared with
ABSTAIN rejection.

## Production Recommendation

For this portfolio prototype, keep one fixed production prompt for
reproducibility, but prefer a prompt-augmented probe if prompt wrappers may vary.
The prompt template should remain part of the model card and artifact metadata.
