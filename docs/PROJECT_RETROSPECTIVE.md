# Project Retrospective

## What Worked

- The zero-generation route works with a real Hugging Face model.
- The action boundary stayed small, typed, and auditable.
- Hook tests and no-generate guard tests made the invariant concrete.
- Saved JSON/CSV artifacts made evaluation easy to inspect.

## What Failed

- V1 synthetic test accuracy was misleading because the dataset leaked template
  families and near-duplicates across splits.
- The first probe was prompt-sensitive.
- The hard eval revealed weak rejection for negation, compounds, and near misses.
- Some OOD thresholds collapsed executable acceptance.

## What Changed After Evaluation

- Added V2 data with zero normalized duplicates and zero template-family split
  overlap.
- Added hard negatives to training/validation.
- Added prompt-augmented training across five prompt templates.
- Added validation-selected calibration and advanced OOD comparisons.
- Added TF-IDF baselines and reported their competitiveness.

## Engineering Lessons

- Invariants need tests, not just README claims.
- Evaluation scripts should produce small, reviewable artifacts.
- Large arrays and probes should stay ignored and reproducible.
- A bounded state machine makes safety analysis tractable.

## Research Lessons

- Perfect synthetic accuracy can be a smell.
- Prompt templates are part of the feature space.
- OOD AUROC and thresholded router behavior can tell different stories.
- Text baselines are necessary, especially for small intent datasets.

## Next Research Directions

1. Run the V2 path with Gemma in Colab.
2. Collect larger independent hard-negative and paraphrase sets.
3. Compare linear probes with small nonlinear probes while keeping the LLM frozen.
4. Validate Mahalanobis OOD on data not used for generator design.
5. Explore activation patching or counterfactual vector swaps.
