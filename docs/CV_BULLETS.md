# CV Bullets

## Software Engineering

- Built a typed Python MVP that routes frozen Hugging Face hidden states into a
  sandboxed task-state machine via `VectorActionPort -> router.predict(z) ->
  TaskFlowKernel.execute(...)`, with CI-ready lint/test coverage.

## AI Research Engineering

- Implemented a zero-generation latent action router using `distilgpt2`
  pre-`lm_head` activations (`X=(480, 768)`), scikit-learn linear probes, hard
  negative evaluation, prompt augmentation, OOD calibration, and reproducible
  JSON/CSV metrics.

## Systems Architecture

- Designed a bounded action-routing architecture that avoids generated JSON,
  regex command parsing, shell execution, and arbitrary APIs, preserving a small
  audited action enum plus `ABSTAIN`.

## Mechanistic Interpretability Angle

- Used PyTorch forward hooks to compare activation sites and measure where
  action intent becomes linearly separable, including sampled layer-sweep
  evidence across early, middle, final, and pre-`lm_head` representations.

## Evaluation And Robustness

- Diagnosed synthetic leakage, rebuilt a V2 dataset with zero normalized
  duplicates and zero template-family split overlap, improved hard-eval accuracy
  from `0.575` to `0.647` with calibrated V2 routing, and reported TF-IDF
  competitiveness honestly.
