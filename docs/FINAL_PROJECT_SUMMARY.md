# Final Project Summary

Tiny Latent Control Lab demonstrates zero-generation latent action routing with
a visual toy app called VectorBot.

## Core idea

A frozen `distilgpt2` model processes natural-language commands. A hook captures
the pre-lm-head hidden-state vector before text generation. A logistic-regression
probe maps that vector to a bounded enum action, and a deterministic grid-world
kernel updates state.

```text
command -> frozen model forward pass -> hidden vector -> linear probe
-> gate -> VectorBot action -> state transition
```

## Full run

- Dataset size: 540
- Feature shape: `(540, 768)`
- Test accuracy: 0.793
- Macro F1: 0.698
- ABSTAIN precision / recall: 0.964 / 0.964
- Scripted demo: 6 executable actions accepted, 4 ABSTAIN inputs rejected

## Delivered components

- `neural_native/vectorbot/`: state, kernel, ASCII renderer, router adapter, and
  vector-facing port.
- `scripts/generate_vectorbot_dataset.py`: deterministic dataset generation and
  audit output.
- `scripts/extract_vectorbot_features.py`: CPU-friendly forward-pass extraction.
- `scripts/train_vectorbot_probe.py`: scikit-learn probe training and metrics.
- `scripts/run_vectorbot_demo.py`: live, scripted, replay, fake-vector modes, and
  Markdown transcript output.
- `scripts/build_vectorbot_visuals.py`: grid, confidence, latent-space,
  pipeline, and composite PNG assets.
- `streamlit_app.py` and `docs/vectorbot_demo.html`: optional dashboard and
  static replay surfaces.

## Limitations

This is a bounded visual demo, not a production agent. It does not prove broad
language robustness, production OOD safety, general tool use, or superiority
over all text classifiers.
