# CV Bullets

- Built Tiny Latent Control Lab, a zero-generation latent-control demo where
  `distilgpt2` hidden states route natural-language commands to a bounded
  VectorBot grid-world app through a trained probe.
- Produced a CPU-friendly full run with 540 hidden-state vectors, 0.793 test
  accuracy, 0.698 macro F1, and 0.964 / 0.964 ABSTAIN precision/recall.
- Implemented deterministic app state transitions, OOD abstention, route logs,
  transcript generation, static dashboard replay, and GitHub-ready visual assets.
- Added tests guarding no `model.generate()`, no generated command parsing, and
  no keyword/regex action router in the core path.
