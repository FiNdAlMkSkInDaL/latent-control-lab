# GitHub Release Notes

## Tiny Latent Control Lab

This release pivots the repository to a visual zero-generation demo:

```text
natural language -> frozen distilgpt2 forward pass -> hidden vector
-> trained probe -> confidence/OOD gate -> VectorBot enum action
-> deterministic grid-world state transition
```

## Highlights

- Added VectorBot, a 5x5 grid-world controller with movement, light toggle,
  reset, and `ABSTAIN`.
- Added deterministic dataset generation with a 540-row full dataset and audit
  artifact showing zero normalized duplicates and zero split overlap.
- Added CPU-friendly feature extraction, probe training, scripted demo,
  transcript generation, and visual asset generation.
- Added README-ready images, including a composite demo visual.
- Added optional Streamlit dashboard and static HTML replay.

## Full distilgpt2 result

- Feature shape: `(540, 768)`
- Test accuracy: 0.793
- Macro F1: 0.698
- ABSTAIN precision / recall: 0.964 / 0.964
- Scripted demo: six executable routes accepted and four ABSTAIN inputs rejected

## Invariants

- No `model.generate()` in core routing.
- No generated JSON/tool-call parsing.
- No regex or keyword parser deciding VectorBot actions.
- No arbitrary filesystem, shell, network, or OS-control action in the toy app.

## Limitations

This is a bounded visual demo, not production safety, general tool use, or proof
of superiority over all text classifiers.
