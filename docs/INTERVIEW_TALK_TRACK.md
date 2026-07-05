# Interview Talk Track

## 30-second version

Tiny Latent Control Lab asks whether software can consume a transformer's hidden
state directly instead of waiting for generated tool text. `distilgpt2` reads a
command, a hook captures the pre-lm-head vector, and a linear probe routes that
vector to a typed VectorBot action. The app is a small grid-world so the whole
path is visual, bounded, and auditable.

## What is unusual

Most LLM app demos rely on generated JSON, tool-call text, or command strings.
This one routes from activations:

```text
text -> frozen forward pass -> hidden vector -> probe -> gate -> enum action
```

The strongest claim is the software boundary, not benchmark dominance.

## What to show

1. README + new composite + animated GIFs (`docs/assets/*.gif`).
2. Live demo: `streamlit run streamlit_app.py` or Gradio (side-by-side safety view + trail grid).
3. `neural_native/vectorbot/vector_port.py` — vector in, typed enum out (no text parsing).
4. `scripts/analyze_concept_vectors.py` + generated `vectorbot_concept_directions.json` (shows work with the geometry of the representation).
5. `tests/test_no_generate_guard.py`.

## Results line

On the full CPU run, `distilgpt2` produced `(540, 768)` features with 0.793 test
accuracy, 0.698 macro F1, and 0.964 / 0.964 ABSTAIN precision/recall.

## TF-IDF question

Yes, a text classifier could be strong on a toy synthetic dataset. This repo is
honest about that. The interesting part is demonstrating frozen activation
vectors as a direct control signal for a bounded deterministic app.

## Limitations

Synthetic data, small action space, heuristic gates, and no production-safety
claim. This is portfolio-grade research engineering, not a general agent.
