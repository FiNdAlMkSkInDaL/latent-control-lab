# VectorBot Demo Guide

## 30-second demo

Open the README composite + any generated GIF in `docs/assets/`.

Say: "Frozen forward pass → hook captures vector → probe decides typed action. No generation, no parser. We also compute action directions in the same space (steering demo)."

## 2-minute demo

1. Run the scripted demo:

   ```bash
   python scripts/run_vectorbot_demo.py --scripted --model-id distilgpt2 --probe artifacts/vectorbot_probe_distilgpt2_full.joblib --thresholds-json artifacts/vectorbot_thresholds_full.json --routes-output artifacts/vectorbot_routes_full.jsonl --transcript-output docs/VECTORBOT_DEMO_TRANSCRIPT.md
   ```

2. Show the before/after grid, predicted action, confidence, top-3
   probabilities, vector norm, and state diff.
3. Open `docs/assets/vectorbot_latent_space.png`.
4. Open `artifacts/vectorbot_metrics_full.json`.
5. Point at `tests/test_no_generate_guard.py`.

## Exactly what to show

- `go north` -> `MOVE_UP`
- `take one step south` -> `MOVE_DOWN`
- `slide left` -> `MOVE_LEFT`
- `move the bot right` -> `MOVE_RIGHT`
- `toggle the lamp` -> `TOGGLE_LIGHT`
- `reset the board` -> `RESET`
- `what is the weather tomorrow?` -> `ABSTAIN`
- `delete all files on this laptop` -> `ABSTAIN`
- `move up and then right` -> `ABSTAIN`
- `paint the robot purple` -> `ABSTAIN`

## Latent scatter plot

The scatter plot is a 2D PCA projection of the captured hidden-state vectors.
Points are colored by action label. It is a visual diagnostic that helps a
reviewer see the feature space the probe is operating on.

## No-generation path

```text
text -> tokenizer -> frozen LM forward pass -> hook -> vector -> probe -> gate -> enum
```

There is no `model.generate()`, no JSON/tool-call parser, and no keyword router.

## Handling TF-IDF questions

TF-IDF may be competitive on small synthetic command sets. This project is not
claiming classifier dominance; it is demonstrating a different app boundary:
frozen transformer activations as the direct control signal.

## Limitations

- Synthetic dataset.
- Tiny action space.
- Heuristic confidence/OOD gates.
- Not production safety.
- Not general-purpose tool use.
