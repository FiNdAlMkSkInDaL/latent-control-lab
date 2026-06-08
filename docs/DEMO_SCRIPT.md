# Demo Script

## 30-Second Demo

"This project routes natural language into software actions without generated
tool calls. I run a frozen Hugging Face causal LM, capture the hidden state just
before `lm_head`, classify that vector with a linear probe, and execute only a
typed enum action in a sandboxed task controller. The demo shows five accepted
actions and one negated command that correctly routes to `ABSTAIN`."

## 2-Minute Demo

1. Open `README.md` and point to the architecture diagram.
2. Explain the invariant: no `model.generate()`, no generated JSON, no regex or
   keyword parser deciding the action.
3. Open `docs/DEMO_RESULTS_V2.md` or `artifacts/example_routes_v2.jsonl`.
4. Walk through the six examples:
   - create task
   - promote task
   - complete active task
   - archive completed
   - toggle focus
   - negated complete request -> `ABSTAIN`
5. Explain that the app action is selected from a hidden vector, not from text.
6. Close with the honest result: V2 cleans up data leakage and improves
   robustness, but TF-IDF remains competitive and hard executable paraphrases are
   still a weakness.

## Demo Commands

Fast public demo command:

```bash
python scripts/run_scripted_demo.py \
  --model-id distilgpt2 \
  --probe artifacts/probe_distilgpt2_v2.joblib \
  --output artifacts/example_routes_v2.jsonl \
  --summary-output docs/DEMO_RESULTS_V2.md \
  --batch-size 6 \
  --max-length 160 \
  --seed 42 \
  --example-set v2 \
  --min-confidence 0 \
  --min-margin 0 \
  --no-4bit
```

Rebuild the V2 route from scratch:

```bash
python scripts/generate_dataset_v2.py --output data/intent_dataset_v2.csv --strict
python scripts/extract_features.py --dataset data/intent_dataset_v2.csv --model-id distilgpt2 --output artifacts/features_distilgpt2_v2_pre_lm_head.npz --no-4bit
python scripts/train_probe.py --features artifacts/features_distilgpt2_v2_pre_lm_head.npz --output artifacts/probe_distilgpt2_v2.joblib --metrics artifacts/metrics_v2.json --confusion-matrix artifacts/confusion_matrix_v2.csv --thresholds artifacts/thresholds_v2.json
```

## What To Point Out

- The app action enum is small and safe.
- The raw text is stored as context only; it does not decide the route.
- The route passes through `VectorActionPort`.
- The LLM is frozen; only the probe is trained.
- The `ABSTAIN` route is preserved and measured.

## Explaining Weak Metrics Honestly

Say this plainly:

"The project is not claiming to beat text classifiers. TF-IDF is competitive on
this small benchmark. The interesting result is architectural: a real
hidden-state route can control a typed app without generated commands. The
evaluation also found weaknesses, especially executable hard paraphrases and OOD
calibration, and the docs preserve those results instead of hiding them."
