# Reproducibility

## Environment Used By Codex

- Date: 2026-06-08
- OS shell: Windows PowerShell
- Python: `Python 3.12.13`
- Runtime Python:
  `C:\Users\finla\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

## Dependencies

Base install:

```bash
python -m pip install -e ".[dev]"
```

Real-model local/GPU install:

```bash
python -m pip install -e ".[dev,llm]"
```

The `llm` extra includes `torch`, `transformers`, `accelerate`, optional
`bitsandbytes` on non-Windows, `numpy`, `pandas`, `scikit-learn`, and `joblib`.

## Artifact Policy

Small JSON/CSV/Markdown artifacts are kept for portfolio review. Large feature
arrays and probe bundles are ignored:

- `artifacts/*.npz`
- `artifacts/*.joblib`
- model weights and Hugging Face caches

This keeps the repository reviewable while preserving commands to regenerate the
large files.

## Commands Codex Ran For Milestone 4

```bash
python scripts/generate_dataset_v2.py --output data/intent_dataset_v2.csv --seed 42 --examples-per-class 60 --abstain-examples 180 --strict
python scripts/audit_dataset.py --dataset data/intent_dataset_v2.csv --output-json artifacts/dataset_v2_audit.json --output-csv artifacts/dataset_v2_near_duplicates.csv --similarity-threshold 0.82
python scripts/extract_features.py --dataset data/intent_dataset_v2.csv --model-id distilgpt2 --output artifacts/features_distilgpt2_v2_pre_lm_head.npz --batch-size 8 --max-length 160 --seed 42 --no-4bit
python scripts/train_probe.py --features artifacts/features_distilgpt2_v2_pre_lm_head.npz --output artifacts/probe_distilgpt2_v2.joblib --metrics artifacts/metrics_v2.json --confusion-matrix artifacts/confusion_matrix_v2.csv --thresholds artifacts/thresholds_v2.json --random-state 42
python scripts/evaluate_ood.py --probe artifacts/probe_distilgpt2_v2.joblib --features artifacts/features_distilgpt2_v2_pre_lm_head.npz --output artifacts/ood_metrics_v2.json --seed 42
python scripts/evaluate_hard_eval.py --model-id distilgpt2 --probe artifacts/probe_distilgpt2_v2.joblib --dataset data/hard_eval.csv --output-metrics artifacts/hard_eval_metrics_v2.json --output-predictions artifacts/hard_eval_predictions_v2.csv --output-confusion artifacts/hard_eval_confusion_matrix_v2.csv --features-output artifacts/hard_eval_features_distilgpt2_v2_pre_lm_head.npz --batch-size 8 --max-length 160 --seed 42 --no-4bit
python scripts/train_prompt_augmented_probe.py --model-id distilgpt2 --dataset data/intent_dataset_v2.csv --hard-dataset data/hard_eval.csv --output-probe artifacts/probe_prompt_augmented.joblib --output-json artifacts/prompt_augmented_metrics.json --output-csv artifacts/prompt_augmented_metrics.csv --output-hard-eval artifacts/prompt_augmented_hard_eval.csv --batch-size 8 --max-length 160 --seed 42 --no-4bit
python scripts/calibrate_router.py --probe artifacts/probe_distilgpt2_v2.joblib --features artifacts/features_distilgpt2_v2_pre_lm_head.npz --output-calibration artifacts/router_calibration_v2.json --output-thresholds artifacts/router_thresholds_calibrated_v2.json
python scripts/evaluate_hard_eval.py --model-id distilgpt2 --probe artifacts/probe_distilgpt2_v2.joblib --dataset data/hard_eval.csv --output-metrics artifacts/hard_eval_calibrated_v2.json --output-predictions artifacts/hard_eval_predictions_calibrated_v2.csv --output-confusion artifacts/hard_eval_confusion_matrix_calibrated_v2.csv --thresholds-json artifacts/router_thresholds_calibrated_v2.json --batch-size 8 --max-length 160 --seed 42 --no-4bit
python scripts/evaluate_ood_advanced.py --probe artifacts/probe_distilgpt2_v2.joblib --features artifacts/features_distilgpt2_v2_pre_lm_head.npz --hard-features artifacts/hard_eval_features_distilgpt2_v2_pre_lm_head.npz --output-json artifacts/ood_advanced_metrics_v2.json --output-csv artifacts/ood_advanced_metrics_v2.csv
python scripts/run_baselines.py --dataset data/intent_dataset_v2.csv --hard-dataset data/hard_eval.csv --latent-metrics artifacts/metrics.json --hard-metrics artifacts/hard_eval_metrics.json --latent-v2-metrics artifacts/metrics_v2.json --hard-v2-metrics artifacts/hard_eval_metrics_v2.json --prompt-augmented-metrics artifacts/prompt_augmented_metrics.json --calibrated-metrics artifacts/hard_eval_calibrated_v2.json --output-json artifacts/baseline_metrics_v2.json --output-csv artifacts/baseline_metrics_v2.csv --seed 42
python scripts/run_model_comparison.py --dataset data/intent_dataset_v2.csv --hard-dataset data/hard_eval.csv --models distilgpt2,gpt2,sshleifer/tiny-gpt2 --output-json artifacts/model_comparison.json --output-csv artifacts/model_comparison.csv --sample-size 120 --hard-sample-size 80 --batch-size 8 --max-length 160 --seed 42 --no-4bit
python scripts/run_scripted_demo.py --model-id distilgpt2 --probe artifacts/probe_distilgpt2_v2.joblib --output artifacts/example_routes_v2.jsonl --summary-output docs/DEMO_RESULTS_V2.md --batch-size 6 --max-length 160 --seed 42 --example-set v2 --min-confidence 0 --min-margin 0 --no-4bit
python -m ruff check .
python -m pytest --basetemp=.pytest_tmp
```

## Reproducing Gemma

Use `docs/COLAB_GEMMA.md` or `notebooks/05_gemma_colab_real_model.ipynb` in a
GPU Colab runtime. Set `HF_TOKEN`, run the V2 extraction/training/calibration
commands, and only claim Gemma results after artifacts are produced.
