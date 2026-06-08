PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-llm:
	$(PYTHON) -m pip install -e ".[dev,llm,viz]"

dataset:
	$(PYTHON) scripts/generate_dataset.py

extract-tiny:
	$(PYTHON) scripts/extract_features.py --model-id sshleifer/tiny-gpt2 --batch-size 4 --output artifacts/features_tiny-gpt2_pre_lm_head.npz

extract-distilgpt2:
	$(PYTHON) scripts/extract_features.py --model-id distilgpt2 --batch-size 4 --no-4bit --output artifacts/features_distilgpt2_pre_lm_head.npz

train:
	$(PYTHON) scripts/train_probe.py --features artifacts/features_distilgpt2_pre_lm_head.npz --output artifacts/probe.joblib

eval:
	$(PYTHON) scripts/evaluate_ood.py --probe artifacts/probe.joblib --features artifacts/features_distilgpt2_pre_lm_head.npz

examples:
	$(PYTHON) scripts/run_scripted_demo.py --model-id distilgpt2 --probe artifacts/probe.joblib --output artifacts/example_routes.jsonl --summary-output docs/DEMO_RESULTS.md --no-4bit

demo:
	$(PYTHON) -m neural_native.cli run --probe-path artifacts/probe.joblib

test:
	$(PYTHON) -m pytest --basetemp=.pytest_tmp -p no:cacheprovider

lint:
	$(PYTHON) -m ruff check neural_native tests scripts
