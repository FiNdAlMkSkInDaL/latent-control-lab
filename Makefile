PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-llm:
	$(PYTHON) -m pip install -e ".[dev,llm,viz]"

dataset:
	$(PYTHON) scripts/generate_dataset.py

vectorbot-dataset:
	$(PYTHON) scripts/generate_vectorbot_dataset.py --output data/vectorbot_intents.csv --strict

vectorbot-train:
	$(PYTHON) scripts/extract_vectorbot_features.py --dataset data/vectorbot_intents.csv --model-id distilgpt2 --output artifacts/vectorbot_features_distilgpt2.npz
	$(PYTHON) scripts/train_vectorbot_probe.py --features artifacts/vectorbot_features_distilgpt2.npz --output artifacts/vectorbot_probe_distilgpt2.joblib

vectorbot-demo:
	$(PYTHON) scripts/run_vectorbot_demo.py --scripted --model-id distilgpt2 --probe artifacts/vectorbot_probe_distilgpt2.joblib --thresholds-json artifacts/vectorbot_thresholds.json

vectorbot-visuals:
	$(PYTHON) scripts/build_vectorbot_visuals.py

vectorbot-quickstart:
	$(PYTHON) scripts/vectorbot_quickstart.py --model-id distilgpt2 --fast --seed 42

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
