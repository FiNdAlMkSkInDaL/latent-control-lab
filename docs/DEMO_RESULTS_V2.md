# Demo Results

- Timestamp: `2026-06-08T22:17:06.835228+00:00`
- Model id: `distilgpt2`
- Feature space: `pre_lm_head_last_token`
- Route artifact: `artifacts/example_routes_v2.jsonl`

This scripted run used the zero-generation path:

```text
text -> tokenizer -> frozen LM forward pass -> pre-lm_head hook -> vector -> probe -> gate -> TaskFlowKernel
```

No generated text, JSON/tool-call parsing, regex, or keyword route selection is used.

## Command

```bash
python scripts/run_scripted_demo.py --model-id distilgpt2 --probe artifacts\probe_distilgpt2_v2.joblib --output artifacts\example_routes_v2.jsonl --summary-output docs\DEMO_RESULTS_V2.md --batch-size 6 --max-length 160 --seed 42 --no-4bit --example-set v2 --min-confidence 0.0 --min-margin 0.0
```

## Routes

| Expected | Predicted | Accepted | Confidence | Margin | App status | State after |
|---|---|---:|---:|---:|---|---|
| `CREATE_TASK` | `CREATE_TASK` | True | 0.4378 | 0.1301 | `ok` | backlog=1, active=None, completed=0, archive=0, focus=False |
| `PROMOTE_TASK` | `PROMOTE_TASK` | True | 0.6436 | 0.3182 | `ok` | backlog=0, active=1, completed=0, archive=0, focus=False |
| `COMPLETE_ACTIVE` | `COMPLETE_ACTIVE` | True | 0.8719 | 0.8067 | `ok` | backlog=0, active=None, completed=1, archive=0, focus=False |
| `ARCHIVE_COMPLETED` | `ARCHIVE_COMPLETED` | True | 0.9373 | 0.9104 | `ok` | backlog=0, active=None, completed=0, archive=1, focus=False |
| `TOGGLE_FOCUS_MODE` | `TOGGLE_FOCUS_MODE` | True | 0.9541 | 0.9330 | `ok` | backlog=0, active=None, completed=0, archive=1, focus=True |
| `ABSTAIN` | `ABSTAIN` | False | 0.9997 | 0.9996 | `abstained` | backlog=0, active=None, completed=0, archive=1, focus=True |
