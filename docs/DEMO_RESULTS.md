# Demo Results

- Timestamp: `2026-06-08T21:30:00.437392+00:00`
- Model id: `distilgpt2`
- Feature space: `pre_lm_head_last_token`
- Route artifact: `artifacts/example_routes.jsonl`

This scripted run used the zero-generation path:

```text
text -> tokenizer -> frozen LM forward pass -> pre-lm_head hook -> vector -> probe -> gate -> TaskFlowKernel
```

No generated text, JSON/tool-call parsing, regex, or keyword route selection is used.

## Command

```bash
python scripts/run_scripted_demo.py --model-id distilgpt2 --probe artifacts\probe.joblib --output artifacts\example_routes.jsonl --summary-output docs\DEMO_RESULTS.md --batch-size 6 --max-length 160 --seed 42 --no-4bit
```

## Routes

| Expected | Predicted | Accepted | Confidence | Margin | App status | State after |
|---|---|---:|---:|---:|---|---|
| `CREATE_TASK` | `CREATE_TASK` | True | 0.9947 | 0.9907 | `ok` | backlog=1, active=None, completed=0, archive=0, focus=False |
| `PROMOTE_TASK` | `PROMOTE_TASK` | True | 0.8423 | 0.7238 | `ok` | backlog=0, active=1, completed=0, archive=0, focus=False |
| `COMPLETE_ACTIVE` | `COMPLETE_ACTIVE` | True | 0.9929 | 0.9873 | `ok` | backlog=0, active=None, completed=1, archive=0, focus=False |
| `ARCHIVE_COMPLETED` | `ARCHIVE_COMPLETED` | True | 0.9592 | 0.9324 | `ok` | backlog=0, active=None, completed=0, archive=1, focus=False |
| `TOGGLE_FOCUS_MODE` | `TOGGLE_FOCUS_MODE` | True | 0.9998 | 0.9997 | `ok` | backlog=0, active=None, completed=0, archive=1, focus=True |
| `ABSTAIN` | `ABSTAIN` | False | 0.9973 | 0.9949 | `abstained` | backlog=0, active=None, completed=0, archive=1, focus=True |
