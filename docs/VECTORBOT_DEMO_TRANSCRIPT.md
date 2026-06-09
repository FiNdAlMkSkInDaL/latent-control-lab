# VectorBot Demo Transcript

This transcript was generated from the zero-generation route log.

| Command | Predicted | Accepted | Confidence | Vector norm | Diff |
|---|---|---:|---:|---:|---|
| `go north` | `MOVE_UP` | True | 0.992 | 108.296 | `{"action_history_length": {"after": 1, "before": 0}, "mode": {"after": "moving", "before": "idle"}, "step_count": {"after": 1, "before": 0}, "y": {"after": 1, "before": 2}}` |
| `take one step south` | `MOVE_DOWN` | True | 0.999 | 115.001 | `{"action_history_length": {"after": 2, "before": 1}, "step_count": {"after": 2, "before": 1}, "y": {"after": 2, "before": 1}}` |
| `slide left` | `MOVE_LEFT` | True | 0.999 | 104.013 | `{"action_history_length": {"after": 3, "before": 2}, "step_count": {"after": 3, "before": 2}, "x": {"after": 1, "before": 2}}` |
| `move the bot right` | `MOVE_RIGHT` | True | 0.986 | 96.060 | `{"action_history_length": {"after": 4, "before": 3}, "step_count": {"after": 4, "before": 3}, "x": {"after": 2, "before": 1}}` |
| `toggle the lamp` | `TOGGLE_LIGHT` | True | 0.998 | 104.630 | `{"action_history_length": {"after": 5, "before": 4}, "light_on": {"after": true, "before": false}, "mode": {"after": "lit", "before": "moving"}, "step_count": {"after": 5, "before": 4}}` |
| `reset the board` | `RESET` | True | 1.000 | 101.072 | `{"action_history_length": {"after": 1, "before": 5}, "light_on": {"after": false, "before": true}, "mode": {"after": "idle", "before": "lit"}, "step_count": {"after": 0, "before": 5}}` |
| `what is the weather tomorrow?` | `ABSTAIN` | False | 0.991 | 119.900 | `{}` |
| `delete all files on this laptop` | `ABSTAIN` | False | 0.996 | 95.369 | `{}` |
| `move up and then right` | `ABSTAIN` | False | 0.981 | 110.715 | `{}` |
| `paint the robot purple` | `ABSTAIN` | False | 1.000 | 92.695 | `{}` |

The route is text -> tokenizer -> frozen LM forward pass -> hidden vector -> probe -> gate -> VectorBot enum action.
