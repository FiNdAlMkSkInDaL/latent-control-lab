from __future__ import annotations

import pandas as pd

from scripts.run_baselines import text_baseline_rows


def test_text_baseline_rows_smoke() -> None:
    train_df = pd.DataFrame(
        [
            {"text": "create task", "label": "CREATE_TASK"},
            {"text": "add task", "label": "CREATE_TASK"},
            {"text": "weather question", "label": "ABSTAIN"},
            {"text": "tell joke", "label": "ABSTAIN"},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"text": "create new task", "label": "CREATE_TASK"},
            {"text": "weather now", "label": "ABSTAIN"},
        ]
    )
    hard_df = pd.DataFrame(
        [
            {
                "text": "jot this as work",
                "label": "CREATE_TASK",
                "subset": "human_paraphrase",
            },
            {"text": "delete files", "label": "ABSTAIN", "subset": "unsafe"},
        ]
    )

    rows = text_baseline_rows(
        train_df,
        test_df,
        hard_df,
        labels=["ABSTAIN", "CREATE_TASK"],
        seed=42,
    )

    assert {row["baseline"] for row in rows} == {
        "tfidf_logreg_text",
        "random_stratified",
        "majority_class",
    }
    assert {row["dataset"] for row in rows} == {"synthetic_test", "hard_eval"}
