from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_hard_eval import build_hard_eval


def test_hard_eval_has_required_columns_subsets_and_counts() -> None:
    df = build_hard_eval()

    assert set(df.columns) == {"text", "label", "subset", "expected_behavior", "notes"}
    assert {
        "human_paraphrase",
        "negation",
        "compound",
        "near_miss",
        "gibberish",
        "unrelated",
        "unsafe",
        "ambiguous",
    }.issubset(set(df["subset"]))

    counts = df["label"].value_counts()
    for label in [
        "CREATE_TASK",
        "PROMOTE_TASK",
        "COMPLETE_ACTIVE",
        "ARCHIVE_COMPLETED",
        "TOGGLE_FOCUS_MODE",
    ]:
        assert counts[label] >= 25
    assert counts["ABSTAIN"] >= 50
    assert len(df[df["subset"] == "negation"]) >= 30
    assert len(df[df["subset"] == "compound"]) >= 30
    assert len(df[df["subset"] == "near_miss"]) >= 30


def test_hard_eval_has_no_exact_overlap_with_synthetic_dataset() -> None:
    df = build_hard_eval()
    synthetic_path = Path("data/intent_dataset.csv")
    if not synthetic_path.exists():
        return

    synthetic = pd.read_csv(synthetic_path)
    hard_text = set(df["text"].astype(str).str.strip().str.lower())
    synthetic_text = set(synthetic["text"].astype(str).str.strip().str.lower())
    assert hard_text.isdisjoint(synthetic_text)
