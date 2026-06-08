from __future__ import annotations

import pandas as pd

from scripts.audit_dataset import audit_dataset, find_near_duplicates, normalize_text


def test_normalize_text_removes_case_and_punctuation() -> None:
    assert normalize_text("Create, A Task!!") == "create a task"


def test_audit_dataset_flags_split_and_template_overlap(tmp_path) -> None:
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "text": "Create a task",
                "label": "CREATE_TASK",
                "split": "train",
                "template_family": "create_0",
            },
            {
                "text": "create a task!",
                "label": "CREATE_TASK",
                "split": "test",
                "template_family": "create_0",
            },
            {
                "text": "what is the weather",
                "label": "ABSTAIN",
                "split": "test",
                "template_family": "abstain_0",
            },
        ]
    ).to_csv(dataset, index=False)

    audit, near_duplicates = audit_dataset(dataset, similarity_threshold=0.75)

    assert audit["normalized_duplicate_count"] == 1
    assert audit["split_text_overlap_count"] == 1
    assert audit["template_family_overlap_count"] == 1
    assert audit["leakage_flags"]["has_template_family_overlap"] is True
    assert not near_duplicates.empty


def test_find_near_duplicates_returns_expected_columns() -> None:
    df = pd.DataFrame(
        [
            {"text": "move next task active", "label": "PROMOTE_TASK", "split": "train"},
            {"text": "move next active task", "label": "PROMOTE_TASK", "split": "test"},
        ]
    )

    near_duplicates = find_near_duplicates(df, similarity_threshold=0.5)

    assert list(near_duplicates.columns) == [
        "row_a",
        "row_b",
        "split_a",
        "split_b",
        "label_a",
        "label_b",
        "similarity",
        "text_a",
        "text_b",
    ]
    assert near_duplicates.iloc[0]["similarity"] >= 0.5
