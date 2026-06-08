from __future__ import annotations

from scripts.generate_dataset_v2 import (
    ABSTAIN_SUBSETS,
    LABELS,
    SPLITS,
    build_dataset_v2,
    normalize_text,
    validate_dataset,
)


def test_dataset_v2_required_columns_labels_and_splits() -> None:
    df = build_dataset_v2(examples_per_class=12, abstain_examples=28, seed=7, strict=True)

    assert list(df.columns) == [
        "text",
        "label",
        "split",
        "template_family",
        "subset",
        "source",
        "notes",
    ]
    assert set(LABELS).issubset(set(df["label"]))
    assert set(SPLITS).issubset(set(df["split"]))


def test_dataset_v2_has_no_duplicate_or_split_leakage() -> None:
    df = build_dataset_v2(examples_per_class=16, abstain_examples=35, seed=13, strict=True)
    validation = validate_dataset(df)

    assert validation["normalized_duplicate_count"] == 0
    assert validation["split_overlap_count"] == 0
    assert validation["template_family_overlap_count"] == 0
    assert len(set(df["text"].map(normalize_text))) == len(df)


def test_dataset_v2_class_counts_and_hard_negative_subsets_are_sane() -> None:
    df = build_dataset_v2(examples_per_class=20, abstain_examples=49, seed=42, strict=True)

    counts = df["label"].value_counts()
    for label in LABELS[:-1]:
        assert counts[label] == 20
    assert counts["ABSTAIN"] == 49
    assert set(ABSTAIN_SUBSETS).issubset(set(df["subset"]))
    assert {"negation", "compound"}.issubset(
        set(df[df["split"].isin(["train", "validation"])]["subset"])
    )
