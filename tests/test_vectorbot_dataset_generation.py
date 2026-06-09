from scripts.generate_vectorbot_dataset import (
    LABELS,
    SPLITS,
    SUBSETS,
    build_vectorbot_dataset,
    normalize_text,
    validate_dataset,
)


def test_vectorbot_dataset_required_columns_labels_and_splits() -> None:
    df = build_vectorbot_dataset(examples_per_class=12, abstain_examples=30, seed=7, strict=True)
    assert list(df.columns) == [
        "text",
        "label",
        "split",
        "template_family",
        "subset",
        "notes",
    ]
    assert set(LABELS).issubset(set(df["label"]))
    assert set(SPLITS).issubset(set(df["split"]))


def test_vectorbot_dataset_has_no_duplicate_or_split_leakage() -> None:
    df = build_vectorbot_dataset(examples_per_class=16, abstain_examples=35, seed=13, strict=True)
    validation = validate_dataset(df)
    assert validation["normalized_duplicate_count"] == 0
    assert validation["split_overlap_count"] == 0
    assert validation["template_family_overlap_count"] == 0
    assert len(set(df["text"].map(normalize_text))) == len(df)


def test_vectorbot_dataset_counts_and_subsets_are_sane() -> None:
    df = build_vectorbot_dataset(examples_per_class=20, abstain_examples=50, seed=42, strict=True)
    counts = df["label"].value_counts()
    for label in LABELS[:-1]:
        assert counts[label] == 20
    assert counts["ABSTAIN"] == 50
    assert set(SUBSETS).issubset(set(df["subset"]))
