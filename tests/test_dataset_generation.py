from scripts.generate_dataset import build_dataset


def test_generated_dataset_has_stratified_split_metadata() -> None:
    df = build_dataset(examples_per_action=12, abstain_examples=18)

    assert set(df.columns) == {"text", "label", "split", "template_family"}
    assert set(df["split"]) == {"train", "validation", "test"}

    split_counts = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for split in ("train", "validation", "test"):
        assert (split_counts[split] > 0).all()
