from __future__ import annotations

import argparse
import json
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PUNCTUATION_TABLE = str.maketrans({char: " " for char in string.punctuation})
NEAR_DUPLICATE_COLUMNS = [
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


def normalize_text(text: str) -> str:
    cleaned = text.lower().translate(PUNCTUATION_TABLE)
    return " ".join(cleaned.split())


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _nested_counts(df: pd.DataFrame, index: str, columns: str) -> dict[str, dict[str, int]]:
    table = df.groupby([index, columns], dropna=False).size().unstack(fill_value=0)
    return {
        str(row_label): {str(col): int(value) for col, value in row.items()}
        for row_label, row in table.iterrows()
    }


def _split_overlap(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    overlaps = []
    for value, group in df.groupby(key, dropna=False):
        splits = sorted(set(group["split"].astype(str)))
        if len(splits) > 1:
            overlaps.append(
                {
                    key: str(value),
                    "splits": splits,
                    "rows": [int(i) for i in group.index.tolist()],
                    "count": int(len(group)),
                }
            )
    return overlaps


def find_near_duplicates(
    df: pd.DataFrame,
    *,
    similarity_threshold: float,
) -> pd.DataFrame:
    tokens = [token_set(str(text)) for text in df["text"]]
    rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = jaccard_similarity(tokens[i], tokens[j])
            if similarity < similarity_threshold:
                continue
            rows.append(
                {
                    "row_a": i,
                    "row_b": j,
                    "split_a": str(df.iloc[i]["split"]),
                    "split_b": str(df.iloc[j]["split"]),
                    "label_a": str(df.iloc[i]["label"]),
                    "label_b": str(df.iloc[j]["label"]),
                    "similarity": float(similarity),
                    "text_a": str(df.iloc[i]["text"]),
                    "text_b": str(df.iloc[j]["text"]),
                }
            )
    return pd.DataFrame(rows, columns=NEAR_DUPLICATE_COLUMNS)


def audit_dataset(
    dataset: str | Path,
    *,
    similarity_threshold: float = 0.82,
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(dataset)
    required = {"text", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["normalized_text"] = df["text"].astype(str).map(normalize_text)
    if "template_family" not in df:
        df["template_family"] = "unknown"

    exact_duplicates = [
        {
            "text": str(text),
            "count": int(count),
        }
        for text, count in Counter(df["text"].astype(str)).items()
        if count > 1
    ]
    normalized_duplicates = [
        {
            "normalized_text": str(text),
            "count": int(count),
        }
        for text, count in Counter(df["normalized_text"].astype(str)).items()
        if count > 1
    ]
    split_text_overlap = _split_overlap(df, "normalized_text")
    template_family_overlap = _split_overlap(df, "template_family")
    near_duplicates = find_near_duplicates(df, similarity_threshold=similarity_threshold)
    cross_split_near_duplicates = near_duplicates[
        near_duplicates["split_a"] != near_duplicates["split_b"]
    ]

    abstain = df[df["label"] == "ABSTAIN"]
    token_counts = [len(token_set(str(text))) for text in abstain["text"]]
    audit = {
        "dataset": str(dataset),
        "n_rows": int(len(df)),
        "labels": sorted(map(str, df["label"].unique())),
        "splits": sorted(map(str, df["split"].unique())),
        "similarity_threshold": float(similarity_threshold),
        "exact_duplicate_count": int(len(exact_duplicates)),
        "normalized_duplicate_count": int(len(normalized_duplicates)),
        "split_text_overlap_count": int(len(split_text_overlap)),
        "template_family_overlap_count": int(len(template_family_overlap)),
        "near_duplicate_pair_count": int(len(near_duplicates)),
        "cross_split_near_duplicate_pair_count": int(len(cross_split_near_duplicates)),
        "exact_duplicates": exact_duplicates[:50],
        "normalized_duplicates": normalized_duplicates[:50],
        "split_text_overlap": split_text_overlap[:50],
        "template_family_overlap": template_family_overlap[:100],
        "per_class_split_counts": _nested_counts(df, "label", "split"),
        "per_class_template_family_counts": _nested_counts(df, "label", "template_family"),
        "abstain_diversity": {
            "rows": int(len(abstain)),
            "unique_normalized_text": int(abstain["normalized_text"].nunique()),
            "unique_template_families": int(abstain["template_family"].nunique()),
            "mean_token_count": float(sum(token_counts) / len(token_counts))
            if token_counts
            else 0.0,
        },
        "leakage_flags": {
            "has_exact_split_overlap": bool(split_text_overlap),
            "has_template_family_overlap": bool(template_family_overlap),
            "has_cross_split_near_duplicates": bool(len(cross_split_near_duplicates)),
        },
    }
    return audit, near_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the synthetic intent dataset for duplicates and split leakage."
    )
    parser.add_argument(
        "--dataset",
        default="data/intent_dataset.csv",
        help="Intent CSV to audit.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/dataset_audit.json",
        help="Path for the JSON audit summary.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/dataset_near_duplicates.csv",
        help="Path for pair-level near-duplicate rows.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.82,
        help="Token Jaccard threshold for near-duplicate pairs.",
    )
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="Exit non-zero when split overlap, template leakage, or near duplicates appear.",
    )
    args = parser.parse_args()

    audit, near_duplicates = audit_dataset(
        args.dataset,
        similarity_threshold=args.similarity_threshold,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    near_duplicates.to_csv(output_csv, index=False)

    print(json.dumps({key: audit[key] for key in audit if key.endswith("_count")}, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")

    if args.fail_on_leakage and any(audit["leakage_flags"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
