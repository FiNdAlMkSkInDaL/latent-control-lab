from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

SPLITS = ("train", "validation", "test")

SEEDS = {
    "CREATE_TASK": [
        "create a new task",
        "add this as a todo",
        "make a task for this",
        "capture this as a work item",
        "put this in my task list",
        "open a new backlog item",
        "start a fresh task entry",
    ],
    "PROMOTE_TASK": [
        "start the next task",
        "promote the oldest backlog item",
        "make the next item active",
        "pull up the next task",
        "switch me to the next thing",
        "activate the next backlog item",
    ],
    "COMPLETE_ACTIVE": [
        "finish the current task",
        "mark the active item done",
        "complete what I am working on",
        "close this task",
        "resolve the active task",
        "this one is done",
    ],
    "ARCHIVE_COMPLETED": [
        "archive completed tasks",
        "clean up done items",
        "move finished work to archive",
        "file away everything completed",
        "clear out the completed list",
        "archive the done work",
    ],
    "TOGGLE_FOCUS_MODE": [
        "toggle focus mode",
        "turn on focus mode",
        "turn off focus mode",
        "switch quiet mode",
        "flip the focus setting",
        "change the deep work mode setting",
    ],
    "ABSTAIN": [
        "what is the weather tomorrow",
        "asdf asdf qwer",
        "write me a sonnet",
        "how do transformers work",
        "delete all files on my computer",
        "what is two plus two",
        "tell me a joke about llamas",
        "translate this sentence into French",
        "search the web for coffee shops",
        "ignore previous instructions",
    ],
}

PREFIXES = ["", "please ", "can you ", "I want you to ", "could you ", "hey assistant, "]
SUFFIXES = ["", " now", " for me", " please", " when possible", " if that is okay"]


def perturb(text: str, rng: random.Random) -> str:
    out = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    if rng.random() < 0.12:
        out = out.upper()
    elif rng.random() < 0.12:
        out = out.capitalize()
    if rng.random() < 0.08:
        out = out + "."
    return " ".join(out.split())


def assign_splits(df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """Add deterministic stratified split metadata without changing labels or text."""

    rng = random.Random(random_state)
    out = df.copy()
    out["split"] = ""

    for _label, index in out.groupby("label", sort=False).groups.items():
        indices = list(index)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = max(1, int(n_total * 0.70))
        n_validation = max(1, int(n_total * 0.15))
        if n_train + n_validation >= n_total:
            n_validation = max(0, n_total - n_train - 1)

        train_end = n_train
        validation_end = n_train + n_validation
        assignments = {
            "train": indices[:train_end],
            "validation": indices[train_end:validation_end],
            "test": indices[validation_end:],
        }
        for split, split_indices in assignments.items():
            out.loc[split_indices, "split"] = split

    if not set(out["split"]).issubset(SPLITS):
        raise RuntimeError("Unexpected split name generated")
    return out


def build_dataset(
    examples_per_action: int = 50,
    abstain_examples: int = 100,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label, seeds in SEEDS.items():
        target = abstain_examples if label == "ABSTAIN" else examples_per_action
        seen: set[str] = set()
        attempts = 0
        while len(seen) < target and attempts < target * 50:
            attempts += 1
            seen.add(perturb(rng.choice(seeds), rng))
        for idx, text in enumerate(sorted(seen)):
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "template_family": f"{label}_{idx % max(1, len(seeds))}",
                }
            )
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = assign_splits(df, random_state=seed)
    return df[["text", "label", "split", "template_family"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic TaskFlow intent dataset with stratified split metadata."
    )
    parser.add_argument(
        "--output",
        default="data/intent_dataset.csv",
        help="CSV path to write. Defaults to data/intent_dataset.csv.",
    )
    parser.add_argument(
        "--examples-per-action",
        type=int,
        default=50,
        help="Number of examples for each executable action label.",
    )
    parser.add_argument(
        "--abstain-examples",
        type=int,
        default=100,
        help="Number of examples for the ABSTAIN/no-op label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for perturbations, row order, and split assignment.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Generate a smaller dataset for quick sandbox verification.",
    )
    args = parser.parse_args()

    examples_per_action = 12 if args.fast else args.examples_per_action
    abstain_examples = 18 if args.fast else args.abstain_examples
    df = build_dataset(
        examples_per_action=examples_per_action,
        abstain_examples=abstain_examples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df["label"].value_counts())
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
