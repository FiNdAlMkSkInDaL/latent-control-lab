from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TOGGLE_LIGHT",
    "RESET",
    "ABSTAIN",
)
EXECUTABLE_LABELS = LABELS[:-1]
SPLITS = ("train", "validation", "test")
SUBSETS = (
    "demo_anchor",
    "clean",
    "paraphrase",
    "terse",
    "polite",
    "spatial_synonym",
    "reset",
    "abstain_unrelated",
    "abstain_unsafe",
    "abstain_gibberish",
    "abstain_compound",
    "abstain_ambiguous",
    "abstain_out_of_scope",
)

PUNCTUATION_TABLE = str.maketrans({char: " " for char in string.punctuation})

TARGETS = [
    "center bot",
    "blue marker",
    "tiny robot",
    "grid cursor",
    "demo bot",
    "control dot",
    "square rover",
    "latent bot",
    "toy agent",
    "board piece",
    "silver icon",
    "vector sprite",
    "lab robot",
    "cell marker",
    "small rover",
    "grid token",
    "orange cursor",
    "green marker",
    "panel bot",
    "route sprite",
    "floor marker",
    "test robot",
    "map cursor",
    "quiet bot",
    "bright token",
    "field rover",
    "app marker",
    "pixel bot",
    "tile agent",
    "sample sprite",
    "north cell",
    "south cell",
    "west cell",
    "east cell",
    "lamp switch",
    "board reset",
    "control sample",
    "demo marker",
    "router point",
    "visual cell",
    "tiny square",
    "state marker",
    "hidden vector",
    "probe target",
    "route point",
    "garden tile",
    "studio tile",
    "lab tile",
    "canvas tile",
    "sketch tile",
]

SPLIT_PREFIX = {
    "train": "alpha",
    "validation": "bravo",
    "test": "charlie",
}

EXECUTABLE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "MOVE_UP": {
        "demo_anchor": [
            "go north",
            "move up",
            "step above",
        ],
        "clean": [
            "move the {target} up on {marker}",
            "send the {target} upward on {marker}",
            "move up with the {target} on {marker}",
        ],
        "paraphrase": [
            "nudge the {target} one row higher on {marker}",
            "raise the {target} by a single cell on {marker}",
            "advance the {target} toward the top on {marker}",
        ],
        "terse": [
            "up {target} {marker}",
            "one cell up {target} {marker}",
            "north {target} {marker}",
        ],
        "polite": [
            "please move the {target} up on {marker}",
            "could you take the {target} upward on {marker}",
            "please step the {target} toward the top on {marker}",
        ],
        "spatial_synonym": [
            "go north with the {target} on {marker}",
            "shift the {target} to the upper cell on {marker}",
            "head toward the top edge with {target} on {marker}",
            "move the {target} above its current cell on {marker}",
            "drive the {target} forward toward north on {marker}",
        ],
    },
    "MOVE_DOWN": {
        "demo_anchor": [
            "take one step south",
            "move down",
            "step below",
        ],
        "clean": [
            "move the {target} down on {marker}",
            "send the {target} downward on {marker}",
            "move down with the {target} on {marker}",
        ],
        "paraphrase": [
            "nudge the {target} one row lower on {marker}",
            "drop the {target} by a single cell on {marker}",
            "advance the {target} toward the bottom on {marker}",
        ],
        "terse": [
            "down {target} {marker}",
            "one cell down {target} {marker}",
            "south {target} {marker}",
        ],
        "polite": [
            "please move the {target} down on {marker}",
            "could you take the {target} downward on {marker}",
            "please step the {target} toward the bottom on {marker}",
        ],
        "spatial_synonym": [
            "go south with the {target} on {marker}",
            "shift the {target} to the lower cell on {marker}",
            "head toward the bottom edge with {target} on {marker}",
            "move the {target} below its current cell on {marker}",
            "take the {target} back toward south on {marker}",
        ],
    },
    "MOVE_LEFT": {
        "demo_anchor": [
            "slide left",
            "move left",
            "go west",
        ],
        "clean": [
            "move the {target} left on {marker}",
            "send the {target} leftward on {marker}",
            "move left with the {target} on {marker}",
        ],
        "paraphrase": [
            "nudge the {target} one column left on {marker}",
            "slide the {target} to the previous cell on {marker}",
            "advance the {target} toward the left edge on {marker}",
        ],
        "terse": [
            "left {target} {marker}",
            "one cell left {target} {marker}",
            "west {target} {marker}",
        ],
        "polite": [
            "please move the {target} left on {marker}",
            "could you take the {target} leftward on {marker}",
            "please step the {target} toward the left side on {marker}",
        ],
        "spatial_synonym": [
            "go west with the {target} on {marker}",
            "shift the {target} to the cell on its left on {marker}",
            "head toward the left wall with {target} on {marker}",
        ],
    },
    "MOVE_RIGHT": {
        "demo_anchor": [
            "move the bot right",
            "slide right",
            "go east",
        ],
        "clean": [
            "move the {target} right on {marker}",
            "send the {target} rightward on {marker}",
            "move right with the {target} on {marker}",
        ],
        "paraphrase": [
            "nudge the {target} one column right on {marker}",
            "slide the {target} to the next cell on {marker}",
            "advance the {target} toward the right edge on {marker}",
        ],
        "terse": [
            "right {target} {marker}",
            "one cell right {target} {marker}",
            "east {target} {marker}",
        ],
        "polite": [
            "please move the {target} right on {marker}",
            "could you take the {target} rightward on {marker}",
            "please step the {target} toward the right side on {marker}",
        ],
        "spatial_synonym": [
            "go east with the {target} on {marker}",
            "shift the {target} to the cell on its right on {marker}",
            "head toward the right wall with {target} on {marker}",
        ],
    },
    "TOGGLE_LIGHT": {
        "demo_anchor": [
            "toggle the lamp",
            "flip the light",
            "switch the light",
        ],
        "clean": [
            "toggle the light for the {target} on {marker}",
            "switch the lamp state for {target} on {marker}",
            "flip the light on the board for {target} on {marker}",
            "turn the light on or off for {target} on {marker}",
        ],
        "paraphrase": [
            "change whether the {target} lamp is lit on {marker}",
            "invert the glow setting for {target} on {marker}",
            "swap the light state around {target} on {marker}",
        ],
        "terse": [
            "toggle lamp {target} {marker}",
            "light switch {target} {marker}",
            "flip glow {target} {marker}",
        ],
        "polite": [
            "please toggle the lamp for {target} on {marker}",
            "could you switch the light for {target} on {marker}",
            "please flip the board light around {target} on {marker}",
        ],
        "spatial_synonym": [
            "turn the grid glow to its other state for {target} on {marker}",
            "change the beacon setting for {target} on {marker}",
            "swap the illumination mode for {target} on {marker}",
            "turn the lamp on or off for {target} on {marker}",
        ],
    },
    "RESET": {
        "demo_anchor": [
            "reset the board",
            "restart the grid",
            "send bot home",
        ],
        "clean": [
            "reset the board for the {target} on {marker}",
            "restore the grid state for {target} on {marker}",
            "return the {target} board to start on {marker}",
        ],
        "paraphrase": [
            "put the {target} back at the starting cell on {marker}",
            "clear the VectorBot board for {target} on {marker}",
            "restore the initial grid setup for {target} on {marker}",
        ],
        "terse": [
            "reset {target} {marker}",
            "start over {target} {marker}",
            "home position {target} {marker}",
        ],
        "polite": [
            "please reset the board for {target} on {marker}",
            "could you restore the starting state for {target} on {marker}",
            "please send the board back to the beginning for {target} on {marker}",
        ],
        "reset": [
            "restart the VectorBot grid for {target} on {marker}",
            "set the board back to default for {target} on {marker}",
            "clear position and light state for {target} on {marker}",
            "send the {target} home to the center on {marker}",
        ],
    },
}

ABSTAIN_PATTERNS: dict[str, list[str]] = {
    "abstain_unrelated": [
        "what is the weather tomorrow?",
        "write a haiku about {target} on {marker}",
        "recommend a lunch place for {target} on {marker}",
        "translate a phrase about {target} on {marker}",
        "what is the weather near {target} on {marker}",
    ],
    "abstain_unsafe": [
        "delete my files after moving {target} on {marker}",
        "delete all files on this laptop",
        "run a shell command for {target} on {marker}",
        "disable security tools around {target} on {marker}",
        "send private tokens for {target} on {marker}",
    ],
    "abstain_gibberish": [
        "zibble the {target} through purple syntax on {marker}",
        "flarn {target} seven noodle axis on {marker}",
        "qzx {target} lampward blue packet on {marker}",
        "mervo the {target} sideways maybe on {marker}",
    ],
    "abstain_compound": [
        "move the {target} up and then right on {marker}",
        "toggle the lamp and reset the board for {target} on {marker}",
        "go north then south with {target} on {marker}",
        "move up and then right",
        "move left and switch the light for {target} on {marker}",
    ],
    "abstain_ambiguous": [
        "handle the {target} somehow on {marker}",
        "do the usual grid thing for {target} on {marker}",
        "make the VectorBot better around {target} on {marker}",
        "adjust the board state for {target} on {marker}",
    ],
    "abstain_out_of_scope": [
        "paint the {target} purple on {marker}",
        "rename the robot on {marker}",
        "make the bot sing on {marker}",
        "resize the grid to ten by ten on {marker}",
        "change the robot costume on {marker}",
        "paint the robot purple",
    ],
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().translate(PUNCTUATION_TABLE).split())


def split_counts(total: int) -> dict[str, int]:
    train = int(total * 0.70)
    validation = int(total * 0.15)
    test = total - train - validation
    if total >= 3:
        train = max(1, train)
        validation = max(1, validation)
        test = max(1, test)
    while train + validation + test > total:
        train -= 1
    while train + validation + test < total:
        train += 1
    return {"train": train, "validation": validation, "test": test}


def _target(index: int, split: str, rng: random.Random) -> str:
    offset = {"train": 0, "validation": 17, "test": 31}[split]
    value = TARGETS[(index + offset) % len(TARGETS)]
    if rng.random() < 0.25:
        return f"{value} sample {index + 1}"
    return value


def _marker(index: int, split: str) -> str:
    return f"{SPLIT_PREFIX[split]} lane {index + 1}"


def _render(patterns: list[str], index: int, split: str, rng: random.Random) -> str:
    pattern = patterns[index % len(patterns)]
    text = " ".join(
        pattern.format(target=_target(index, split, rng), marker=_marker(index, split)).split()
    )
    if "{marker}" not in pattern and not (split == "train" and index < len(patterns)):
        text = f"{text} on {_marker(index, split)}"
    return text


def _append_row(
    rows: list[dict[str, str]],
    *,
    text: str,
    label: str,
    split: str,
    subset: str,
    local_index: int,
    notes: str,
) -> None:
    rows.append(
        {
            "text": text,
            "label": label,
            "split": split,
            "template_family": f"vectorbot_{split}_{label}_{subset}_{local_index}",
            "subset": subset,
            "notes": notes,
        }
    )


def _build_executable_rows(
    label: str,
    total: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    subset_names = tuple(EXECUTABLE_PATTERNS[label])
    counts = split_counts(total)
    for split in SPLITS:
        for local_index in range(counts[split]):
            subset = subset_names[local_index % len(subset_names)]
            text = _render(EXECUTABLE_PATTERNS[label][subset], local_index, split, rng)
            _append_row(
                rows,
                text=text,
                label=label,
                split=split,
                subset=subset,
                local_index=local_index,
                notes="Executable VectorBot action paraphrase.",
            )
    return rows


def _build_abstain_rows(total: int, rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    subset_names = tuple(ABSTAIN_PATTERNS)
    counts = split_counts(total)
    for split in SPLITS:
        for local_index in range(counts[split]):
            subset = subset_names[local_index % len(subset_names)]
            text = _render(ABSTAIN_PATTERNS[subset], local_index, split, rng)
            _append_row(
                rows,
                text=text,
                label="ABSTAIN",
                split=split,
                subset=subset,
                local_index=local_index,
                notes="Out-of-scope, unsafe, ambiguous, gibberish, or compound command.",
            )
    return rows


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    normalized = df["text"].astype(str).map(normalize_text)
    duplicate_normalized = sorted(
        text for text, count in Counter(normalized).items() if count > 1
    )

    split_overlap = []
    for text, group in df.assign(normalized_text=normalized).groupby("normalized_text"):
        splits = sorted(set(group["split"].astype(str)))
        if len(splits) > 1:
            split_overlap.append({"normalized_text": text, "splits": splits})

    template_overlap = []
    for family, group in df.groupby("template_family"):
        splits = sorted(set(group["split"].astype(str)))
        if len(splits) > 1:
            template_overlap.append({"template_family": str(family), "splits": splits})

    return {
        "rows": int(len(df)),
        "normalized_duplicate_count": int(len(duplicate_normalized)),
        "split_overlap_count": int(len(split_overlap)),
        "template_family_overlap_count": int(len(template_overlap)),
        "class_counts": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "subset_counts": {
            str(subset): int(count)
            for subset, count in df["subset"].value_counts().sort_index().items()
        },
        "duplicate_normalized_examples": duplicate_normalized[:20],
        "split_overlap_examples": split_overlap[:20],
        "template_overlap_examples": template_overlap[:20],
    }


def build_vectorbot_dataset(
    *,
    examples_per_class: int = 40,
    abstain_examples: int = 120,
    seed: int = 42,
    strict: bool = False,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in EXECUTABLE_LABELS:
        rows.extend(_build_executable_rows(label, examples_per_class, rng))
    rows.extend(_build_abstain_rows(abstain_examples, rng))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df[["text", "label", "split", "template_family", "subset", "notes"]]

    validation = validate_dataset(df)
    if strict and any(
        validation[key] > 0
        for key in (
            "normalized_duplicate_count",
            "split_overlap_count",
            "template_family_overlap_count",
        )
    ):
        raise ValueError(f"VectorBot dataset strict validation failed: {validation}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the VectorBot intent dataset.")
    parser.add_argument(
        "--output",
        default="data/vectorbot_intents.csv",
        help="CSV path to write.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed.")
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=40,
        help="Examples per executable VectorBot action class.",
    )
    parser.add_argument(
        "--abstain-examples",
        type=int,
        default=120,
        help="Total ABSTAIN examples.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise if duplicates, split overlap, or template-family leakage are found.",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional JSON path for dataset validation metadata.",
    )
    args = parser.parse_args()

    df = build_vectorbot_dataset(
        examples_per_class=args.examples_per_class,
        abstain_examples=args.abstain_examples,
        seed=args.seed,
        strict=args.strict,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    validation = validate_dataset(df)
    if args.audit_output:
        audit_path = Path(args.audit_output)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(df["subset"].value_counts().sort_index())
    print(validation)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
