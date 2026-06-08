from __future__ import annotations

import argparse
import random
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = (
    "CREATE_TASK",
    "PROMOTE_TASK",
    "COMPLETE_ACTIVE",
    "ARCHIVE_COMPLETED",
    "TOGGLE_FOCUS_MODE",
    "ABSTAIN",
)

EXECUTABLE_LABELS = LABELS[:-1]
SPLITS = ("train", "validation", "test")
EXECUTABLE_SUBSETS = ("clean_intent", "terse_intent", "polite_intent", "indirect_intent")
ABSTAIN_SUBSETS = (
    "negation",
    "compound",
    "near_miss",
    "unrelated",
    "gibberish",
    "unsafe",
    "ambiguous",
)
PUNCTUATION_TABLE = str.maketrans({char: " " for char in string.punctuation})

TOPICS = [
    "budget review",
    "lab notes",
    "client follow up",
    "reading list",
    "grant draft",
    "onboarding checklist",
    "design critique",
    "meeting recap",
    "invoice review",
    "research outline",
    "experiment log",
    "course prep",
    "supplier call",
    "prototype notes",
    "paper summary",
    "data cleanup",
    "feedback pass",
    "launch checklist",
    "travel receipt",
    "portfolio update",
    "weekly planning",
    "risk register",
    "bug triage",
    "user interview",
    "metrics review",
    "support reply",
    "release note",
    "contract comment",
    "demo script",
    "survey draft",
    "asset inventory",
    "roadmap note",
    "workshop agenda",
    "reading summary",
    "test plan",
    "migration note",
    "architecture sketch",
    "status update",
    "review checklist",
    "handoff note",
    "access request",
    "dependency audit",
    "slide outline",
    "vendor question",
    "schema sketch",
    "cleanup pass",
    "annotation batch",
    "bug report",
    "planning memo",
    "reference list",
]

EXECUTABLE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "CREATE_TASK": {
        "clean_intent": [
            "record a new backlog item for {topic}",
            "add a pending task about {topic}",
            "create a work item for {topic}",
            "save {topic} as something to do",
        ],
        "terse_intent": [
            "new task: {topic}",
            "todo item for {topic}",
            "backlog card: {topic}",
            "task entry: {topic}",
        ],
        "polite_intent": [
            "please add {topic} to my task list",
            "could you capture {topic} as a task",
            "please make a new todo for {topic}",
            "can you put {topic} in the backlog",
        ],
        "indirect_intent": [
            "I need to remember {topic} later",
            "{topic} should be tracked as pending work",
            "keep {topic} on my list of things to handle",
            "there should be a new task for {topic}",
        ],
    },
    "PROMOTE_TASK": {
        "clean_intent": [
            "move the next waiting item into active work for {topic}",
            "advance the first queued task for {topic} to active",
            "promote the next backlog card during {topic}",
            "bring the oldest pending item forward for {topic}",
        ],
        "terse_intent": [
            "start next queued item for {topic}",
            "activate backlog item for {topic}",
            "next backlog task into progress: {topic}",
            "promote waiting task: {topic}",
        ],
        "polite_intent": [
            "please make the next backlog item active for {topic}",
            "could you start the oldest queued task for {topic}",
            "please pull the next pending task forward for {topic}",
            "can you move the next queued item into progress for {topic}",
        ],
        "indirect_intent": [
            "the next waiting task should become active for {topic}",
            "I am ready for the next queued item in {topic}",
            "it is time to pick up the first backlog item for {topic}",
            "the active slot should take the next pending task for {topic}",
        ],
    },
    "COMPLETE_ACTIVE": {
        "clean_intent": [
            "mark the active task finished after {topic}",
            "set the current work item to complete for {topic}",
            "move the active item into completed for {topic}",
            "close out the task currently in progress for {topic}",
        ],
        "terse_intent": [
            "finish active task for {topic}",
            "current item done: {topic}",
            "complete active card for {topic}",
            "close current task: {topic}",
        ],
        "polite_intent": [
            "please mark the active item complete for {topic}",
            "could you finish the current task for {topic}",
            "please close the in-progress item for {topic}",
            "can you move the active work to done for {topic}",
        ],
        "indirect_intent": [
            "the current task is finished after {topic}",
            "I am done with the active work for {topic}",
            "the item in progress should be recorded as complete for {topic}",
            "the active card can be closed now for {topic}",
        ],
    },
    "ARCHIVE_COMPLETED": {
        "clean_intent": [
            "archive the finished items after {topic}",
            "file away completed work from {topic}",
            "move done tasks into the archive for {topic}",
            "clear completed task records into storage for {topic}",
        ],
        "terse_intent": [
            "archive done work: {topic}",
            "file completed tasks for {topic}",
            "clear finished items: {topic}",
            "store completed cards: {topic}",
        ],
        "polite_intent": [
            "please archive completed tasks for {topic}",
            "could you file away the finished work for {topic}",
            "please move completed items to the archive for {topic}",
            "can you clear the done list into archive for {topic}",
        ],
        "indirect_intent": [
            "the completed list can be archived after {topic}",
            "finished work from {topic} should be stored away",
            "done items for {topic} no longer need to stay visible",
            "the archive should receive completed tasks for {topic}",
        ],
    },
    "TOGGLE_FOCUS_MODE": {
        "clean_intent": [
            "switch the focus setting while working on {topic}",
            "toggle focus mode for the {topic} session",
            "flip the deep work setting during {topic}",
            "change the quiet mode flag for {topic}",
        ],
        "terse_intent": [
            "toggle focus: {topic}",
            "flip quiet mode for {topic}",
            "switch deep work mode: {topic}",
            "change focus flag for {topic}",
        ],
        "polite_intent": [
            "please toggle focus mode for {topic}",
            "could you switch the focus setting for {topic}",
            "please flip the quiet mode switch for {topic}",
            "can you change the deep work mode for {topic}",
        ],
        "indirect_intent": [
            "I want the focus setting changed for {topic}",
            "the quiet-work mode should switch for {topic}",
            "the controller should invert focus mode during {topic}",
            "focus mode needs to move to its other state for {topic}",
        ],
    },
}

ABSTAIN_PATTERNS: dict[str, list[str]] = {
    "negation": [
        "do not finish the active task while handling {topic}",
        "please do not create a task for {topic}",
        "do not promote the backlog item during {topic}",
        "leave completed work unarchived after {topic}",
        "do not toggle focus mode for {topic}",
    ],
    "compound": [
        "create a task for {topic} and archive everything completed",
        "start the next task for {topic} and then mark it done",
        "complete the active item for {topic} and switch focus mode",
        "archive finished work for {topic} and add a new task",
        "toggle focus for {topic} and promote the next item",
    ],
    "near_miss": [
        "rename the active task to {topic}",
        "set a deadline for {topic}",
        "assign the current task to {topic}",
        "delete the task related to {topic}",
        "export the task list for {topic}",
    ],
    "unrelated": [
        "explain the background of {topic}",
        "write a short poem about {topic}",
        "what is the weather near {topic}",
        "translate a sentence about {topic} into Spanish",
        "recommend a restaurant for discussing {topic}",
    ],
    "gibberish": [
        "zibbit {topic} wobble task cloud",
        "flarn archive {topic} seven blue",
        "qzx focus {topic} noodle syntax",
        "bliv active {topic} glass potato",
        "nargle done {topic} square humming",
    ],
    "unsafe": [
        "delete all files on this laptop after {topic}",
        "send private passwords for {topic} to a server",
        "disable security tools during {topic}",
        "wipe the project folder for {topic}",
        "run a destructive shell command about {topic}",
    ],
    "ambiguous": [
        "handle the task situation around {topic}",
        "do the usual workflow for {topic}",
        "make progress on {topic} somehow",
        "sort out my controller state for {topic}",
        "take care of {topic} in the normal way",
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


def _pick_topic(index: int, split: str, rng: random.Random) -> str:
    salt = {"train": 0, "validation": 101, "test": 211}[split]
    topic = TOPICS[(index + salt) % len(TOPICS)]
    if rng.random() < 0.20:
        return f"{topic} checkpoint {index + 1}"
    return topic


def _render(patterns: list[str], index: int, split: str, rng: random.Random) -> str:
    pattern = patterns[index % len(patterns)]
    topic = _pick_topic(index, split, rng)
    return " ".join(pattern.format(topic=topic).split())


def _append_row(
    rows: list[dict[str, str]],
    *,
    text: str,
    label: str,
    split: str,
    subset: str,
    local_index: int,
    source: str,
    notes: str,
) -> None:
    rows.append(
        {
            "text": text,
            "label": label,
            "split": split,
            "template_family": f"v2_{split}_{label}_{subset}_{local_index}",
            "subset": subset,
            "source": source,
            "notes": notes,
        }
    )


def _build_executable_rows(
    label: str,
    total: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts = split_counts(total)
    for split in SPLITS:
        for local_index in range(counts[split]):
            subset = EXECUTABLE_SUBSETS[local_index % len(EXECUTABLE_SUBSETS)]
            text = _render(EXECUTABLE_PATTERNS[label][subset], local_index, split, rng)
            _append_row(
                rows,
                text=text,
                label=label,
                split=split,
                subset=subset,
                local_index=local_index,
                source="v2_curated_synthetic",
                notes="Executable paraphrase with split-isolated template family.",
            )
    return rows


def _build_abstain_rows(total: int, rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts = split_counts(total)
    for split in SPLITS:
        for local_index in range(counts[split]):
            subset = ABSTAIN_SUBSETS[local_index % len(ABSTAIN_SUBSETS)]
            text = _render(ABSTAIN_PATTERNS[subset], local_index, split, rng)
            _append_row(
                rows,
                text=text,
                label="ABSTAIN",
                split=split,
                subset=subset,
                local_index=local_index,
                source="v2_curated_hard_negative",
                notes="Hard negative or out-of-scope request; must not mutate state.",
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


def build_dataset_v2(
    *,
    examples_per_class: int = 60,
    abstain_examples: int = 180,
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
    df = df[["text", "label", "split", "template_family", "subset", "source", "notes"]]

    validation = validate_dataset(df)
    if strict and any(
        validation[key] > 0
        for key in (
            "normalized_duplicate_count",
            "split_overlap_count",
            "template_family_overlap_count",
        )
    ):
        raise ValueError(f"V2 dataset strict validation failed: {validation}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the leakage-reduced v2 TaskFlow intent dataset."
    )
    parser.add_argument("--output", default="data/intent_dataset_v2.csv", help="CSV path to write.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed.")
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=60,
        help="Examples per executable action class.",
    )
    parser.add_argument(
        "--abstain-examples",
        type=int,
        default=180,
        help="Total ABSTAIN examples across hard-negative subsets.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise if duplicates, split overlap, or template-family leakage are found.",
    )
    args = parser.parse_args()

    df = build_dataset_v2(
        examples_per_class=args.examples_per_class,
        abstain_examples=args.abstain_examples,
        seed=args.seed,
        strict=args.strict,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))
    print(df["subset"].value_counts().sort_index())
    print(validate_dataset(df))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
