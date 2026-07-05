"""Concept vector / directional analysis over frozen LLM activations.

This module operates purely on extracted feature matrices + labels.
It never calls model.generate() and does not influence the production routing path.

Used for analysis + visuals that demonstrate the geometry of the latent action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class DirectionBundle:
    labels: list[str]
    means: dict[str, np.ndarray]  # class centroid in activation space
    directions: dict[str, np.ndarray]  # contrastive direction vectors (executable labels)
    abstain_label: str = "ABSTAIN"


def compute_class_means(
    X: np.ndarray, y: np.ndarray, labels: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Return mean activation vector per class label."""
    means: dict[str, np.ndarray] = {}
    uniq = np.unique(y) if labels is None else labels
    for lab in uniq:
        mask = y == lab
        if np.any(mask):
            means[str(lab)] = X[mask].mean(axis=0).astype(np.float32)
    return means


def compute_contrastive_directions(
    means: dict[str, np.ndarray], abstain_label: str = "ABSTAIN"
) -> dict[str, np.ndarray]:
    """For each executable label, direction = mean(label) - mean(other executables).

    ABSTAIN is excluded from the 'other' pool for direction computation.
    """
    directions: dict[str, np.ndarray] = {}
    exec_labels = [lab for lab in means if lab != abstain_label]
    if not exec_labels:
        return directions

    # Global executable mean (for contrast)
    exec_stack = np.stack([means[lab] for lab in exec_labels])
    global_exec = exec_stack.mean(axis=0)

    for lab in exec_labels:
        directions[lab] = (means[lab] - global_exec).astype(np.float32)
    return directions


def build_direction_bundle(
    X: np.ndarray, y: np.ndarray, *, abstain_label: str = "ABSTAIN"
) -> DirectionBundle:
    labels = sorted({str(v) for v in np.unique(y)})
    means = compute_class_means(X, y, labels=labels)
    directions = compute_contrastive_directions(means, abstain_label=abstain_label)
    return DirectionBundle(
        labels=labels, means=means, directions=directions, abstain_label=abstain_label
    )


def steer_vector(
    z: np.ndarray, direction: np.ndarray, eps: float = 0.6
) -> np.ndarray:
    """Return z + eps * unit(direction). Purely for analysis / illustration."""
    d = direction.astype(np.float32)
    norm = np.linalg.norm(d) + 1e-9
    unit = d / norm
    return (z.astype(np.float32) + eps * unit).astype(np.float32)


def apply_probe_to_vectors(
    probe: Any,  # sklearn pipeline with predict_proba
    vectors: list[np.ndarray],
) -> list[dict[str, float]]:
    """Return top label + prob for a list of vectors. For steering demos."""
    out = []
    for z in vectors:
        z2 = np.asarray(z, dtype=np.float32).reshape(1, -1)
        probs = probe.predict_proba(z2)[0]
        classes = list(probe.classes_)
        idx = int(np.argmax(probs))
        out.append({"label": classes[idx], "confidence": float(probs[idx])})
    return out
