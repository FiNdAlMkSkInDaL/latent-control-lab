#!/usr/bin/env python
"""Compute concept vectors (class means + contrastive directions) over frozen LLM activations.

This is analysis / visualization code. It does not affect the zero-generation routing path
used by VectorBotVectorPort / kernel.

Run after extracting features (or use --use-probe-centroids for a lighter path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from neural_native.bridge.directions import (
    DirectionBundle,
    apply_probe_to_vectors,
    build_direction_bundle,
    steer_vector,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def load_features_and_labels(features_path: Path | None, probe_path: Path | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prefer raw features npz if present. Fall back to synthetic data for demo."""
    if features_path and features_path.exists():
        data = np.load(features_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        labels = list(data["labels"]) if "labels" in data else sorted(set(str(v) for v in y))
        return X, y, labels

    if probe_path and probe_path.exists():
        try:
            bundle = joblib.load(probe_path)
            centroids = bundle.get("centroids")
            if centroids:
                labels = list(centroids.keys())
                rng = np.random.default_rng(42)
                X_list, y_list = [], []
                for lab, c in centroids.items():
                    for _ in range(6):
                        X_list.append(c + rng.normal(0, 0.07, size=c.shape).astype(np.float32))
                        y_list.append(lab)
                return np.stack(X_list), np.array(y_list, dtype=object), labels
        except Exception:
            pass

    # Final fallback: pure synthetic for demo / CI smoke
    rng = np.random.default_rng(123)
    labels = ["ABSTAIN", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "TOGGLE_LIGHT", "RESET"]
    X_list, y_list = [], []
    for lab in labels:
        c = rng.normal(0, 1, size=16).astype(np.float32)
        for _ in range(7):
            X_list.append(c + rng.normal(0, 0.1, size=16).astype(np.float32))
            y_list.append(lab)
    return np.stack(X_list), np.array(y_list, dtype=object), labels


def save_bundle(bundle: DirectionBundle, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": bundle.labels,
        "abstain_label": bundle.abstain_label,
        "means": {k: v.tolist() for k, v in bundle.means.items()},
        "directions": {k: v.tolist() for k, v in bundle.directions.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_directions(bundle: DirectionBundle, out_path: Path, max_dim: int = 32) -> None:
    if plt is None:
        return
    # Show a slice of a few directions as bar plots (first max_dim dims)
    exec_dirs = {k: v for k, v in bundle.directions.items() if k != bundle.abstain_label}
    if not exec_dirs:
        return
    n = min(len(exec_dirs), 4)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (lab, vec) in zip(axes, list(exec_dirs.items())[:n], strict=False):
        ax.bar(range(min(max_dim, len(vec))), vec[:max_dim])
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("dim slice")
    fig.suptitle("Contrastive direction vectors (activation space slice)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def demo_steering(bundle: DirectionBundle, probe: Any, out_json: Path) -> dict[str, Any]:
    """Pick a borderline example (synthetic) and show steering effect."""
    # Use an ABSTAIN-ish or low-conf vector: average of two close dirs or random around mean
    rng = np.random.default_rng(123)
    abstain_mean = bundle.means.get(bundle.abstain_label)
    if abstain_mean is None:
        abstain_mean = next(iter(bundle.means.values()))

    z0 = (abstain_mean + rng.normal(0, 0.04, size=abstain_mean.shape)).astype(np.float32)

    results: list[dict[str, Any]] = []
    base = apply_probe_to_vectors(probe, [z0])[0]
    results.append({"eps": 0.0, **base})

    for lab, d in bundle.directions.items():
        z_steered = steer_vector(z0, d, eps=0.75)
        steered = apply_probe_to_vectors(probe, [z_steered])[0]
        results.append({"eps": 0.75, "steered_toward": lab, **steered})

    payload = {"base_vector_stats": {"norm": float(np.linalg.norm(z0))}, "steering_results": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction / concept vector analysis for VectorBot latent space.")
    parser.add_argument("--features", default="artifacts/vectorbot_features_distilgpt2_full.npz")
    parser.add_argument("--probe", default="artifacts/vectorbot_probe_distilgpt2_full.joblib")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--directions-out", default=None)
    parser.add_argument("--steering-out", default=None)
    parser.add_argument("--plot-out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    feats = Path(args.features)
    probe_p = Path(args.probe)

    X, y, labels = load_features_and_labels(feats if feats.exists() else None, probe_p if probe_p.exists() else None)

    bundle = build_direction_bundle(X, y, abstain_label="ABSTAIN")

    directions_path = Path(args.directions_out) if args.directions_out else out_dir / "vectorbot_concept_directions.json"
    save_bundle(bundle, directions_path)
    print(f"wrote {directions_path}")

    # Probe for steering demo
    probe = None
    if probe_p.exists():
        b = joblib.load(probe_p)
        probe = b["probe"]

    if probe is not None:
        steering_path = Path(args.steering_out) if args.steering_out else out_dir / "vectorbot_steering_demo.json"
        demo_steering(bundle, probe, steering_path)
        print(f"wrote {steering_path}")

    if args.plot_out or True:
        plot_path = Path(args.plot_out) if args.plot_out else out_dir / "vectorbot_direction_slices.png"
        plot_directions(bundle, plot_path)
        if plot_path.exists():
            print(f"wrote {plot_path}")

    print("Concept vector analysis complete. (evaluation-only; does not change routing path)")


if __name__ == "__main__":
    main()
