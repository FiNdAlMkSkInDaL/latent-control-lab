from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from neural_native.eval.ood import score_id_vs_abstain


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ABSTAIN/OOD separation with max probability, top-2 margin, "
            "and executable-centroid distance."
        )
    )
    parser.add_argument(
        "--probe",
        required=True,
        help="Joblib probe bundle produced by scripts/train_probe.py.",
    )
    parser.add_argument(
        "--features",
        required=True,
        help="NPZ activation artifact to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/ood_metrics.json",
        help="JSON path for OOD AUROC metrics.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of feature rows to sample for quick verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-size is provided.",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.probe)
    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype(str)
    if args.sample_size is not None:
        rng = np.random.default_rng(args.seed)
        sample_size = min(args.sample_size, len(y))
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    # Treat non-ABSTAIN examples as ID and ABSTAIN examples as OOD/no-op for a first baseline.
    id_mask = y != "ABSTAIN"
    ood_mask = y == "ABSTAIN"
    if not id_mask.any() or not ood_mask.any():
        raise ValueError("Need both ID and ABSTAIN/OOD rows in features")

    probe = bundle["probe"]
    scores = score_id_vs_abstain(probe, X, y, bundle.get("centroids"))
    y_true = id_mask.astype(int)

    metrics = {
        name: float(roc_auc_score(y_true, score[~np.isnan(score)]))
        if not np.isnan(score).any()
        else float(roc_auc_score(y_true[~np.isnan(score)], score[~np.isnan(score)]))
        for name, score in scores.items()
    }

    print(json.dumps(metrics, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
