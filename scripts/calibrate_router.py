from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from neural_native.eval.calibration import (
    mahalanobis_parameters,
    score_methods,
    select_threshold,
)
from neural_native.eval.ood import classwise_centroid_thresholds

OBJECTIVE_FORMULA = (
    "0.4 * executable_accept_rate + "
    "0.4 * hard_negative_reject_rate + "
    "0.2 * macro_f1"
)


def _metric_value(metric_fn: Any, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    finite = np.isfinite(scores)
    if finite.sum() == 0 or len(set(y_true[finite].tolist())) < 2:
        return None
    return float(metric_fn(y_true[finite], scores[finite]))


def load_feature_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    y = data["y"].astype(str)
    return {
        "X": data["X"].astype(np.float32),
        "y": y,
        "split": data["split"].astype(str)
        if "split" in data.files
        else np.array(["unknown"] * len(y)),
        "subset": data["subset"].astype(str)
        if "subset" in data.files
        else np.array(["unknown"] * len(y)),
        "model_id": str(data["model_id"][0]) if "model_id" in data.files else "unknown",
        "feature_space": str(data["feature_space"][0])
        if "feature_space" in data.files
        else "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select calibrated router/OOD thresholds from validation features."
    )
    parser.add_argument("--probe", default="artifacts/probe_distilgpt2_v2.joblib")
    parser.add_argument(
        "--features",
        default="artifacts/features_distilgpt2_v2_pre_lm_head.npz",
        help="V2 feature NPZ with train/validation/test split metadata.",
    )
    parser.add_argument(
        "--output-calibration",
        default="artifacts/router_calibration_v2.json",
        help="Full calibration report.",
    )
    parser.add_argument(
        "--output-thresholds",
        default="artifacts/router_thresholds_calibrated_v2.json",
        help="Router thresholds JSON.",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.probe)
    data = load_feature_npz(args.features)
    probe = bundle["probe"]
    classes = list(bundle["classes"])
    labels = sorted(set(classes) | set(data["y"]))
    centroids = bundle.get("centroids", {})

    train_exec_mask = (data["split"] == "train") & (data["y"] != "ABSTAIN")
    validation_mask = data["split"] == "validation"
    if not train_exec_mask.any() or not validation_mask.any():
        raise ValueError("Need train executable rows and validation rows for calibration")

    X_train_exec = data["X"][train_exec_mask]
    y_train_exec = data["y"][train_exec_mask]
    X_validation = data["X"][validation_mask]
    y_validation = data["y"][validation_mask]
    probs_validation = probe.predict_proba(X_validation)
    predicted_validation = np.asarray(classes, dtype=object)[np.argmax(probs_validation, axis=1)]
    id_labels = (y_validation != "ABSTAIN").astype(int)

    method_scores = score_methods(
        X_validation,
        probs=probs_validation,
        classes=classes,
        centroids=centroids,
        X_train_executable=X_train_exec,
        y_train_executable=y_train_exec,
    )

    rows = []
    best_row = None
    for method, scores in method_scores.items():
        result = select_threshold(
            y_validation,
            predicted_validation.astype(str),
            scores,
            labels,
        )
        result.method = method
        row = {
            "method": method,
            "threshold": result.threshold,
            "objective": result.objective,
            "executable_accept_rate": result.executable_accept_rate,
            "hard_negative_reject_rate": result.hard_negative_reject_rate,
            "macro_f1": result.macro_f1,
            "auroc": _metric_value(roc_auc_score, id_labels, scores),
            "auprc": _metric_value(average_precision_score, id_labels, scores),
        }
        rows.append(row)
        if best_row is None or row["objective"] > best_row["objective"]:
            best_row = row

    if best_row is None:
        raise RuntimeError("No calibration method was selected")

    selected_parameters: dict[str, Any] = {}
    parameter_shapes: dict[str, Any] = {}
    if best_row["method"] == "classwise_centroid_threshold":
        selected_parameters["classwise_centroid_thresholds"] = classwise_centroid_thresholds(
            X_train_exec,
            y_train_exec,
            centroids,
            excluded_labels={"ABSTAIN"},
        )
    elif best_row["method"] == "negative_mahalanobis_distance":
        mahalanobis = mahalanobis_parameters(X_train_exec)
        selected_parameters["mahalanobis"] = mahalanobis
        parameter_shapes["mahalanobis_mean"] = [len(mahalanobis["mean"])]
        parameter_shapes["mahalanobis_precision"] = [
            len(mahalanobis["precision"]),
            len(mahalanobis["precision"][0]) if mahalanobis["precision"] else 0,
        ]
    thresholds_payload = {
        "selection": "validation_only",
        "objective_formula": OBJECTIVE_FORMULA,
        "ood_method": best_row["method"],
        "ood_threshold": best_row["threshold"],
        "min_confidence": 0.0,
        "min_margin": 0.0,
        "max_centroid_distance": None,
        "parameters": selected_parameters,
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe": args.probe,
        "features": args.features,
        "model_id": data["model_id"],
        "feature_space": data["feature_space"],
        "objective_formula": OBJECTIVE_FORMULA,
        "validation_rows": int(validation_mask.sum()),
        "train_executable_rows": int(train_exec_mask.sum()),
        "methods": rows,
        "selected": best_row,
        "selected_parameter_shapes": parameter_shapes,
        "thresholds": thresholds_payload,
    }

    output_calibration = Path(args.output_calibration)
    output_calibration.parent.mkdir(parents=True, exist_ok=True)
    output_calibration.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_thresholds = Path(args.output_thresholds)
    output_thresholds.parent.mkdir(parents=True, exist_ok=True)
    output_thresholds.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    pd.DataFrame(rows).to_csv(output_calibration.with_suffix(".csv"), index=False)
    print(json.dumps(best_row, indent=2))
    print(f"wrote {output_calibration}")
    print(f"wrote {output_thresholds}")


if __name__ == "__main__":
    main()
