from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def safe_model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def run_step(command: list[str], *, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    print("\n$ " + " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(result.returncode)
    return result


def feature_is_fake(path: Path) -> bool:
    if not path.exists():
        return False
    data = np.load(path, allow_pickle=True)
    model_id = str(data["model_id"][0]) if "model_id" in data else ""
    return model_id.startswith("fake-vectorbot-plumbing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CPU-friendly VectorBot dataset -> features -> probe -> "
            "demo -> visuals path."
        )
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Frozen causal LM id.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller dataset and max length.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional feature extraction row sample.",
    )
    parser.add_argument("--skip-extract", action="store_true", help="Reuse existing feature NPZ.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing probe bundle.")
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip PNG/projection generation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset and sampling seed.")
    parser.add_argument("--dataset-output", default=None, help="Dataset CSV output path.")
    parser.add_argument("--features-output", default=None, help="Feature NPZ output path.")
    parser.add_argument("--probe-output", default=None, help="Probe joblib output path.")
    parser.add_argument("--metrics-output", default=None, help="Metrics JSON output path.")
    parser.add_argument(
        "--confusion-output",
        default=None,
        help="Confusion matrix CSV output path.",
    )
    parser.add_argument("--thresholds-output", default=None, help="Thresholds JSON output path.")
    parser.add_argument("--routes-output", default=None, help="Route log JSONL output path.")
    parser.add_argument("--projection-output", default=None, help="Projection CSV output path.")
    args = parser.parse_args()

    py = sys.executable
    slug = safe_model_slug(args.model_id)
    dataset_path = Path(args.dataset_output or "data/vectorbot_intents.csv")
    features_path = Path(f"artifacts/vectorbot_features_{slug}.npz")
    if args.model_id == "distilgpt2":
        features_path = Path("artifacts/vectorbot_features_distilgpt2.npz")
    if args.features_output:
        features_path = Path(args.features_output)
    probe_path = Path(f"artifacts/vectorbot_probe_{slug}.joblib")
    if args.model_id == "distilgpt2":
        probe_path = Path("artifacts/vectorbot_probe_distilgpt2.joblib")
    if args.probe_output:
        probe_path = Path(args.probe_output)
    metrics_path = args.metrics_output or "artifacts/vectorbot_metrics.json"
    confusion_path = args.confusion_output or "artifacts/vectorbot_confusion_matrix.csv"
    thresholds_path = args.thresholds_output or "artifacts/vectorbot_thresholds.json"
    routes_path = args.routes_output or "artifacts/vectorbot_routes.jsonl"
    projection_path = args.projection_output or "artifacts/vectorbot_projection.csv"

    examples_per_class = "12" if args.fast else "40"
    abstain_examples = "36" if args.fast else "120"
    max_length = "64" if args.fast else "96"
    batch_size = "8"

    run_step(
        [
            py,
            "scripts/generate_vectorbot_dataset.py",
            "--output",
            str(dataset_path),
            "--seed",
            str(args.seed),
            "--examples-per-class",
            examples_per_class,
            "--abstain-examples",
            abstain_examples,
            "--strict",
        ]
    )

    used_fake_features = feature_is_fake(features_path)
    if not args.skip_extract:
        extract_command = [
            py,
            "scripts/extract_vectorbot_features.py",
            "--dataset",
            str(dataset_path),
            "--model-id",
            args.model_id,
            "--output",
            str(features_path),
            "--batch-size",
            batch_size,
            "--max-length",
            max_length,
            "--seed",
            str(args.seed),
        ]
        if args.sample_size is not None:
            extract_command.extend(["--sample-size", str(args.sample_size)])
        result = run_step(extract_command, allow_fail=True)
        if result.returncode != 0:
            print(
                "Real model extraction did not complete. Falling back to deterministic "
                "fake vectors for plumbing, visuals, and replay-only demo evidence."
            )
            run_step(
                [
                    py,
                    "scripts/extract_vectorbot_features.py",
                    "--dataset",
                    str(dataset_path),
                    "--model-id",
                    args.model_id,
                    "--output",
                    str(features_path),
                    "--seed",
                    str(args.seed),
                    "--fake",
                ]
            )
            used_fake_features = True
        else:
            used_fake_features = feature_is_fake(features_path)

    if not args.skip_train:
        run_step(
            [
                py,
                "scripts/train_vectorbot_probe.py",
                "--features",
                str(features_path),
                "--output",
                str(probe_path),
                "--metrics",
                metrics_path,
                "--confusion-matrix",
                confusion_path,
                "--thresholds",
                thresholds_path,
            ]
        )

    demo_command = [
        py,
        "scripts/run_vectorbot_demo.py",
        "--scripted",
        "--output",
        routes_path,
    ]
    if used_fake_features:
        demo_command.append("--fake")
    else:
        demo_command.extend(
            [
                "--model-id",
                args.model_id,
                "--probe",
                str(probe_path),
                "--thresholds-json",
                thresholds_path,
                "--batch-size",
                batch_size,
                "--max-length",
                max_length,
            ]
        )
    run_step(demo_command)

    if not args.skip_visuals:
        run_step(
            [
                py,
                "scripts/build_vectorbot_visuals.py",
                "--features",
                str(features_path),
                "--dataset",
                str(dataset_path),
                "--routes",
                routes_path,
                "--projection-output",
                projection_path,
            ]
        )

    mode = "fake plumbing fallback" if used_fake_features else "real hidden-state extraction"
    print(f"\nVectorBot quickstart complete ({mode}).")


if __name__ == "__main__":
    main()
