from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_native.bridge.train_sklearn import train_probe
from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

try:
    from scripts.evaluate_hard_eval import classification_metrics
    from scripts.extract_features import safe_model_slug
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from evaluate_hard_eval import classification_metrics
    from extract_features import safe_model_slug


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def save_features(path: Path, X: np.ndarray, df: pd.DataFrame, model_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=df["label"].astype(str).to_numpy(),
        text=df["text"].astype(str).to_numpy(),
        split=df["split"].astype(str).to_numpy(),
        template_family=df["template_family"].astype(str).to_numpy(),
        subset=df["subset"].astype(str).to_numpy(),
        model_id=np.array([model_id]),
        prompt_template=np.array([PROMPT_TEMPLATE]),
        feature_space=np.array(["pre_lm_head_last_token"]),
    )


def compare_model(
    model_id: str,
    normal_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    *,
    batch_size: int,
    max_length: int,
    work_dir: Path,
    no_4bit: bool,
) -> dict[str, Any]:
    tokenizer, model = load_causal_lm(model_id, use_4bit=not no_4bit)
    tap = PreLMHeadActivationTap(model)
    try:
        X = extract_vectors(
            normal_df["text"].astype(str).tolist(),
            tokenizer,
            model,
            tap,
            batch_size=batch_size,
            max_length=max_length,
        )
        X_hard = extract_vectors(
            hard_df["text"].astype(str).tolist(),
            tokenizer,
            model,
            tap,
            batch_size=batch_size,
            max_length=max_length,
        )
    finally:
        tap.close()

    slug = safe_model_slug(model_id)
    feature_path = work_dir / f"features_{slug}_comparison.npz"
    probe_path = work_dir / f"probe_{slug}_comparison.joblib"
    save_features(feature_path, X, normal_df, model_id)
    bundle = train_probe(feature_path, probe_path)
    test_metrics = bundle["metrics"]["splits"]["test"]
    hard_pred = bundle["probe"].predict(X_hard).astype(str)
    hard_metrics = classification_metrics(
        hard_df["label"].astype(str).to_numpy(),
        hard_pred,
        hard_df["subset"].astype(str).to_numpy(),
        labels=sorted(set(bundle["metrics"]["labels"]) | set(hard_df["label"].astype(str))),
    )
    return {
        "model_id": model_id,
        "status": "ok",
        "semantic_evidence": model_id != "sshleifer/tiny-gpt2",
        "feature_shape": list(X.shape),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "hard_eval_accuracy": hard_metrics["overall_accuracy"],
        "hard_eval_macro_f1": hard_metrics["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare small HF causal LMs on the v2 latent-routing task."
    )
    parser.add_argument("--dataset", default="data/intent_dataset_v2.csv")
    parser.add_argument("--hard-dataset", default="data/hard_eval.csv")
    parser.add_argument(
        "--models",
        default="distilgpt2,gpt2,sshleifer/tiny-gpt2",
        help="Comma-separated model ids to try.",
    )
    parser.add_argument("--output-json", default="artifacts/model_comparison.json")
    parser.add_argument("--output-csv", default="artifacts/model_comparison.csv")
    parser.add_argument("--sample-size", type=int, default=180)
    parser.add_argument("--hard-sample-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", default="tmp/model_comparison")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    normal_df = sample_frame(pd.read_csv(args.dataset), args.sample_size, args.seed)
    hard_df = sample_frame(pd.read_csv(args.hard_dataset), args.hard_sample_size, args.seed)
    rows = []
    for model_id in [model.strip() for model in args.models.split(",") if model.strip()]:
        try:
            row = compare_model(
                model_id,
                normal_df,
                hard_df,
                batch_size=args.batch_size,
                max_length=args.max_length,
                work_dir=Path(args.work_dir),
                no_4bit=args.no_4bit,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "model_id": model_id,
                "status": "failed",
                "semantic_evidence": False,
                "error": str(exc)[:500],
            }
        rows.append(row)
        print(row)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "hard_dataset": args.hard_dataset,
        "sample_size": args.sample_size,
        "hard_sample_size": args.hard_sample_size,
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
