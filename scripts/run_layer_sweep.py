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
from neural_native.llm.hooks import PreLMHeadActivationTap, TransformerBlockActivationTap
from neural_native.llm.loader import load_causal_lm

try:
    from scripts.evaluate_hard_eval import classification_metrics
    from scripts.extract_features import safe_model_slug
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from evaluate_hard_eval import classification_metrics
    from extract_features import safe_model_slug


def sample_by_label_split(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    groups = list(df.groupby(["label", "split"], sort=False))
    for _key, group in groups:
        proportional = int(round(sample_size * len(group) / len(df)))
        n_take = min(len(group), max(1, proportional))
        chosen = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        selected.update(int(i) for i in chosen)

    if len(selected) > sample_size:
        selected = set(rng.choice(np.array(sorted(selected)), size=sample_size, replace=False))
    elif len(selected) < sample_size:
        remaining = np.array([idx for idx in df.index if idx not in selected])
        if len(remaining):
            fill_n = min(sample_size - len(selected), len(remaining))
            selected.update(int(i) for i in rng.choice(remaining, size=fill_n, replace=False))

    return df.loc[sorted(selected)].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_tap(model: Any, layer_spec: str) -> tuple[str, Any]:
    if layer_spec == "pre_lm_head":
        return "pre_lm_head", PreLMHeadActivationTap(model)
    if layer_spec in {"block_0", "early"}:
        tap = TransformerBlockActivationTap(model, 0)
    elif layer_spec in {"block_mid", "middle"}:
        tap = TransformerBlockActivationTap(model, "middle")
    elif layer_spec in {"block_last", "final"}:
        tap = TransformerBlockActivationTap(model, "final")
    elif layer_spec.startswith("block_"):
        tap = TransformerBlockActivationTap(model, int(layer_spec.split("_", maxsplit=1)[1]))
    else:
        raise ValueError(f"Unknown layer spec: {layer_spec}")
    return tap.layer_name, tap


def save_features(
    path: Path,
    X: np.ndarray,
    df: pd.DataFrame,
    *,
    model_id: str,
    feature_space: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template_family = df.get("template_family", pd.Series(["unknown"] * len(df)))
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=df["label"].astype(str).to_numpy(),
        text=df["text"].astype(str).to_numpy(),
        split=df["split"].astype(str).to_numpy(),
        template_family=template_family.astype(str).to_numpy(),
        model_id=np.array([model_id]),
        prompt_template=np.array([PROMPT_TEMPLATE]),
        feature_space=np.array([feature_space]),
    )


def _report_value(report: dict[str, Any], label: str, key: str) -> float | None:
    if label not in report:
        return None
    return report[label].get(key)


def evaluate_hard(
    probe: Any,
    X: np.ndarray,
    hard_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    y_true = hard_df["label"].astype(str).to_numpy()
    y_pred = probe.predict(X)
    return classification_metrics(
        y_true,
        y_pred.astype(str),
        hard_df["subset"].astype(str).to_numpy(),
        labels=sorted(set(labels) | set(y_true) | {"ABSTAIN"}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probes on early/middle/final/pre-lm_head activations."
    )
    parser.add_argument("--model-id", default="distilgpt2", help="Hugging Face causal LM id.")
    parser.add_argument("--dataset", default="data/intent_dataset.csv", help="Synthetic CSV.")
    parser.add_argument("--hard-dataset", default="data/hard_eval.csv", help="Hard eval CSV.")
    parser.add_argument(
        "--layers",
        default="block_0,block_mid,block_last,pre_lm_head",
        help="Comma-separated layer specs: block_0, block_mid, block_last, pre_lm_head.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/layer_sweep_metrics.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/layer_sweep_metrics.csv",
        help="CSV output path.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per forward batch.")
    parser.add_argument("--max-length", type=int, default=160, help="Tokenizer max length.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional normal sample.")
    parser.add_argument(
        "--hard-sample-size",
        type=int,
        default=None,
        help="Optional hard-eval sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--work-dir",
        default="tmp/layer_sweep",
        help="Temporary feature directory.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    args = parser.parse_args()

    normal_df = sample_by_label_split(pd.read_csv(args.dataset), args.sample_size, args.seed)
    hard_df = pd.read_csv(args.hard_dataset)
    if args.hard_sample_size is not None and args.hard_sample_size < len(hard_df):
        hard_df = hard_df.sample(
            n=args.hard_sample_size,
            random_state=args.seed,
        ).reset_index(drop=True)

    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    work_dir = Path(args.work_dir)
    rows: list[dict[str, Any]] = []
    layer_specs = [layer.strip() for layer in args.layers.split(",") if layer.strip()]

    for layer_spec in layer_specs:
        layer_name, tap = build_tap(model, layer_spec)
        feature_space = f"{layer_name}_last_token"
        try:
            X = extract_vectors(
                normal_df["text"].astype(str).tolist(),
                tokenizer,
                model,
                tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            X_hard = extract_vectors(
                hard_df["text"].astype(str).tolist(),
                tokenizer,
                model,
                tap,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
        finally:
            tap.close()

        slug = f"{safe_model_slug(args.model_id)}_{layer_name}"
        features_path = work_dir / f"features_{slug}.npz"
        probe_path = work_dir / f"probe_{slug}.joblib"
        save_features(
            features_path,
            X,
            normal_df,
            model_id=args.model_id,
            feature_space=feature_space,
        )
        bundle = train_probe(features_path, probe_path)
        metrics = bundle["metrics"]
        test_report = metrics["splits"]["test"]["classification_report"]
        hard_metrics = evaluate_hard(bundle["probe"], X_hard, hard_df, metrics["labels"])

        rows.append(
            {
                "layer_spec": layer_spec,
                "layer_name": layer_name,
                "feature_space": feature_space,
                "activation_shape": list(X.shape),
                "train_accuracy": metrics["splits"]["train"]["accuracy"],
                "validation_accuracy": metrics["splits"]["validation"]["accuracy"],
                "test_accuracy": metrics["splits"]["test"]["accuracy"],
                "macro_f1": metrics["splits"]["test"]["macro_f1"],
                "abstain_precision": _report_value(test_report, "ABSTAIN", "precision"),
                "abstain_recall": _report_value(test_report, "ABSTAIN", "recall"),
                "hard_eval_accuracy": hard_metrics["overall_accuracy"],
                "hard_eval_macro_f1": hard_metrics["macro_f1"],
                "hard_eval_abstain_precision": hard_metrics["abstain_precision"],
                "hard_eval_abstain_recall": hard_metrics["abstain_recall"],
            }
        )
        print(
            f"{layer_name}: test={rows[-1]['test_accuracy']}, "
            f"hard={rows[-1]['hard_eval_accuracy']}"
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
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
