from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm


def safe_model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract final-token pre-lm_head activation vectors from a frozen "
            "Hugging Face causal LM. This performs forward passes only."
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/intent_dataset.csv",
        help="Intent CSV produced by scripts/generate_dataset.py.",
    )
    parser.add_argument(
        "--model-id",
        default="distilgpt2",
        help="Hugging Face causal LM id, for example distilgpt2 or sshleifer/tiny-gpt2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="NPZ feature path. Defaults to artifacts/features_<model>_pre_lm_head.npz.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Texts per forward-pass batch.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=160,
        help="Tokenizer truncation length for the router prompt.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization on CUDA even when bitsandbytes is available.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of dataset rows to sample for quick verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-size is provided.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if args.sample_size is not None:
        sample_size = min(args.sample_size, len(df))
        df = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    tokenizer, model = load_causal_lm(args.model_id, use_4bit=not args.no_4bit)
    tap = PreLMHeadActivationTap(model)

    try:
        X = extract_vectors(
            df["text"].tolist(),
            tokenizer,
            model,
            tap,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    finally:
        tap.close()

    output = args.output or f"artifacts/features_{safe_model_slug(args.model_id)}_pre_lm_head.npz"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=df["label"].astype(str).to_numpy(),
        text=df["text"].astype(str).to_numpy(),
        split=df["split"].astype(str).to_numpy()
        if "split" in df
        else np.array(["unknown"] * len(df)),
        template_family=df["template_family"].astype(str).to_numpy()
        if "template_family" in df
        else np.array(["unknown"] * len(df)),
        model_id=np.array([args.model_id]),
        prompt_template=np.array([PROMPT_TEMPLATE]),
        feature_space=np.array(["pre_lm_head_last_token"]),
    )
    print(f"X={X.shape}; wrote {output_path}")


if __name__ == "__main__":
    main()
