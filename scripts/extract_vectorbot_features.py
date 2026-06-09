from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from neural_native.llm.extractor import PROMPT_TEMPLATE, extract_vectors
from neural_native.llm.hooks import PreLMHeadActivationTap
from neural_native.llm.loader import load_causal_lm

try:
    from scripts.generate_vectorbot_dataset import LABELS
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_vectorbot_dataset import LABELS


def safe_model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_")


def _sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    groups = list(df.groupby(["label", "split"], sort=True))
    minimum = min(len(df), max(sample_size, len(groups)))
    base_rows = [group.sample(n=1, random_state=seed).index[0] for _key, group in groups]
    remaining = df.drop(index=base_rows)
    extra_count = minimum - len(base_rows)
    if extra_count > 0:
        extra = remaining.sample(n=extra_count, random_state=seed).index.tolist()
    else:
        extra = []
    selected = base_rows + extra
    return df.loc[selected].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _fake_feature_vector(text: str, label: str, *, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{label}\n{text}".encode(), digest_size=32).digest()
    noise = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    noise = np.resize((noise - 127.5) / 255.0, dim) * 0.08
    z = noise.astype(np.float32)
    label_index = LABELS.index(label)
    offset = label_index * 4
    z[offset] += 4.0
    z[offset + 1] += 2.0
    z[offset + 2] += 1.0
    return z


def make_fake_features(df: pd.DataFrame, *, dim: int = 32) -> np.ndarray:
    min_dim = len(LABELS) * 4
    dim = max(dim, min_dim)
    return np.stack(
        [
            _fake_feature_vector(str(row.text), str(row.label), dim=dim)
            for row in df.itertuples(index=False)
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract VectorBot final-token pre-lm_head activations from a frozen causal LM. "
            "This performs forward passes only and never calls model.generate()."
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/vectorbot_intents.csv",
        help="VectorBot intent CSV from scripts/generate_vectorbot_dataset.py.",
    )
    parser.add_argument(
        "--model-id",
        default="distilgpt2",
        help="Hugging Face causal LM id, for example distilgpt2 or sshleifer/tiny-gpt2.",
    )
    parser.add_argument(
        "--output",
        "--features-output",
        dest="output",
        default=None,
        help="NPZ feature path. Defaults to artifacts/vectorbot_features_<model>.npz.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per forward batch.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Tokenizer truncation length for the router prompt.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Prompt template containing a {text} placeholder.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified-ish row sample for fast verification.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Create deterministic label-coded fake vectors for blocked-download plumbing checks.",
    )
    parser.add_argument(
        "--fake-dim",
        type=int,
        default=32,
        help="Feature dimension for --fake vectors.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = _sample_frame(df, args.sample_size, args.seed)

    feature_space = "pre_lm_head_last_token"
    model_id = args.model_id
    if args.fake:
        X = make_fake_features(df, dim=args.fake_dim)
        feature_space = "fake_label_coded_vectorbot_plumbing"
        model_id = f"fake-vectorbot-plumbing:{args.model_id}"
    else:
        try:
            tokenizer, model = load_causal_lm(args.model_id, use_4bit=False)
            tap = PreLMHeadActivationTap(model)
            try:
                X = extract_vectors(
                    df["text"].astype(str).tolist(),
                    tokenizer,
                    model,
                    tap,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    prompt_template=args.prompt_template,
                )
            finally:
                tap.close()
        except Exception as exc:  # noqa: BLE001
            print(
                "Failed to load model or extract VectorBot activations. "
                "If this environment blocks Hugging Face downloads, rerun with --fake "
                "for plumbing-only artifacts or provide a cached model.",
                file=sys.stderr,
            )
            print(f"Model id: {args.model_id}", file=sys.stderr)
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    output = args.output or f"artifacts/vectorbot_features_{safe_model_slug(args.model_id)}.npz"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=df["label"].astype(str).to_numpy(),
        text=df["text"].astype(str).to_numpy(),
        split=df["split"].astype(str).to_numpy(),
        template_family=df["template_family"].astype(str).to_numpy(),
        subset=df["subset"].astype(str).to_numpy(),
        model_id=np.array([model_id]),
        prompt_template=np.array([args.prompt_template]),
        feature_space=np.array([feature_space]),
    )
    print(f"X={X.shape}; wrote {output_path}")


if __name__ == "__main__":
    main()
