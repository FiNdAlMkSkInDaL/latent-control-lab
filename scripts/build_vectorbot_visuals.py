from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from scripts.generate_vectorbot_dataset import LABELS, build_vectorbot_dataset
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from generate_vectorbot_dataset import LABELS, build_vectorbot_dataset

PALETTE = {
    "MOVE_UP": "#2563eb",
    "MOVE_DOWN": "#16a34a",
    "MOVE_LEFT": "#d97706",
    "MOVE_RIGHT": "#dc2626",
    "TOGGLE_LIGHT": "#9333ea",
    "RESET": "#0891b2",
    "ABSTAIN": "#525252",
}


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    return plt, Rectangle, FancyArrowPatch


def _load_pillow():
    from PIL import Image, ImageColor, ImageDraw, ImageFont

    return Image, ImageColor, ImageDraw, ImageFont


def _hex(color: str) -> tuple[int, int, int]:
    try:
        _image, image_color, _draw, _font = _load_pillow()
        return image_color.getrgb(color)
    except Exception:  # noqa: BLE001
        color = color.lstrip("#")
        return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(
        b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _fallback_png(path: Path, *, color: tuple[int, int, int]) -> None:
    width, height = 900, 520
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            shade = 1.0 - 0.25 * (x / width) - 0.15 * (y / height)
            pixels.extend(max(0, min(255, int(channel * shade))) for channel in color)
    _write_png(path, width, height, bytes(pixels))


def _pil_grid(path: Path) -> None:
    try:
        Image, _image_color, ImageDraw, ImageFont = _load_pillow()
    except Exception:  # noqa: BLE001
        _fallback_png(path, color=(59, 130, 246))
        return
    image = Image.new("RGB", (900, 620), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((36, 28), "VectorBot Grid Demo", fill="#0f172a", font=font)
    draw.text((36, 52), "state: x=3, y=2, light=ON, action=MOVE_RIGHT", fill="#334155", font=font)
    left, top, cell = 250, 105, 72
    for y in range(5):
        for x in range(5):
            fill = "#eef2ff" if (x + y) % 2 == 0 else "#f8fafc"
            draw.rectangle(
                [left + x * cell, top + y * cell, left + (x + 1) * cell, top + (y + 1) * cell],
                fill=fill,
                outline="#334155",
                width=2,
            )
            draw.text(
                (left + x * cell + 31, top + y * cell + 27),
                ".",
                fill="#64748b",
                font=font,
            )
    bx, by = 3, 2
    draw.rectangle(
        [
            left + bx * cell + 7,
            top + by * cell + 7,
            left + (bx + 1) * cell - 7,
            top + (by + 1) * cell - 7,
        ],
        fill="#facc15",
        outline="#854d0e",
        width=3,
    )
    draw.text((left + bx * cell + 31, top + by * cell + 27), "B", fill="#0f172a", font=font)
    image.save(path)


def _pil_confidence(path: Path, routes_path: Path) -> None:
    try:
        Image, _image_color, ImageDraw, ImageFont = _load_pillow()
    except Exception:  # noqa: BLE001
        _fallback_png(path, color=(22, 163, 74))
        return
    routes = _routes_by_text(routes_path)
    if routes:
        first = next(iter(routes.values()))
        items = first.get("top_probabilities", [])
    else:
        items = [
            {"label": "MOVE_UP", "probability": 0.73},
            {"label": "MOVE_RIGHT", "probability": 0.18},
            {"label": "ABSTAIN", "probability": 0.04},
        ]
    image = Image.new("RGB", (900, 520), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((36, 28), "Top VectorBot Route Probabilities", fill="#0f172a", font=font)
    for index, item in enumerate(items[:3]):
        label = str(item["label"])
        value = float(item["probability"])
        y = 110 + index * 90
        draw.text((60, y), f"{label}  {value:.2f}", fill="#0f172a", font=font)
        draw.rectangle([60, y + 24, 780, y + 54], fill="#e5e7eb")
        draw.rectangle(
            [60, y + 24, 60 + int(720 * value), y + 54],
            fill=PALETTE.get(label, "#64748b"),
        )
    image.save(path)


def _pil_latent(path: Path, projection: pd.DataFrame) -> None:
    try:
        Image, _image_color, ImageDraw, ImageFont = _load_pillow()
    except Exception:  # noqa: BLE001
        _fallback_png(path, color=(147, 51, 234))
        return
    image = Image.new("RGB", (900, 620), "#ffffff")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (36, 28),
        "2D Projection of VectorBot Hidden-State Features",
        fill="#0f172a",
        font=font,
    )
    x_min, x_max = float(projection["x"].min()), float(projection["x"].max())
    y_min, y_max = float(projection["y"].min()), float(projection["y"].max())
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    left, top, width, height = 80, 78, 620, 470
    draw.rectangle([left, top, left + width, top + height], outline="#cbd5e1", width=2)
    for row in projection.itertuples(index=False):
        px = left + int((float(row.x) - x_min) / (x_max - x_min) * width)
        py = top + height - int((float(row.y) - y_min) / (y_max - y_min) * height)
        color = PALETTE.get(str(row.label), "#64748b")
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color, outline="#ffffff")
    for index, label in enumerate(LABELS):
        y = 100 + index * 28
        draw.rectangle([735, y, 749, y + 14], fill=PALETTE.get(label, "#64748b"))
        draw.text((758, y - 1), label, fill="#0f172a", font=font)
    image.save(path)


def _pil_pipeline(path: Path) -> None:
    try:
        Image, _image_color, ImageDraw, ImageFont = _load_pillow()
    except Exception:  # noqa: BLE001
        _fallback_png(path, color=(8, 145, 178))
        return
    image = Image.new("RGB", (1100, 360), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    stages = [
        "command",
        "tiny model\nforward pass",
        "hidden vector\nhook",
        "linear probe\n+ gate",
        "VectorBot\naction",
    ]
    for index, stage in enumerate(stages):
        x = 42 + index * 205
        draw.rectangle([x, 115, x + 150, 190], fill="#ffffff", outline="#0f172a", width=2)
        for line_index, line in enumerate(stage.splitlines()):
            draw.text((x + 18, 132 + line_index * 18), line, fill="#0f172a", font=font)
        if index < len(stages) - 1:
            draw.line([x + 154, 152, x + 196, 152], fill="#334155", width=3)
            draw.polygon(
                [(x + 196, 152), (x + 184, 144), (x + 184, 160)],
                fill="#334155",
            )
    draw.text(
        (42, 252),
        "No generated commands. No JSON/tool-call parser. No keyword router.",
        fill="#334155",
        font=font,
    )
    image.save(path)


def plot_composite(path: Path, output_dir: Path) -> None:
    try:
        Image, _image_color, ImageDraw, ImageFont = _load_pillow()
    except Exception:  # noqa: BLE001
        _fallback_png(path, color=(8, 145, 178))
        return

    image = Image.new("RGB", (1200, 760), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((34, 28), "Tiny Latent Control Lab", fill="#0f172a", font=font)
    draw.text(
        (34, 52),
        "Frozen transformer activations -> probe -> VectorBot action",
        fill="#334155",
        font=font,
    )

    slots = [
        ("vectorbot_grid_demo.png", (34, 96, 570, 430), "Grid state"),
        ("vectorbot_confidence_bars.png", (618, 96, 1148, 330), "Route confidence"),
        ("vectorbot_latent_space.png", (34, 472, 570, 720), "Latent projection"),
        ("vectorbot_pipeline_diagram.png", (618, 382, 1148, 604), "No-generation path"),
    ]
    for filename, box, title in slots:
        source = output_dir / filename
        if not source.exists():
            continue
        panel = Image.open(source).convert("RGB")
        width = box[2] - box[0]
        height = box[3] - box[1]
        panel.thumbnail((width, height - 24))
        draw.rectangle(box, fill="#ffffff", outline="#cbd5e1", width=2)
        draw.text((box[0] + 12, box[1] + 8), title, fill="#0f172a", font=font)
        image.paste(panel, (box[0] + 8, box[1] + 30))

    draw.text(
        (618, 650),
        "Invariant: no model.generate(), no JSON/tool parser, no keyword router.",
        fill="#334155",
        font=font,
    )
    image.save(path)


def _routes_by_text(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["input_text"]): row for row in rows}


def _fallback_projection(dataset: pd.DataFrame) -> pd.DataFrame:
    angles = {label: index * (2 * math.pi / len(LABELS)) for index, label in enumerate(LABELS)}
    rows = []
    for row in dataset.itertuples(index=False):
        label = str(row.label)
        digest = hashlib.blake2b(str(row.text).encode("utf-8"), digest_size=4).digest()
        jitter_a = (digest[0] / 255.0 - 0.5) * 0.35
        jitter_b = (digest[1] / 255.0 - 0.5) * 0.35
        angle = angles[label]
        radius = 1.0 + (digest[2] / 255.0) * 0.35
        rows.append(
            {
                "x": math.cos(angle) * radius + jitter_a,
                "y": math.sin(angle) * radius + jitter_b,
                "label": label,
                "split": str(row.split),
                "text": str(row.text),
            }
        )
    return pd.DataFrame(rows)


def build_projection(
    *,
    features_path: Path,
    dataset_path: Path,
    routes_path: Path,
    projection_output: Path,
) -> pd.DataFrame:
    routes = _routes_by_text(routes_path)
    if features_path.exists():
        data = np.load(features_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        labels = data["y"].astype(str)
        split = (
            data["split"].astype(str)
            if "split" in data
            else np.array(["unknown"] * len(labels))
        )
        text = data["text"].astype(str) if "text" in data else np.array([""] * len(labels))
        projection = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "label": labels,
                "split": split,
                "text": text,
            }
        )
    else:
        if dataset_path.exists():
            dataset = pd.read_csv(dataset_path)
        else:
            dataset = build_vectorbot_dataset(examples_per_class=10, abstain_examples=30, seed=42)
        projection = _fallback_projection(dataset)

    projection["accepted"] = [
        bool(routes[row.text]["accepted"])
        if row.text in routes
        else str(row.label) != "ABSTAIN"
        for row in projection.itertuples(index=False)
    ]
    projection_output.parent.mkdir(parents=True, exist_ok=True)
    projection.to_csv(projection_output, index=False, quoting=csv.QUOTE_MINIMAL)
    return projection


def plot_grid(path: Path) -> None:
    try:
        plt, Rectangle, _arrow = _load_matplotlib()
    except Exception:  # noqa: BLE001
        _pil_grid(path)
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("VectorBot Grid Demo", fontsize=18, pad=16)
    for y in range(5):
        for x in range(5):
            face = "#f8fafc" if (x + y) % 2 == 0 else "#eef2ff"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor="#334155", lw=1.2))
    ax.add_patch(Rectangle((3.08, 2.08), 0.84, 0.84, facecolor="#facc15", edgecolor="#854d0e"))
    ax.text(3.5, 2.5, "B", ha="center", va="center", fontsize=24, weight="bold")
    ax.text(0.05, 5.15, "state: x=3, y=2, light=ON, action=MOVE_RIGHT", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confidence(path: Path, routes_path: Path) -> None:
    try:
        plt, _rect, _arrow = _load_matplotlib()
    except Exception:  # noqa: BLE001
        _pil_confidence(path, routes_path)
        return

    routes = _routes_by_text(routes_path)
    if routes:
        first = next(iter(routes.values()))
        labels = [item["label"] for item in first.get("top_probabilities", [])]
        values = [item["probability"] for item in first.get("top_probabilities", [])]
    else:
        labels = ["MOVE_UP", "MOVE_RIGHT", "ABSTAIN"]
        values = [0.73, 0.18, 0.04]
    colors = [PALETTE.get(label, "#64748b") for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probe probability")
    ax.set_title("Top VectorBot Route Probabilities", fontsize=16)
    for index, value in enumerate(values[::-1]):
        ax.text(value + 0.02, index, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latent(path: Path, projection: pd.DataFrame) -> None:
    try:
        plt, _rect, _arrow = _load_matplotlib()
    except Exception:  # noqa: BLE001
        _pil_latent(path, projection)
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label in LABELS:
        subset = projection[projection["label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=34,
            c=PALETTE.get(label, "#64748b"),
            label=label,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title("2D Projection of VectorBot Hidden-State Features", fontsize=15)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pipeline(path: Path) -> None:
    try:
        plt, Rectangle, FancyArrowPatch = _load_matplotlib()
    except Exception:  # noqa: BLE001
        _pil_pipeline(path)
        return

    stages = [
        "command",
        "tiny model\nforward pass",
        "hidden vector\nhook",
        "linear probe\n+ gate",
        "VectorBot\naction",
    ]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    for index, stage in enumerate(stages):
        x = index * 2.0
        ax.add_patch(
            Rectangle((x, 0.55), 1.42, 0.8, facecolor="#f8fafc", edgecolor="#0f172a", lw=1.1)
        )
        ax.text(x + 0.71, 0.95, stage, ha="center", va="center", fontsize=10)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.45, 0.95),
                    (x + 1.92, 0.95),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    lw=1.2,
                    color="#334155",
                )
            )
    ax.text(
        0,
        0.22,
        "No generated commands. No JSON/tool-call parser. No keyword router.",
        fontsize=10,
    )
    ax.set_xlim(-0.1, 9.55)
    ax.set_ylim(0, 1.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build committed VectorBot visual assets.")
    parser.add_argument(
        "--features",
        default="artifacts/vectorbot_features_distilgpt2.npz",
        help="Optional feature NPZ for PCA projection.",
    )
    parser.add_argument(
        "--dataset",
        default="data/vectorbot_intents.csv",
        help="Dataset fallback for synthetic projection points.",
    )
    parser.add_argument(
        "--routes",
        default="artifacts/vectorbot_routes.jsonl",
        help="Route log used for accepted flags and confidence bars.",
    )
    parser.add_argument("--output-dir", default="docs/assets", help="PNG output directory.")
    parser.add_argument(
        "--projection-output",
        default="artifacts/vectorbot_projection.csv",
        help="Projection CSV output path.",
    )
    parser.add_argument(
        "--composite-output",
        default=None,
        help="Optional composite PNG path. Defaults to output-dir/vectorbot_demo_composite.png.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projection = build_projection(
        features_path=Path(args.features),
        dataset_path=Path(args.dataset),
        routes_path=Path(args.routes),
        projection_output=Path(args.projection_output),
    )
    plot_grid(output_dir / "vectorbot_grid_demo.png")
    plot_latent(output_dir / "vectorbot_latent_space.png", projection)
    plot_confidence(output_dir / "vectorbot_confidence_bars.png", Path(args.routes))
    plot_pipeline(output_dir / "vectorbot_pipeline_diagram.png")
    composite_path = Path(args.composite_output) if args.composite_output else (
        output_dir / "vectorbot_demo_composite.png"
    )
    plot_composite(composite_path, output_dir)
    print(f"wrote {args.projection_output}")
    print(f"wrote {output_dir / 'vectorbot_grid_demo.png'}")
    print(f"wrote {output_dir / 'vectorbot_latent_space.png'}")
    print(f"wrote {output_dir / 'vectorbot_confidence_bars.png'}")
    print(f"wrote {output_dir / 'vectorbot_pipeline_diagram.png'}")
    print(f"wrote {composite_path}")


if __name__ == "__main__":
    main()
