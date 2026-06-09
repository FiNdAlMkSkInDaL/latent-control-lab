from pathlib import Path

from scripts.build_vectorbot_visuals import (
    build_projection,
    plot_confidence,
    plot_grid,
    plot_latent,
)
from scripts.generate_vectorbot_dataset import build_vectorbot_dataset


def test_vectorbot_visual_asset_generation_smoke(tmp_path) -> None:
    dataset = build_vectorbot_dataset(examples_per_class=4, abstain_examples=10, seed=5)
    dataset_path = tmp_path / "vectorbot_intents.csv"
    dataset.to_csv(dataset_path, index=False)
    projection_path = tmp_path / "projection.csv"
    projection = build_projection(
        features_path=tmp_path / "missing_features.npz",
        dataset_path=dataset_path,
        routes_path=tmp_path / "missing_routes.jsonl",
        projection_output=projection_path,
    )
    grid_path = tmp_path / "grid.png"
    latent_path = tmp_path / "latent.png"
    bars_path = tmp_path / "bars.png"
    plot_grid(grid_path)
    plot_latent(latent_path, projection)
    plot_confidence(bars_path, tmp_path / "missing_routes.jsonl")
    assert projection_path.exists()
    assert Path(grid_path).stat().st_size > 0
    assert Path(latent_path).stat().st_size > 0
    assert Path(bars_path).stat().st_size > 0
