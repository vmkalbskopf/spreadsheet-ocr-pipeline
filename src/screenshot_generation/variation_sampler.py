"""
Samples a single randomized UI/rendering configuration for one screenshot,
drawing from config/screenshot_variation.yaml.

The output dict is consumed by render_libreoffice.py (or a future
render_excel.py) to actually apply the settings, and is also stored
verbatim in the manifest so every screenshot's ground-truth config is
reproducible and inspectable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScreenshotConfig:
    software: str
    theme: str
    zoom_pct: int
    gridlines: bool
    frozen_rows: int
    frozen_cols: int
    column_width_mode: str
    row_height_mode: str
    font_family: str
    font_size_pt: int
    cell_highlight: bool
    n_selected_cells: int
    crop_top_pct: float
    crop_bottom_pct: float
    crop_left_pct: float
    crop_right_pct: float
    conditional_formatting: str | None
    filter_dropdowns: bool
    os_window_chrome: str
    resolution: str
    dpi_scaling: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weighted_choice(options: list[str], weights: list[float] | None) -> str:
    if weights is None:
        return random.choice(options)
    return random.choices(options, weights=weights, k=1)[0]


def _uniform(bounds: dict[str, float]) -> float:
    return random.uniform(bounds["min"], bounds["max"])


def load_variation_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sample_config(variation_cfg: dict, seed: int | None = None) -> ScreenshotConfig:
    """Draw one random screenshot configuration. Pass `seed` for reproducibility
    (e.g. seed = hash(csv_path) + variant_index) when regenerating a specific sample."""
    rng_state = random.getstate()
    if seed is not None:
        random.seed(seed)

    try:
        sw = variation_cfg["software"]
        software = _weighted_choice(list(sw["options"].keys()), list(sw["options"].values()))

        th = variation_cfg["theme"]
        theme = _weighted_choice(th["options"], th["weights"])

        zoom_pct = int(_uniform(variation_cfg["zoom_pct"]))
        gridlines = random.choice(variation_cfg["gridlines"])

        fp = variation_cfg["frozen_panes"]
        if random.random() < fp["probability"]:
            frozen_rows = random.randint(0, fp["max_frozen_rows"])
            frozen_cols = random.randint(0, fp["max_frozen_cols"])
        else:
            frozen_rows = frozen_cols = 0

        cw = variation_cfg["column_width"]["mode_weights"]
        column_width_mode = _weighted_choice(list(cw.keys()), list(cw.values()))

        rh = variation_cfg["row_height"]["mode_weights"]
        row_height_mode = _weighted_choice(list(rh.keys()), list(rh.values()))

        font_family = random.choice(variation_cfg["font"]["families"])
        font_size_pt = random.randint(
            variation_cfg["font"]["size_pt"]["min"], variation_cfg["font"]["size_pt"]["max"]
        )

        ch = variation_cfg["cell_highlight"]
        cell_highlight = random.random() < ch["probability"]
        n_selected_cells = random.randint(1, ch["max_selected_cells"]) if cell_highlight else 0

        sc = variation_cfg["scroll_crop"]
        if random.random() < sc["probability"]:
            crop_top = _uniform(sc["crop_top_pct"])
            crop_bottom = _uniform(sc["crop_bottom_pct"])
            crop_left = _uniform(sc["crop_left_pct"])
            crop_right = _uniform(sc["crop_right_pct"])
        else:
            crop_top = crop_bottom = crop_left = crop_right = 0.0

        cf = variation_cfg["conditional_formatting"]
        conditional_formatting = (
            random.choice(cf["types"]) if random.random() < cf["probability"] else None
        )

        filter_dropdowns = random.random() < variation_cfg["filter_dropdowns"]["probability"]

        owc = variation_cfg["os_window_chrome"]
        os_window_chrome = _weighted_choice(owc["options"], owc["weights"])

        resolution = random.choice(variation_cfg["display"]["resolution_options"])
        dpi_scaling = random.choice(variation_cfg["display"]["dpi_scaling_options"])

        return ScreenshotConfig(
            software=software,
            theme=theme,
            zoom_pct=zoom_pct,
            gridlines=gridlines,
            frozen_rows=frozen_rows,
            frozen_cols=frozen_cols,
            column_width_mode=column_width_mode,
            row_height_mode=row_height_mode,
            font_family=font_family,
            font_size_pt=font_size_pt,
            cell_highlight=cell_highlight,
            n_selected_cells=n_selected_cells,
            crop_top_pct=crop_top,
            crop_bottom_pct=crop_bottom,
            crop_left_pct=crop_left,
            crop_right_pct=crop_right,
            conditional_formatting=conditional_formatting,
            filter_dropdowns=filter_dropdowns,
            os_window_chrome=os_window_chrome,
            resolution=resolution,
            dpi_scaling=dpi_scaling,
        )
    finally:
        if seed is not None:
            random.setstate(rng_state)


if __name__ == "__main__":
    cfg = load_variation_config("config/screenshot_variation.yaml")
    for i in range(3):
        print(sample_config(cfg, seed=i).to_dict())
