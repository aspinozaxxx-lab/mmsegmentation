"""Render examples produced by the exact strong MMSeg augmentation pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from PIL import Image, ImageDraw

from mmseg.utils import register_all_modules


COLORS = np.asarray(
    [(35, 35, 35), (240, 70, 70), (65, 135, 245)], dtype=np.uint8
)


def colorize(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    ignored = safe == 255
    safe[ignored] = 0
    output = COLORS[safe]
    output[ignored] = (255, 220, 30)
    return output


def render(
    dataset_root: Path,
    config_path: Path,
    output_path: Path,
    stems: list[str],
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    pipeline = Compose(
        [
            transform
            for transform in cfg.train_pipeline
            if transform["type"] != "PackSegInputs"
        ]
    )
    row_height = 286
    canvas = Image.new("RGB", (256 * 4, row_height * len(stems)), "white")
    draw = ImageDraw.Draw(canvas)
    headings = ("original", "original mask", "augmented", "augmented mask")
    for row, stem in enumerate(stems):
        image_path = dataset_root / "img" / "train" / f"{stem}.jpg"
        mask_path = dataset_root / "labels" / "train" / f"{stem}.png"
        original = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        result = pipeline(
            {
                "img_path": str(image_path),
                "seg_map_path": str(mask_path),
                "reduce_zero_label": False,
                "seg_fields": [],
            }
        )
        arrays = (
            original,
            colorize(mask),
            result["img"].astype(np.uint8),
            colorize(result["gt_seg_map"]),
        )
        for column, (array, heading) in enumerate(zip(arrays, headings)):
            canvas.paste(
                Image.fromarray(array),
                (column * 256, row * row_height),
            )
            draw.text(
                (column * 256 + 4, row * row_height + 260),
                heading,
                fill="black",
            )
        draw.text(
            (4, row * row_height + 274),
            f"{stem} | seed={seed}",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("train_dataset_cleaned"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("practicum_work/configs/exp02_mitb0_strong_aug.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "practicum_work/supplementary/viz/augmentations/"
            "strong_augmentation_examples.png"
        ),
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=[
            "000000229631_480",
            "000000312712_4537",
            "000000481212_908",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(
        args.dataset,
        args.config,
        args.output,
        args.stems,
        args.seed,
    )


if __name__ == "__main__":
    main()
