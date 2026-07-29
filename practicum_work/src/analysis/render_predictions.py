"""Render best/worst semantic-segmentation predictions as report panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = np.asarray(
    [(35, 35, 35), (240, 70, 70), (65, 135, 245)], dtype=np.uint8
)


def colorize(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    ignored = safe == 255
    safe[ignored] = 0
    result = COLORS[safe]
    result[ignored] = (255, 220, 30)
    return result


def make_panel(
    image_path: Path,
    label_path: Path,
    prediction_path: Path,
    score: float,
) -> Image.Image:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    target = np.asarray(Image.open(label_path))
    prediction = np.asarray(Image.open(prediction_path))
    target_color = colorize(target)
    prediction_color = colorize(prediction)
    overlay = image.copy()
    foreground = prediction > 0
    overlay[foreground] = (
        image[foreground] * 0.45 + prediction_color[foreground] * 0.55
    ).astype(np.uint8)
    panel = Image.new("RGB", (256 * 4, 292), "white")
    draw = ImageDraw.Draw(panel)
    labels = ("image", "ground truth", "prediction", "prediction overlay")
    for index, (array, label) in enumerate(
        zip((image, target_color, prediction_color, overlay), labels)
    ):
        panel.paste(Image.fromarray(array), (index * 256, 0))
        draw.text((index * 256 + 4, 261), label, fill="black")
    draw.text(
        (4, 278),
        f"{image_path.stem} | foreground mDice={score:.4f}",
        fill="black",
    )
    return panel


def render(
    images_dir: Path,
    labels_dir: Path,
    predictions_dir: Path,
    sample_metrics_path: Path,
    output_dir: Path,
    count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with sample_metrics_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: float(row["sample_foreground_mDice"]))
    groups = {"worst": rows[:count], "best": rows[-count:][::-1]}
    for group_name, group_rows in groups.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        for rank, row in enumerate(group_rows, start=1):
            stem = row["stem"]
            panel = make_panel(
                images_dir / f"{stem}.jpg",
                labels_dir / f"{stem}.png",
                predictions_dir / f"{stem}.png",
                float(row["sample_foreground_mDice"]),
            )
            panel.save(group_dir / f"{rank:02d}_{stem}.png")
            panels.append(panel)
        contact_sheet = Image.new(
            "RGB", (256 * 4, 292 * len(panels)), "white"
        )
        for index, panel in enumerate(panels):
            contact_sheet.paste(panel, (0, index * 292))
        contact_sheet.save(output_dir / f"{group_name}_contact_sheet.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(
        args.images,
        args.labels,
        args.predictions,
        args.sample_metrics,
        args.output,
        args.count,
    )


if __name__ == "__main__":
    main()
