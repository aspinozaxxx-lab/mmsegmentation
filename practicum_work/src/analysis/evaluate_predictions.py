"""Independently evaluate raw PNG segmentation predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = ("background", "cat", "dog")
IGNORE_INDEX = 255


def confusion_for_pair(
    target: np.ndarray, prediction: np.ndarray, num_classes: int = 3
) -> np.ndarray:
    valid = target != IGNORE_INDEX
    target = target[valid].astype(np.int64)
    prediction = prediction[valid].astype(np.int64)
    if prediction.size and (
        prediction.min() < 0 or prediction.max() >= num_classes
    ):
        raise ValueError("Prediction contains labels outside [0, 2]")
    encoded = target * num_classes + prediction
    return np.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    true_positive = np.diag(confusion).astype(np.float64)
    gt_pixels = confusion.sum(axis=1).astype(np.float64)
    pred_pixels = confusion.sum(axis=0).astype(np.float64)
    union = gt_pixels + pred_pixels - true_positive
    dice_denominator = gt_pixels + pred_pixels
    dice = np.divide(
        2 * true_positive,
        dice_denominator,
        out=np.full_like(true_positive, np.nan),
        where=dice_denominator > 0,
    )
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    accuracy = np.divide(
        true_positive,
        gt_pixels,
        out=np.full_like(true_positive, np.nan),
        where=gt_pixels > 0,
    )
    return {
        "mDice": float(np.nanmean(dice)),
        "mIoU": float(np.nanmean(iou)),
        "aAcc": float(true_positive.sum() / gt_pixels.sum()),
        "foreground_mDice": float(np.nanmean(dice[1:])),
        "worst_foreground_Dice": float(np.nanmin(dice[1:])),
        "class_Dice": {
            name: float(value) for name, value in zip(CLASS_NAMES, dice)
        },
        "class_IoU": {
            name: float(value) for name, value in zip(CLASS_NAMES, iou)
        },
        "class_accuracy": {
            name: float(value)
            for name, value in zip(CLASS_NAMES, accuracy)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def per_sample_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[float, dict[str, float]]:
    class_scores: dict[str, float] = {}
    selected = []
    for class_index, class_name in enumerate(CLASS_NAMES[1:], start=1):
        gt_class = (target == class_index) & (target != IGNORE_INDEX)
        pred_class = prediction == class_index
        denominator = int(gt_class.sum() + pred_class.sum())
        if denominator == 0:
            continue
        score = 2 * int((gt_class & pred_class).sum()) / denominator
        class_scores[class_name] = score
        selected.append(score)
    return (float(np.mean(selected)) if selected else 1.0), class_scores


def draw_confusion_matrix(
    confusion: np.ndarray, output_path: Path
) -> None:
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    cell = 180
    margin = 180
    size = margin + cell * len(CLASS_NAMES)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Rows: ground truth | Columns: prediction", fill="black")
    for index, name in enumerate(CLASS_NAMES):
        draw.text((margin + index * cell + 20, 65), name, fill="black")
        draw.text((20, margin + index * cell + 65), name, fill="black")
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = float(normalized[row, column])
            shade = int(255 - 180 * value)
            x1 = margin + column * cell
            y1 = margin + row * cell
            draw.rectangle(
                (x1, y1, x1 + cell, y1 + cell),
                fill=(shade, shade, 255),
                outline="black",
            )
            draw.text(
                (x1 + 35, y1 + 70),
                f"{value:.4f}\n({int(confusion[row, column])})",
                fill="black",
            )
    image.save(output_path)


def evaluate(
    labels_dir: Path, predictions_dir: Path, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(labels_dir.glob("*.png"))
    if not label_paths:
        raise FileNotFoundError(f"No PNG labels found in {labels_dir}")
    confusion = np.zeros((3, 3), dtype=np.int64)
    sample_rows = []
    for label_path in label_paths:
        prediction_path = predictions_dir / label_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction: {prediction_path}")
        target = np.asarray(Image.open(label_path))
        prediction = np.asarray(Image.open(prediction_path))
        if target.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch for {label_path.stem}: "
                f"{target.shape} vs {prediction.shape}"
            )
        pair_confusion = confusion_for_pair(target, prediction)
        confusion += pair_confusion
        sample_score, class_scores = per_sample_metrics(target, prediction)
        pair_metrics = metrics_from_confusion(pair_confusion)
        sample_rows.append(
            {
                "stem": label_path.stem,
                "sample_foreground_mDice": sample_score,
                "cat_Dice": class_scores.get("cat", ""),
                "dog_Dice": class_scores.get("dog", ""),
                "sample_mDice_all_classes": pair_metrics["mDice"],
                "foreground_pixels": int(
                    ((target > 0) & (target != IGNORE_INDEX)).sum()
                ),
            }
        )

    metrics = metrics_from_confusion(confusion)
    metrics["samples"] = len(sample_rows)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (output_dir / "sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(sample_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sample_rows)
    draw_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.labels, args.predictions, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
