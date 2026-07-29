"""Audit the raw/cleaned dataset and generate EDA tables and PNG figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = ("background", "cat", "dog")
CLASS_COLORS = np.asarray(
    [(35, 35, 35), (240, 70, 70), (65, 135, 245)], dtype=np.uint8
)
SPLITS = ("train", "val", "test")
KNOWN_BAD = {
    "000000028253_7169",
    "000000574769_0",
    "000000121530_5761",
    "000000275919_4499",
    "000000247301_4455",
}
MERGE_STEMS = ("000000481212_908", "000000481212_908_1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference_hash(image: Image.Image, size: int = 8) -> str:
    grayscale = image.convert("L").resize((size + 1, size))
    values = np.asarray(grayscale, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.ravel():
        number = (number << 1) | int(bit)
    return f"{number:0{size * size // 4}x}"


def components(binary: np.ndarray) -> list[int]:
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    for y, x in zip(*np.where(binary & ~visited)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for ny, nx in (
                (cy - 1, cx),
                (cy + 1, cx),
                (cy, cx - 1),
                (cy, cx + 1),
            ):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def sample_record(root: Path, split: str, image_path: Path) -> dict:
    mask_path = root / "labels" / split / f"{image_path.stem}.png"
    image = Image.open(image_path)
    mask = np.asarray(Image.open(mask_path))
    valid = mask != 255
    counts = np.bincount(mask[valid].ravel(), minlength=3)[:3]
    foreground = (mask > 0) & valid
    if foreground.any():
        ys, xs = np.where(foreground)
        bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
    else:
        bbox = (0, 0, 0, 0)
        bbox_width = bbox_height = 0
    component_sizes = components(foreground)
    return {
        "split": split,
        "stem": image_path.stem,
        "image_width": image.width,
        "image_height": image.height,
        "background_pixels": int(counts[0]),
        "cat_pixels": int(counts[1]),
        "dog_pixels": int(counts[2]),
        "ignored_pixels": int((mask == 255).sum()),
        "foreground_pixels": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "component_count": len(component_sizes),
        "largest_component_pixels": component_sizes[0]
        if component_sizes
        else 0,
        "present_classes": ",".join(
            CLASS_NAMES[index] for index in (1, 2) if counts[index] > 0
        ),
        "image_sha256": sha256(image_path),
        "image_dhash": difference_hash(image),
        "mask_sha256": sha256(mask_path),
        "quality_status": "remove_broken_mask"
        if image_path.stem in KNOWN_BAD
        else "merge_duplicate"
        if image_path.stem in MERGE_STEMS
        else "ok",
    }


def draw_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    output_path: Path,
    value_suffix: str = "",
) -> None:
    width, height = 1000, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), title, fill="black")
    max_value = max(values) or 1
    bar_left, bar_right = 210, 940
    bar_height = 55
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 80 + index * 105
        draw.text((25, y + 18), label, fill="black")
        draw.rectangle(
            (bar_left, y, bar_right, y + bar_height),
            outline=(180, 180, 180),
        )
        extent = int((bar_right - bar_left) * value / max_value)
        color = tuple(CLASS_COLORS[min(index, 2)].tolist())
        draw.rectangle(
            (bar_left, y, bar_left + extent, y + bar_height), fill=color
        )
        draw.text(
            (bar_left + 8, y + 18),
            f"{value:.4f}{value_suffix}",
            fill="white" if index != 0 else "black",
        )
    image.save(output_path)


def draw_histogram(
    values: list[float], title: str, output_path: Path, bins: int = 20
) -> None:
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), title, fill="black")
    counts, edges = np.histogram(np.asarray(values), bins=bins)
    plot_left, plot_top, plot_right, plot_bottom = 70, 70, 960, 500
    max_count = max(int(counts.max()), 1)
    bar_width = (plot_right - plot_left) / bins
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black")
    for index, count in enumerate(counts):
        x1 = int(plot_left + index * bar_width)
        x2 = int(plot_left + (index + 1) * bar_width - 2)
        y1 = int(
            plot_bottom - (plot_bottom - plot_top) * int(count) / max_count
        )
        draw.rectangle((x1, y1, x2, plot_bottom), fill=(65, 135, 245))
    draw.text(
        (plot_left, plot_bottom + 15), f"{edges[0]:.4f}", fill="black"
    )
    draw.text(
        (plot_right - 80, plot_bottom + 15),
        f"{edges[-1]:.4f}",
        fill="black",
    )
    draw.text((8, plot_top), str(max_count), fill="black")
    image.save(output_path)


def colorize(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    ignored = safe == 255
    safe[ignored] = 0
    result = CLASS_COLORS[safe]
    result[ignored] = (255, 220, 30)
    return result


def panel_for_samples(
    root: Path, split: str, stems: list[str], output_path: Path
) -> None:
    tile = 256
    row_height = tile + 24
    canvas = Image.new("RGB", (tile * 3, row_height * len(stems)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, stem in enumerate(stems):
        image = np.asarray(
            Image.open(root / "img" / split / f"{stem}.jpg").convert("RGB")
        )
        mask = np.asarray(
            Image.open(root / "labels" / split / f"{stem}.png")
        )
        colored = colorize(mask)
        overlay = image.copy()
        foreground = (mask > 0) & (mask != 255)
        overlay[foreground] = (
            image[foreground] * 0.45 + colored[foreground] * 0.55
        ).astype(np.uint8)
        ignored = mask == 255
        overlay[ignored] = colored[ignored]
        for column, array in enumerate((image, colored, overlay)):
            canvas.paste(
                Image.fromarray(array), (column * tile, row * row_height)
            )
        draw.text(
            (4, row * row_height + tile + 4),
            f"{stem} | foreground={int(foreground.sum())}",
            fill="black",
        )
    canvas.save(output_path)


def duplicate_before_after(
    raw_root: Path, clean_root: Path, output_path: Path
) -> None:
    target, source = MERGE_STEMS
    image = np.asarray(
        Image.open(raw_root / "img" / "train" / f"{target}.jpg").convert(
            "RGB"
        )
    )
    raw_target = np.asarray(
        Image.open(raw_root / "labels" / "train" / f"{target}.png")
    )
    raw_source = np.asarray(
        Image.open(raw_root / "labels" / "train" / f"{source}.png")
    )
    cleaned = np.asarray(
        Image.open(clean_root / "labels" / "train" / f"{target}.png")
    )
    arrays = [
        image,
        colorize(raw_target),
        colorize(raw_source),
        colorize(cleaned),
    ]
    canvas = Image.new("RGB", (256 * 4, 292), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("image", "cat-only raw", "dog-only raw", "merged cleaned")
    for index, (array, label) in enumerate(zip(arrays, labels)):
        canvas.paste(Image.fromarray(array), (index * 256, 0))
        draw.text((index * 256 + 4, 262), label, fill="black")
    canvas.save(output_path)


def run_audit(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    duplicate_hashes: dict[str, list[str]] = {}
    base_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        base_ids[split] = set()
        for image_path in sorted((root / "img" / split).glob("*.jpg")):
            record = sample_record(root, split, image_path)
            records.append(record)
            duplicate_hashes.setdefault(record["image_sha256"], []).append(
                f"{split}/{image_path.name}"
            )
            base_ids[split].add(image_path.stem.split("_")[0])

    csv_path = output_dir / "sample_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    pixel_counts = {
        class_name: sum(record[f"{class_name}_pixels"] for record in records)
        for class_name in CLASS_NAMES
    }
    total_pixels = sum(pixel_counts.values())
    pixel_percent = {
        name: count / total_pixels * 100
        for name, count in pixel_counts.items()
    }
    split_counts = Counter(record["split"] for record in records)
    presence_counts = Counter(
        (record["split"], record["present_classes"]) for record in records
    )
    duplicate_groups = [
        members for members in duplicate_hashes.values() if len(members) > 1
    ]
    cross_split_base_overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        cross_split_base_overlap[f"{left}-{right}"] = sorted(
            base_ids[left] & base_ids[right]
        )
    near_duplicate_pairs = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_records = [
            record for record in records if record["split"] == left
        ]
        right_records = [
            record for record in records if record["split"] == right
        ]
        for left_record in left_records:
            left_hash = int(left_record["image_dhash"], 16)
            for right_record in right_records:
                distance = (
                    left_hash ^ int(right_record["image_dhash"], 16)
                ).bit_count()
                if distance <= 5:
                    near_duplicate_pairs.append(
                        {
                            "left": f"{left}/{left_record['stem']}",
                            "right": f"{right}/{right_record['stem']}",
                            "dhash_distance": distance,
                        }
                    )

    summary = {
        "root": root.name,
        "split_counts": dict(split_counts),
        "image_sizes": sorted(
            {
                (
                    record["image_width"],
                    record["image_height"],
                )
                for record in records
            }
        ),
        "pixel_counts": pixel_counts,
        "pixel_percent": pixel_percent,
        "class_presence": {
            f"{split}|{classes}": count
            for (split, classes), count in sorted(presence_counts.items())
        },
        "foreground_fraction": {
            "min": min(record["foreground_fraction"] for record in records),
            "median": float(
                np.median(
                    [record["foreground_fraction"] for record in records]
                )
            ),
            "max": max(record["foreground_fraction"] for record in records),
        },
        "component_count": {
            "min": min(record["component_count"] for record in records),
            "median": float(
                np.median([record["component_count"] for record in records])
            ),
            "max": max(record["component_count"] for record in records),
        },
        "bbox_scale": {
            "min": min(
                math.sqrt(record["bbox_width"] * record["bbox_height"]) / 256
                for record in records
                if record["bbox_width"] > 0
            ),
            "median": float(
                np.median(
                    [
                        math.sqrt(
                            record["bbox_width"] * record["bbox_height"]
                        )
                        / 256
                        for record in records
                        if record["bbox_width"] > 0
                    ]
                )
            ),
            "max": max(
                math.sqrt(record["bbox_width"] * record["bbox_height"]) / 256
                for record in records
                if record["bbox_width"] > 0
            ),
        },
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_base_id_overlap": cross_split_base_overlap,
        "cross_split_dhash_distance_le_5": near_duplicate_pairs,
        "quality_findings": [
            {
                "stem": record["stem"],
                "status": record["quality_status"],
                "foreground_pixels": record["foreground_pixels"],
            }
            for record in records
            if record["quality_status"] != "ok"
        ],
    }
    with (output_dir / "dataset_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    draw_bar_chart(
        list(CLASS_NAMES),
        [pixel_percent[name] for name in CLASS_NAMES],
        "Pixel distribution by semantic class (all splits)",
        output_dir / "class_pixel_distribution.png",
        "%",
    )
    draw_histogram(
        [
            record["foreground_fraction"]
            for record in records
            if record["foreground_pixels"] > 0
        ],
        "Foreground area fraction per image",
        output_dir / "foreground_area_histogram.png",
    )
    draw_histogram(
        [
            math.sqrt(record["bbox_width"] * record["bbox_height"]) / 256
            for record in records
            if record["bbox_width"] > 0
        ],
        "Normalized square root of foreground bounding-box area",
        output_dir / "bbox_scale_histogram.png",
    )
    if all(
        (root / "img" / "train" / f"{stem}.jpg").exists()
        for stem in sorted(KNOWN_BAD)
    ):
        panel_for_samples(
            root,
            "train",
            sorted(KNOWN_BAD),
            output_dir / "broken_masks_before_cleaning.png",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("train_dataset_for_students"),
    )
    parser.add_argument(
        "--clean-dataset",
        type=Path,
        default=Path("train_dataset_cleaned"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("practicum_work/supplementary/viz/eda"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args.dataset, args.output)
    if args.clean_dataset.exists():
        duplicate_before_after(
            args.dataset,
            args.clean_dataset,
            args.output / "duplicate_merge_before_after.png",
        )
        cleaned_output = args.output / "cleaned"
        run_audit(args.clean_dataset, cleaned_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
