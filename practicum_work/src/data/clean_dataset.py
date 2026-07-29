"""Build the cleaned cat/dog semantic-segmentation dataset.

The transformation is intentionally manifest-driven. Five train masks contain
only broken contour fragments and are removed. One source image occurs twice,
once with the cat mask and once with the dog mask; the masks are merged and
the 35 overlapping pixels are marked with the MMSeg ignore label (255).
Validation and test files are copied byte-for-byte and verified by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
BAD_TRAIN_STEMS = (
    "000000028253_7169",
    "000000574769_0",
    "000000121530_5761",
    "000000275919_4499",
    "000000247301_4455",
)
MERGE_TARGET = "000000481212_908"
MERGE_SOURCE = "000000481212_908_1"
CLASS_NAMES = ("background", "cat", "dog")
IGNORE_INDEX = 255


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path, splits: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in splits:
        for kind in ("img", "labels"):
            for path in sorted((root / kind / split).glob("*")):
                result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def hash_manifest_digest(hashes: dict[str, str]) -> str:
    canonical = json.dumps(
        hashes, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_raw_dataset(root: Path) -> dict[str, int]:
    expected = {"train": 200, "val": 120, "test": 120}
    counts: dict[str, int] = {}
    for split in SPLITS:
        images = sorted((root / "img" / split).glob("*.jpg"))
        masks = sorted((root / "labels" / split).glob("*.png"))
        image_stems = {path.stem for path in images}
        mask_stems = {path.stem for path in masks}
        if image_stems != mask_stems:
            raise ValueError(
                f"{split}: image/mask mismatch: "
                f"images_only={sorted(image_stems - mask_stems)}, "
                f"masks_only={sorted(mask_stems - image_stems)}"
            )
        if len(images) != expected[split]:
            raise ValueError(
                f"{split}: expected {expected[split]} pairs, got {len(images)}"
            )
        for image_path, mask_path in zip(images, masks):
            with Image.open(image_path) as image:
                if image.size != (256, 256) or image.mode != "RGB":
                    raise ValueError(
                        f"Unexpected image format: {image_path} "
                        f"{image.size} {image.mode}"
                    )
            with Image.open(mask_path) as mask:
                values = set(np.unique(np.asarray(mask)).tolist())
                if mask.size != (256, 256) or mask.mode != "L":
                    raise ValueError(
                        f"Unexpected mask format: {mask_path} "
                        f"{mask.size} {mask.mode}"
                    )
                if not values.issubset({0, 1, 2}):
                    raise ValueError(
                        f"Unexpected labels in {mask_path}: {sorted(values)}"
                    )
        counts[split] = len(images)
    return counts


def ensure_safe_output(input_root: Path, output_root: Path) -> None:
    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == resolved_input:
        raise ValueError("Output dataset must differ from the raw dataset")
    if resolved_output in (Path(resolved_output.anchor), Path.home().resolve()):
        raise ValueError(f"Refusing unsafe output path: {resolved_output}")
    if resolved_output in resolved_input.parents:
        raise ValueError("Output must not be a parent of the input dataset")
    if "clean" not in resolved_output.name.lower():
        raise ValueError(
            "For safety, the output directory name must contain 'clean'"
        )


def count_pixels(root: Path) -> tuple[list[int], int]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ignored = 0
    for mask_path in sorted((root / "labels" / "train").glob("*.png")):
        array = np.asarray(Image.open(mask_path))
        ignored += int((array == IGNORE_INDEX).sum())
        valid = array != IGNORE_INDEX
        counts += np.bincount(
            array[valid].ravel(), minlength=len(CLASS_NAMES)
        )[: len(CLASS_NAMES)]
    return counts.tolist(), ignored


def build_clean_dataset(
    input_root: Path, output_root: Path, overwrite: bool = False
) -> dict:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    ensure_safe_output(input_root, output_root)
    raw_counts = validate_raw_dataset(input_root)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists; pass --overwrite to rebuild"
            )
        shutil.rmtree(output_root)

    val_test_hashes_before = file_hashes(input_root, ("val", "test"))
    shutil.copytree(input_root, output_root)

    removed = []
    for stem in BAD_TRAIN_STEMS:
        image_path = output_root / "img" / "train" / f"{stem}.jpg"
        mask_path = output_root / "labels" / "train" / f"{stem}.png"
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing declared bad pair: {stem}")
        foreground_pixels = int(
            (np.asarray(Image.open(mask_path)) > 0).sum()
        )
        removed.append(
            {
                "stem": stem,
                "foreground_pixels": foreground_pixels,
                "reason": "broken contour fragments",
            }
        )
        image_path.unlink()
        mask_path.unlink()

    target_image = output_root / "img" / "train" / f"{MERGE_TARGET}.jpg"
    source_image = output_root / "img" / "train" / f"{MERGE_SOURCE}.jpg"
    if sha256(target_image) != sha256(source_image):
        raise ValueError("Declared merge pair does not contain identical images")

    target_mask_path = (
        output_root / "labels" / "train" / f"{MERGE_TARGET}.png"
    )
    source_mask_path = (
        output_root / "labels" / "train" / f"{MERGE_SOURCE}.png"
    )
    target_mask = np.asarray(Image.open(target_mask_path)).copy()
    source_mask = np.asarray(Image.open(source_mask_path))
    overlap = (target_mask > 0) & (source_mask > 0)
    conflicts = overlap & (target_mask != source_mask)
    merged = np.where(source_mask > 0, source_mask, target_mask)
    merged[conflicts] = IGNORE_INDEX
    Image.fromarray(merged.astype(np.uint8), mode="L").save(target_mask_path)
    source_image.unlink()
    source_mask_path.unlink()

    if int(conflicts.sum()) != 35:
        raise ValueError(
            f"Expected 35 conflicting pixels, got {int(conflicts.sum())}"
        )

    val_test_hashes_after = file_hashes(output_root, ("val", "test"))
    if val_test_hashes_before != val_test_hashes_after:
        raise ValueError("Validation or test files changed during cleaning")

    clean_counts = {}
    class_presence: Counter[tuple[int, ...]] = Counter()
    for split in SPLITS:
        image_paths = sorted((output_root / "img" / split).glob("*.jpg"))
        clean_counts[split] = len(image_paths)
        for image_path in image_paths:
            mask = np.asarray(
                Image.open(
                    output_root
                    / "labels"
                    / split
                    / f"{image_path.stem}.png"
                )
            )
            present = tuple(
                int(value)
                for value in np.unique(mask)
                if value not in (0, IGNORE_INDEX)
            )
            class_presence[(split, *present)] += 1

    expected_clean = {"train": 194, "val": 120, "test": 120}
    if clean_counts != expected_clean:
        raise ValueError(
            f"Unexpected cleaned counts: {clean_counts}, "
            f"expected {expected_clean}"
        )

    train_pixel_counts, ignored_pixels = count_pixels(output_root)
    total_valid = sum(train_pixel_counts)
    manifest = {
        "schema_version": 1,
        "input_root": input_root.name,
        "output_root": output_root.name,
        "classes": dict(enumerate(CLASS_NAMES)),
        "ignore_index": IGNORE_INDEX,
        "raw_counts": raw_counts,
        "clean_counts": clean_counts,
        "removed": removed,
        "merge": {
            "target_stem": MERGE_TARGET,
            "source_stem": MERGE_SOURCE,
            "images_identical": True,
            "overlap_policy": "set conflicting pixels to ignore_index",
            "conflicting_pixels": int(conflicts.sum()),
        },
        "train_valid_pixel_counts": dict(
            zip(CLASS_NAMES, train_pixel_counts)
        ),
        "train_valid_pixel_percent": {
            name: round(count / total_valid * 100, 6)
            for name, count in zip(CLASS_NAMES, train_pixel_counts)
        },
        "train_ignored_pixels": ignored_pixels,
        "val_test_sha256_unchanged": True,
        "val_test_sha256_manifest": {
            "files": len(val_test_hashes_before),
            "before": hash_manifest_digest(val_test_hashes_before),
            "after": hash_manifest_digest(val_test_hashes_after),
        },
        "class_presence": {
            "|".join(map(str, key)): value
            for key, value in sorted(class_presence.items())
        },
    }
    manifest_path = output_root / "cleaning_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train_dataset_for_students"),
        help="Raw dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_dataset_cleaned"),
        help="Cleaned dataset root (name must contain 'clean')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Safely rebuild an existing cleaned output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_clean_dataset(args.input, args.output, args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
