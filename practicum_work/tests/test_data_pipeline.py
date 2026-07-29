"""Regression tests for the real Sprint 6 dataset cleaning contract."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from practicum_work.src.analysis.compare_experiments import (
    Experiment,
    rank_experiments,
)
from practicum_work.src.analysis.evaluate_predictions import (
    confusion_for_pair,
    metrics_from_confusion,
)
from practicum_work.src.data.clean_dataset import (
    BAD_TRAIN_STEMS,
    MERGE_SOURCE,
    MERGE_TARGET,
    build_clean_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "train_dataset_for_students"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DataCleaningContractTest(unittest.TestCase):
    @unittest.skipUnless(RAW_ROOT.exists(), "raw student dataset is absent")
    def test_cleaning_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sprint6_clean_") as directory:
            output = Path(directory) / "dataset_cleaned"
            manifest = build_clean_dataset(RAW_ROOT, output)
            self.assertEqual(
                manifest["clean_counts"],
                {"train": 194, "val": 120, "test": 120},
            )
            self.assertEqual(
                manifest["merge"]["conflicting_pixels"],
                35,
            )
            for stem in BAD_TRAIN_STEMS:
                self.assertFalse(
                    (output / "img" / "train" / f"{stem}.jpg").exists()
                )
                self.assertFalse(
                    (output / "labels" / "train" / f"{stem}.png").exists()
                )
            self.assertTrue(
                (output / "img" / "train" / f"{MERGE_TARGET}.jpg").exists()
            )
            self.assertFalse(
                (output / "img" / "train" / f"{MERGE_SOURCE}.jpg").exists()
            )
            merged = np.asarray(
                Image.open(
                    output
                    / "labels"
                    / "train"
                    / f"{MERGE_TARGET}.png"
                )
            )
            self.assertEqual(int((merged == 255).sum()), 35)
            for split in ("val", "test"):
                for kind in ("img", "labels"):
                    for raw_path in (RAW_ROOT / kind / split).glob("*"):
                        clean_path = output / kind / split / raw_path.name
                        self.assertEqual(sha256(raw_path), sha256(clean_path))


class MetricContractTest(unittest.TestCase):
    def test_confusion_and_dice(self) -> None:
        target = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        prediction = np.array([[0, 2], [2, 1]], dtype=np.uint8)
        confusion = confusion_for_pair(target, prediction)
        np.testing.assert_array_equal(
            confusion,
            np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]]),
        )
        metrics = metrics_from_confusion(confusion)
        self.assertAlmostEqual(metrics["class_Dice"]["background"], 1.0)
        self.assertAlmostEqual(metrics["class_Dice"]["cat"], 0.0)
        self.assertAlmostEqual(metrics["class_Dice"]["dog"], 2 / 3)
        self.assertAlmostEqual(metrics["mDice"], 5 / 9)

    def test_validation_tie_break_uses_foreground(self) -> None:
        base = dict(
            config="config.py",
            metrics_path="metrics.json",
            background_dice=0.99,
            cat_dice=0.75,
            dog_dice=0.75,
        )
        experiments = [
            Experiment(
                name="highest_raw_mdice",
                mdice=0.8000,
                foreground_mdice=0.7500,
                worst_foreground_dice=0.7400,
                **base,
            ),
            Experiment(
                name="within_tie_and_better_foreground",
                mdice=0.7995,
                foreground_mdice=0.7800,
                worst_foreground_dice=0.7600,
                **base,
            ),
        ]
        self.assertEqual(
            rank_experiments(experiments)[0].name,
            "within_tie_and_better_foreground",
        )


if __name__ == "__main__":
    unittest.main()
