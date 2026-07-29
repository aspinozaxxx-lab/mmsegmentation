"""Plot local loss and validation Dice curves from MMEngine JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No scalar records in {path}")
    return records


def plot(scalars_path: Path, output_path: Path, title: str) -> None:
    records = load_records(scalars_path)
    training = [record for record in records if "loss" in record]
    validation = [record for record in records if "mDice" in record]
    if not training or not validation:
        raise ValueError("Both training loss and validation mDice are required")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        [record["step"] for record in training],
        [record["loss"] for record in training],
        color="#3d7dd8",
        linewidth=1.5,
    )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    series = (
        ("mDice", "mDice", "#222222"),
        ("foreground_mDice", "foreground mDice", "#d94a4a"),
        ("Dice/cat", "cat Dice", "#f08a24"),
        ("Dice/dog", "dog Dice", "#3d7dd8"),
    )
    for key, label, color in series:
        selected = [record for record in validation if key in record]
        if selected:
            axes[1].plot(
                [record["step"] for record in selected],
                [record[key] / 100 for record in selected],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label,
                color=color,
            )
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Dice, 0–1")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.scalars, args.output, args.title)


if __name__ == "__main__":
    main()
