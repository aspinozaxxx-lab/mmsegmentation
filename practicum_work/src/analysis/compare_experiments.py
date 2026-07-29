"""Compare validation metrics and select a model without consulting test data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

TIE_THRESHOLD = 0.001


@dataclass(frozen=True)
class Experiment:
    name: str
    config: str
    metrics_path: str
    mdice: float
    foreground_mdice: float
    worst_foreground_dice: float
    background_dice: float
    cat_dice: float
    dog_dice: float


def load_experiment(specification: str) -> Experiment:
    """Load ``NAME:CONFIG:METRICS_JSON`` into a normalized record."""
    try:
        name, config, metrics_name = specification.split(":", maxsplit=2)
    except ValueError as error:
        raise ValueError(
            "Experiment must use NAME:CONFIG:METRICS_JSON format"
        ) from error
    metrics_path = Path(metrics_name)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    class_dice = metrics["class_Dice"]
    return Experiment(
        name=name,
        config=config,
        metrics_path=metrics_path.as_posix(),
        mdice=float(metrics["mDice"]),
        foreground_mdice=float(metrics["foreground_mDice"]),
        worst_foreground_dice=float(metrics["worst_foreground_Dice"]),
        background_dice=float(class_dice["background"]),
        cat_dice=float(class_dice["cat"]),
        dog_dice=float(class_dice["dog"]),
    )


def rank_experiments(experiments: list[Experiment]) -> list[Experiment]:
    """Apply the documented validation-only tie-breaking rule."""
    if not experiments:
        raise ValueError("At least one experiment is required")
    ordered = sorted(experiments, key=lambda item: item.mdice, reverse=True)
    best_mdice = ordered[0].mdice
    tied = [
        item
        for item in ordered
        if best_mdice - item.mdice < TIE_THRESHOLD
    ]
    tied_names = {item.name for item in tied}
    tied.sort(
        key=lambda item: (
            item.foreground_mdice,
            item.worst_foreground_dice,
            item.mdice,
        ),
        reverse=True,
    )
    remainder = [item for item in ordered if item.name not in tied_names]
    return tied + remainder


def save_outputs(experiments: list[Experiment], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_experiments(experiments)
    rows = []
    for rank, experiment in enumerate(ranked, start=1):
        row = {"rank": rank, **experiment.__dict__}
        rows.append(row)
    with (output_dir / "validation_experiments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "selection_split": "validation",
        "tie_threshold": TIE_THRESHOLD,
        "selected": ranked[0].name,
        "experiments": rows,
    }
    (output_dir / "validation_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [item.name for item in ranked]
    positions = range(len(ranked))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        [position - 0.25 for position in positions],
        [item.mdice for item in ranked],
        width=0.25,
        label="mDice",
    )
    axis.bar(
        positions,
        [item.foreground_mdice for item in ranked],
        width=0.25,
        label="foreground mDice",
    )
    axis.bar(
        [position + 0.25 for position in positions],
        [item.worst_foreground_dice for item in ranked],
        width=0.25,
        label="worst foreground Dice",
    )
    axis.set_xticks(list(positions), labels, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Dice, 0–1")
    axis.set_title("Validation-only model comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_experiments.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="NAME:CONFIG:METRICS_JSON; repeat once per experiment",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = [load_experiment(spec) for spec in args.experiment]
    save_outputs(experiments, args.output)
    print(
        json.dumps(
            {
                "selected": rank_experiments(experiments)[0].name,
                "count": len(experiments),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
