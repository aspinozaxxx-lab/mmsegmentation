"""Attach provenance and independently computed metrics to a ClearML task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearml import Task
from mmengine.config import Config


def parse_artifact(specification: str) -> tuple[str, Path]:
    try:
        name, path = specification.split("=", maxsplit=1)
    except ValueError as error:
        raise ValueError("Artifact must use NAME=PATH format") from error
    return name, Path(path)


def publish(
    task_id: str,
    git_sha: str,
    config_path: Path,
    split: str,
    metrics_path: Path,
    artifacts: list[tuple[str, Path]],
    mark_completed: bool,
) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    task = Task.get_task(task_id=task_id)
    was_completed = str(task.get_status()).lower().endswith("completed")
    if was_completed:
        task.mark_started(force=True)
    task.set_parameter("provenance/git_sha", git_sha)
    task.set_parameter("provenance/config", config_path.as_posix())
    task.set_parameter("evaluation/selection_split", split)
    task.connect_configuration(
        Config.fromfile(str(config_path)).to_dict(),
        name="resolved_config_postrun",
    )

    logger = task.get_logger()
    iteration = int(metrics.get("checkpoint_iteration", 0))
    logger.report_scalar(
        title=f"{split}/aggregate",
        series="mDice",
        value=float(metrics["mDice"]) * 100,
        iteration=iteration,
    )
    logger.report_scalar(
        title=f"{split}/aggregate",
        series="foreground_mDice",
        value=float(metrics["foreground_mDice"]) * 100,
        iteration=iteration,
    )
    for class_name, value in metrics["class_Dice"].items():
        logger.report_scalar(
            title=f"{split}/Dice",
            series=class_name,
            value=float(value) * 100,
            iteration=iteration,
        )

    task.upload_artifact(
        name=f"{split}_independent_metrics",
        artifact_object=metrics_path,
        wait_on_upload=True,
    )
    for name, path in artifacts:
        if not path.exists():
            raise FileNotFoundError(path)
        task.upload_artifact(
            name=name,
            artifact_object=path,
            wait_on_upload=True,
        )
    task.flush(wait_for_uploads=False)
    if mark_completed or was_completed:
        task.mark_completed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="NAME=PATH; repeat for additional CSV/PNG artifacts",
    )
    parser.add_argument("--mark-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publish(
        task_id=args.task_id,
        git_sha=args.git_sha,
        config_path=args.config,
        split=args.split,
        metrics_path=args.metrics,
        artifacts=[parse_artifact(spec) for spec in args.artifact],
        mark_completed=args.mark_completed,
    )
    print(
        json.dumps(
            {"task_id": args.task_id, "published": True},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
