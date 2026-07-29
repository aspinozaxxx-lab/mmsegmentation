"""Run MMSeg validation/test and dump raw label PNGs plus runner metrics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--disable-clearml",
        action="store_true",
        help="Keep auxiliary validation analysis out of ClearML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    if args.split == "val":
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
        cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model
    if args.disable_clearml:
        cfg.visualizer.vis_backends = [
            backend
            for backend in cfg.visualizer.vis_backends
            if backend["type"] != "ClearMLVisBackend"
        ]
    args.predictions.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.test_evaluator.output_dir = str(args.predictions)
    cfg.load_from = str(args.checkpoint)
    cfg.work_dir = str(args.work_dir)
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    (args.work_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
