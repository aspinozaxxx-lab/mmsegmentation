"""Validate the isolated CUDA/MMSeg/ClearML training environment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from clearml import Task
from mmengine.config import Config
from mmengine.runner import Runner

import mmcv
import mmengine
import mmseg


def parse_all_configs(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No configs matched {pattern}")
    for path in paths:
        Config.fromfile(path)
    return paths


def run_single_update(config_path: Path, work_dir: Path) -> None:
    cfg = Config.fromfile(str(config_path))
    cfg.work_dir = str(work_dir)
    cfg.train_cfg.max_iters = 1
    cfg.train_cfg.val_interval = 2
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.default_hooks.checkpoint.interval = 2
    cfg.default_hooks.checkpoint.save_best = None
    cfg.custom_hooks = []
    cfg.visualizer.vis_backends = [
        backend
        for backend in cfg.visualizer.vis_backends
        if backend["type"] != "ClearMLVisBackend"
    ]
    runner = Runner.from_cfg(cfg)
    runner.train()
    non_finite = [
        name
        for name, parameter in runner.model.named_parameters()
        if parameter.is_floating_point()
        and not bool(torch.isfinite(parameter).all())
    ]
    if non_finite:
        raise FloatingPointError(
            f"Non-finite model parameters after update: {non_finite[:5]}"
        )


def create_clearml_smoke_task(git_sha: str) -> str:
    task = Task.init(
        project_name="Practicum/Sprint6-mmsegmentation",
        task_name="00_smoke_test",
        task_type=Task.TaskTypes.testing,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
    )
    task.connect(
        {
            "git_sha": git_sha,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(0))
            ),
        },
        name="preflight",
    )
    task.get_logger().report_scalar(
        title="smoke",
        series="forward_backward_ok",
        value=1,
        iteration=0,
    )
    task_id = task.id
    task.close()
    return task_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("practicum_work/configs/baseline_segformer_mitb0.py"),
    )
    parser.add_argument(
        "--config-pattern",
        default="practicum_work/configs/*.py",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_dirs/practicum/00_smoke_test"),
    )
    parser.add_argument("--git-sha", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the training container")
    capability = torch.cuda.get_device_capability(0)
    if capability != (12, 0):
        raise RuntimeError(f"Expected RTX 5090 capability (12, 0), got {capability}")
    configs = parse_all_configs(args.config_pattern)
    run_single_update(args.config, args.work_dir)
    task_id = create_clearml_smoke_task(args.git_sha)
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": capability,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmseg": mmseg.__version__,
        "parsed_configs": configs,
        "forward_backward": "ok",
        "finite_parameters": True,
        "clearml_task_id": task_id,
        "git_sha": args.git_sha,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
