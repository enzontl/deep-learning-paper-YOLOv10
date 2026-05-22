#!/usr/bin/env python3
"""Local YOLOv10 reproduction script tuned for Apple Silicon (MPS).

The goal is not to fully reproduce the original paper at COCO scale, but to
test the core idea of YOLOv10 in a lighter setup:
- fine-tune YOLOv10n on a small deterministic fraction of Pascal VOC,
- compare it to a YOLOv8n baseline,
- use a setup closer to the paper with SGD,
- keep the training budget realistic for a Mac Mini.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics import settings as ul_settings


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "mps_runs"
DATASETS_DIR = ROOT / "data" / "datasets"


@dataclass
class ExperimentConfig:
    data: str = "VOC.yaml"
    fraction: float = 0.20
    epochs: int = 40
    imgsz: int = 416
    batch: int = 8
    lr0: float = 0.01
    lrf: float = 0.01
    warmup_epochs: float = 3.0
    optimizer: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 5e-4
    seed: int = 42
    workers: int = 0
    cache: bool = False
    patience: int = 50
    device: str = "mps"
    project: str = str(RESULTS_DIR)
    force_retrain: bool = False
    models: tuple[str, ...] = ("yolov10n.pt", "yolov8n.pt")
    tag: str = "voc20_mps_sgd"


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Reduced YOLOv10 reproduction on Pascal VOC for Apple Silicon."
    )
    parser.add_argument("--data", default="VOC.yaml", help="Ultralytics dataset yaml.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.20,
        help="Deterministic training fraction of the training split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Recommended: around 30 to 50 for this lighter reproduction.",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size.")
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. Lower to 4 if your Mac runs out of memory.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate. Matches a more paper-like SGD setup.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final LR multiplier. Final LR = lr0 * lrf with cosine decay.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        dest="warmup_epochs",
        help="Warmup for stable SGD fine-tuning.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="Momentum used by SGD.",
    )
    parser.add_argument(
        "--optimizer",
        default="SGD",
        choices=["Adam", "AdamW", "SGD", "auto"],
        help="Optimizer. Default is SGD for a more paper-like setup.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Preferred device. Default is mps; cpu is used automatically if needed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov10n.pt", "yolov8n.pt"],
        help="Weights to train. Default: YOLOv10n + YOLOv8n baseline.",
    )
    parser.add_argument(
        "--tag",
        default="voc20_mps_sgd",
        help="Experiment tag used in result folder names.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if checkpoints already exist.",
    )

    args = parser.parse_args()
    return ExperimentConfig(
        data=args.data,
        fraction=args.fraction,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        momentum=args.momentum,
        device=args.device,
        models=tuple(args.models),
        tag=args.tag,
        force_retrain=args.force_retrain,
    )


def resolve_device(preferred: str) -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cpu":
        return "cpu"
    if preferred == "mps":
        print("MPS indisponible sur cette machine, bascule vers CPU.")
        return "cpu"
    return preferred


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def configure_project_dirs(project_dir: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    ul_settings.update({"datasets_dir": str(DATASETS_DIR), "runs_dir": str(project_dir)})


def checkpoint_path(project_dir: Path, run_name: str) -> Path:
    return project_dir / run_name / "weights" / "best.pt"


def safe_float(value: Any) -> float:
    return round(float(value), 6)


def summarize_metrics(metrics: Any) -> dict[str, Any]:
    names = metrics.names
    per_class_map50_95 = {}
    for idx, ap in enumerate(metrics.box.maps):
        if isinstance(names, dict):
            class_name = names.get(idx, str(idx))
        else:
            class_name = names[idx]
        per_class_map50_95[class_name] = safe_float(ap)

    speed = {key: safe_float(val) for key, val in metrics.speed.items()}
    return {
        "map50": safe_float(metrics.box.map50),
        "map50_95": safe_float(metrics.box.map),
        "precision": safe_float(metrics.box.mp),
        "recall": safe_float(metrics.box.mr),
        "per_class_map50_95": per_class_map50_95,
        "speed_ms": speed,
    }


def train_one(weights: str, cfg: ExperimentConfig, project_dir: Path) -> Path:
    run_stem = Path(weights).stem
    run_name = f"{run_stem}_{cfg.tag}"
    best_ckpt = checkpoint_path(project_dir, run_name)

    if best_ckpt.exists() and not cfg.force_retrain:
        print(f"[skip] {run_name} existe deja: {best_ckpt}")
        return best_ckpt

    print(f"[train] {run_name} on device={cfg.device}")
    model = YOLO(weights)

    try:
        model.train(
            data=cfg.data,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            fraction=cfg.fraction,
            device=cfg.device,
            optimizer=cfg.optimizer,
            lr0=cfg.lr0,
            lrf=cfg.lrf,
            cos_lr=True,
            warmup_epochs=cfg.warmup_epochs,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            workers=cfg.workers,
            cache=cfg.cache,
            seed=cfg.seed,
            deterministic=True,
            pretrained=True,
            amp=False,
            patience=cfg.patience,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
            verbose=True,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "mps" in msg:
            raise RuntimeError(
                "Echec d'entrainement sur MPS. Essayez --batch 4 ou --imgsz 416."
            ) from exc
        raise

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint introuvable apres entrainement: {best_ckpt}")
    return best_ckpt


def evaluate_one(checkpoint: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=cfg.data,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        plots=False,
        verbose=False,
    )
    return summarize_metrics(metrics)


def save_summary(project_dir: Path, cfg: ExperimentConfig, summary: dict[str, Any]) -> Path:
    out_path = project_dir / f"summary_{cfg.tag}.json"
    payload = {"config": asdict(cfg), "results": summary}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def run_experiment(cfg: ExperimentConfig) -> Path:
    cfg.device = resolve_device(cfg.device)
    cfg.project = str(Path(cfg.project).resolve())
    project_dir = Path(cfg.project)

    set_seed(cfg.seed)
    configure_project_dirs(project_dir)

    summary: dict[str, Any] = {}
    for weights in cfg.models:
        best_ckpt = train_one(weights, cfg, project_dir)
        metrics = evaluate_one(best_ckpt, cfg)
        summary[Path(weights).stem] = {
            "checkpoint": str(best_ckpt),
            "metrics": metrics,
        }

    summary_path = save_summary(project_dir, cfg, summary)

    print("\nResume final")
    for model_name, result in summary.items():
        metrics = result["metrics"]
        print(
            f"- {model_name}: mAP50={metrics['map50']:.4f} | "
            f"mAP50-95={metrics['map50_95']:.4f} | "
            f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f}"
        )
    print(f"JSON sauvegarde dans {summary_path}")
    return summary_path


def main() -> None:
    cfg = parse_args()
    run_experiment(cfg)


if __name__ == "__main__":
    main()
