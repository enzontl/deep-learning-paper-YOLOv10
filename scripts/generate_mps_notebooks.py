#!/usr/bin/env python3
"""Generate local Jupyter notebooks for the YOLOv10 MPS workflow."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


TRAIN_NOTEBOOK = notebook(
    [
        md_cell(
            """
            # 01. Train YOLOv10 sur Mac (`mps`) [v2 - regime principal SGD]

            Notebook d'entraînement local pour une reproduction réduite de **YOLOv10**
            sur **Pascal VOC** avec :

            - `mps` au lieu de `cuda`
            - un régime principal unique
            - plus d'epochs
            - `SGD` plus proche du papier

            Exécuter ce notebook avec **Run All**. Il sauvegarde pour chaque run :

            - `best.pt`
            - `last.pt`
            - `results.csv`
            - un registre JSON partagé avec le notebook de visualisation
            """
        ),
        md_cell(
            """
            ## Dépendances

            Si besoin, installer une seule fois les dépendances du projet :

            ```python
            # import sys, subprocess
            # subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            ```
            """
        ),
        code_cell(
            """
            import os
            import json
            import random
            from datetime import datetime
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import torch
            from ultralytics import YOLO
            from ultralytics import settings as ul_settings

            def find_root() -> Path:
                candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent]
                for candidate in candidates:
                    if (candidate / "requirements.txt").exists() and (candidate / "notebooks").exists():
                        return candidate
                raise FileNotFoundError("Impossible de trouver la racine du projet.")

            ROOT = find_root()
            os.chdir(ROOT)

            RESULTS = ROOT / "results" / "mps_runs"
            ANALYSIS = RESULTS / "analysis"
            DATASETS_DIR = ROOT / "data" / "datasets"
            REGISTRY_PATH = RESULTS / "experiment_registry.json"

            for path in [RESULTS, ANALYSIS, DATASETS_DIR]:
                path.mkdir(parents=True, exist_ok=True)

            DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
            ul_settings.update({"datasets_dir": str(DATASETS_DIR), "runs_dir": str(RESULTS)})

            SEED = 42
            DATA_YAML = "VOC.yaml"
            IMGSZ = 416
            BATCH = 8
            LR0 = 0.01
            LRF = 0.01
            WARMUP_EPOCHS = 3.0
            MOMENTUM = 0.937
            WEIGHT_DECAY = 5e-4
            OPTIMIZER = "SGD"
            WORKERS = 0
            RUN_TAG = "mps_sgd_main"

            REGIMES = {
                "main": {"epochs": 40, "fraction": 0.20},
            }

            MODELS = {
                "v10": "yolov10n.pt",
                "v8": "yolov8n.pt",
            }

            print({
                "root": str(ROOT),
                "device": DEVICE,
                "datasets_dir": str(DATASETS_DIR),
                "results_dir": str(RESULTS),
                "regimes": REGIMES,
            })
            """
        ),
        code_cell(
            """
            def set_seed(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                    torch.mps.manual_seed(seed)

            def save_json(obj, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

            def load_json(path: Path):
                return json.loads(path.read_text(encoding="utf-8"))

            def load_train_csv(path: Path):
                if not path.exists():
                    return None
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                return df

            def latest_train_metrics(csv_path: Path) -> dict:
                df = load_train_csv(csv_path)
                if df is None or df.empty:
                    return {}
                row = df.iloc[-1].to_dict()
                keep = {}
                for key, value in row.items():
                    if any(tag in key for tag in ["mAP50", "mAP50-95", "precision", "recall", "epoch"]):
                        try:
                            keep[key] = float(value)
                        except Exception:
                            keep[key] = value
                return keep

            def train_if_needed(model_name: str, regime_name: str, regime_cfg: dict) -> dict:
                run_name = f"{Path(model_name).stem}_{regime_name}_{RUN_TAG}"
                run_dir = RESULTS / run_name
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                csv_path = run_dir / "results.csv"

                if best.exists() and last.exists():
                    print(f"[skip] {run_name}")
                else:
                    print(f"[train] {run_name} on {DEVICE}")
                    set_seed(SEED)
                    model = YOLO(model_name)
                    model.train(
                        data=DATA_YAML,
                        epochs=regime_cfg["epochs"],
                        fraction=regime_cfg["fraction"],
                        imgsz=IMGSZ,
                        batch=BATCH,
                        device=DEVICE,
                        optimizer=OPTIMIZER,
                        lr0=LR0,
                        lrf=LRF,
                        cos_lr=True,
                        warmup_epochs=WARMUP_EPOCHS,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY,
                        workers=WORKERS,
                        cache=False,
                        amp=False,
                        pretrained=True,
                        seed=SEED,
                        deterministic=True,
                        patience=100,
                        project=str(RESULTS),
                        name=run_name,
                        exist_ok=True,
                        save=True,
                        plots=True,
                        verbose=False,
                    )

                if not best.exists() or not last.exists():
                    raise FileNotFoundError(f"Run incomplet pour {run_name}.")

                return {
                    "run_name": run_name,
                    "model_key": "v10" if "yolov10" in model_name else "v8",
                    "source_weights": model_name,
                    "regime": regime_name,
                    "epochs": regime_cfg["epochs"],
                    "fraction": regime_cfg["fraction"],
                    "device": DEVICE,
                    "optimizer": OPTIMIZER,
                    "lr0": LR0,
                    "lrf": LRF,
                    "imgsz": IMGSZ,
                    "batch": BATCH,
                    "best": str(best),
                    "last": str(last),
                    "results_csv": str(csv_path),
                    "metrics_from_csv": latest_train_metrics(csv_path),
                }
            """
        ),
        code_cell(
            """
            registry = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "root": str(ROOT),
                "results_dir": str(RESULTS),
                "analysis_dir": str(ANALYSIS),
                "datasets_dir": str(DATASETS_DIR),
                "data_yaml": DATA_YAML,
                "device": DEVICE,
                "seed": SEED,
                "config": {
                    "imgsz": IMGSZ,
                    "batch": BATCH,
                    "lr0": LR0,
                    "lrf": LRF,
                    "warmup_epochs": WARMUP_EPOCHS,
                    "momentum": MOMENTUM,
                    "weight_decay": WEIGHT_DECAY,
                    "optimizer": OPTIMIZER,
                    "workers": WORKERS,
                    "run_tag": RUN_TAG,
                },
                "regimes": REGIMES,
                "runs": {},
            }

            if REGISTRY_PATH.exists():
                registry.update(load_json(REGISTRY_PATH))

            for regime_name, regime_cfg in REGIMES.items():
                for model_key, model_name in MODELS.items():
                    run_key = f"{model_key}_{regime_name}"
                    registry["runs"][run_key] = train_if_needed(model_name, regime_name, regime_cfg)
                    save_json(registry, REGISTRY_PATH)

            print(f"Registre sauvegardé dans: {REGISTRY_PATH}")
            """
        ),
        code_cell(
            """
            rows = []
            for run_key, info in registry["runs"].items():
                rows.append(
                    {
                        "run_key": run_key,
                        "run_name": info["run_name"],
                        "model": info["model_key"],
                        "regime": info["regime"],
                        "epochs": info["epochs"],
                        "fraction": info["fraction"],
                        "best_exists": Path(info["best"]).exists(),
                        "last_exists": Path(info["last"]).exists(),
                        "best": info["best"],
                        "last": info["last"],
                    }
                )

            summary_df = pd.DataFrame(rows).sort_values(["regime", "model"]).reset_index(drop=True)
            summary_df
            """
        ),
        md_cell(
            """
            ## Suite

            Quand l'entraînement est terminé, ouvrir :

            - `notebooks/02_viz_compare_yolov10_mps.ipynb`

            Ce second notebook recharge automatiquement le registre, les checkpoints
            `best` ou `last`, et génère les comparaisons et visualisations.
            """
        ),
    ]
)


VIZ_NOTEBOOK = notebook(
    [
        md_cell(
            """
            # 02. Visualisations et comparaisons YOLOv10 vs YOLOv8 [v2]

            Ce notebook recharge les runs entraînés dans `01_train_yolov10_mps.ipynb`
            et produit :

            - un tableau de comparaison final
            - une comparaison de taille des modèles
            - les courbes d'apprentissage
            - une ablation simple `end2end` vs `NMS forcée`
            - la distribution du nombre de boîtes
            - la latence
            - une grille qualitative de prédictions
            """
        ),
        code_cell(
            """
            import os
            import json
            import time
            import shutil
            from pathlib import Path

            import cv2
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import torch
            from ultralytics import YOLO
            from ultralytics import settings as ul_settings

            plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

            def find_root() -> Path:
                candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent]
                for candidate in candidates:
                    if (candidate / "requirements.txt").exists() and (candidate / "notebooks").exists():
                        return candidate
                raise FileNotFoundError("Impossible de trouver la racine du projet.")

            ROOT = find_root()
            os.chdir(ROOT)

            RESULTS = ROOT / "results" / "mps_runs"
            ANALYSIS = RESULTS / "analysis"
            FIGS = ANALYSIS / "figures"
            DATASETS_DIR = ROOT / "data" / "datasets"
            REGISTRY_PATH = RESULTS / "experiment_registry.json"

            for path in [RESULTS, ANALYSIS, FIGS, DATASETS_DIR]:
                path.mkdir(parents=True, exist_ok=True)

            DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
            ul_settings.update({"datasets_dir": str(DATASETS_DIR), "runs_dir": str(RESULTS)})

            WEIGHT_KIND = "best"  # passer à "last" pour comparer les derniers checkpoints

            if not REGISTRY_PATH.exists():
                raise FileNotFoundError(
                    "Le registre d'expérience est introuvable. Lance d'abord le notebook d'entraînement."
                )

            registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            REGIME_NAMES = list(registry["regimes"].keys())
            print({"device": DEVICE, "weight_kind": WEIGHT_KIND, "registry": str(REGISTRY_PATH)})

            OFFICIAL_BENCH = {
                "YOLOv8n": {
                    "params_m": 3.2,
                    "flops_g": 8.7,
                    "ap_val_ref_pct": 37.3,
                    "latency_ref_ms": 6.16,
                    "latencyf_ref_ms": 1.77,
                },
                "YOLOv10n": {
                    "params_m": 2.3,
                    "flops_g": 6.7,
                    "ap_val_ref_pct": 39.5,
                    "latency_ref_ms": 1.84,
                    "latencyf_ref_ms": 1.79,
                },
            }
            """
        ),
        code_cell(
            """
            def save_json(obj, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

            def load_json(path: Path):
                return json.loads(path.read_text(encoding="utf-8"))

            def safe_float(value):
                return round(float(value), 6)

            def get_run(run_key: str) -> dict:
                if run_key not in registry["runs"]:
                    raise KeyError(f"Run inconnu: {run_key}")
                return registry["runs"][run_key]

            def make_run_key(model_key: str, regime: str) -> str:
                return f"{model_key}_{regime}"

            def regime_title(regime: str) -> str:
                return regime if regime != "main" else "principal"

            def model_title(model_key: str) -> str:
                return "YOLOv10n" if model_key == "v10" else "YOLOv8n"

            def run_title(model_key: str, regime: str) -> str:
                return f"{model_title(model_key)} - {regime_title(regime)}"

            def ensure_axes_list(axes):
                if isinstance(axes, np.ndarray):
                    return axes.flatten().tolist()
                return [axes]

            def get_weights(run_key: str, kind: str = WEIGHT_KIND) -> str:
                info = get_run(run_key)
                path = Path(info[kind])
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint introuvable: {path}")
                return str(path)

            def maybe_set_end2end(model: YOLO, end2end):
                if end2end is None:
                    return
                head = None
                if hasattr(model.model, "model") and len(model.model.model):
                    head = model.model.model[-1]
                if head is not None and hasattr(head, "end2end"):
                    head.end2end = end2end

            def summarize_metrics(metrics) -> dict:
                names = metrics.names
                per_class = {}
                for idx, ap in enumerate(metrics.box.maps):
                    class_name = names[idx] if not isinstance(names, dict) else names.get(idx, str(idx))
                    per_class[class_name] = safe_float(ap)
                return {
                    "map50": safe_float(metrics.box.map50),
                    "map50_95": safe_float(metrics.box.map),
                    "precision": safe_float(metrics.box.mp),
                    "recall": safe_float(metrics.box.mr),
                    "per_class_map50_95": per_class,
                }

            def evaluate(weights: str, conf: float = 0.001, iou: float = 0.7, end2end=None) -> dict:
                model = YOLO(weights)
                maybe_set_end2end(model, end2end)
                metrics = model.val(
                    data=registry["data_yaml"],
                    imgsz=registry["config"]["imgsz"],
                    batch=registry["config"]["batch"],
                    conf=conf,
                    iou=iou,
                    split="val",
                    device=DEVICE,
                    workers=0,
                    plots=False,
                    verbose=False,
                )
                return summarize_metrics(metrics)

            def load_train_csv(run_key: str):
                path = Path(get_run(run_key)["results_csv"])
                if not path.exists():
                    return None
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                return df

            def find_metric_col(df: pd.DataFrame, keyword: str):
                for col in df.columns:
                    if keyword.lower() in col.lower():
                        return col
                return None

            def sync_device():
                if DEVICE == "mps" and hasattr(torch, "mps"):
                    torch.mps.synchronize()
                elif DEVICE == "cuda":
                    torch.cuda.synchronize()

            def checkpoint_size_mb(path: str | Path) -> float:
                return Path(path).stat().st_size / (1024 ** 2)

            def count_model_parameters(weights: str) -> dict:
                model = YOLO(weights)
                total = sum(p.numel() for p in model.model.parameters())
                trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                return {
                    "params_total": total,
                    "params_trainable": trainable,
                    "params_total_m": total / 1e6,
                    "params_trainable_m": trainable / 1e6,
                }

            def measure_latency(weights: str, image_dir: Path, n_iter: int = 30, warmup: int = 10) -> dict:
                model = YOLO(weights)
                imgs = sorted(image_dir.glob("*.jpg"))[: warmup + n_iter]
                for path in imgs[:warmup]:
                    model.predict(str(path), imgsz=registry["config"]["imgsz"], device=DEVICE, verbose=False)
                sync_device()
                t0 = time.perf_counter()
                for path in imgs[warmup : warmup + n_iter]:
                    model.predict(str(path), imgsz=registry["config"]["imgsz"], device=DEVICE, verbose=False)
                sync_device()
                dt = time.perf_counter() - t0
                return {"ms": 1000 * dt / n_iter, "fps": n_iter / dt, "device": DEVICE}

            def predict_boxes(weights: str, image_dir: Path, conf: float = 0.25, n: int = 100, end2end=None):
                model = YOLO(weights)
                maybe_set_end2end(model, end2end)
                counts, scores = [], []
                for path in sorted(image_dir.glob("*.jpg"))[:n]:
                    result = model.predict(
                        str(path),
                        conf=conf,
                        imgsz=registry["config"]["imgsz"],
                        device=DEVICE,
                        verbose=False,
                    )[0]
                    counts.append(len(result.boxes))
                    if len(result.boxes):
                        scores.extend(result.boxes.conf.detach().cpu().numpy().tolist())
                return np.array(counts), np.array(scores)
            """
        ),
        code_cell(
            """
            display_rows = []
            for run_key, info in registry["runs"].items():
                display_rows.append(
                    {
                        "run_key": run_key,
                        "run_name": info["run_name"],
                        "régime": regime_title(info["regime"]),
                        "modèle": model_title(info["model_key"]),
                        "best": Path(info["best"]).name,
                        "last": Path(info["last"]).name,
                    }
                )
            pd.DataFrame(display_rows).sort_values(["régime", "modèle"]).reset_index(drop=True)
            """
        ),
        code_cell(
            """
            SIZE_CACHE = ANALYSIS / "model_size_summary.json"
            if SIZE_CACHE.exists():
                size_payload = load_json(SIZE_CACHE)
            else:
                first_regime = REGIME_NAMES[0]
                model_refs = {
                    "YOLOv10n": make_run_key("v10", first_regime),
                    "YOLOv8n": make_run_key("v8", first_regime),
                }
                architecture_rows = []
                run_rows = []

                for model_label, ref_run in model_refs.items():
                    param_info = count_model_parameters(get_weights(ref_run))
                    architecture_rows.append(
                        {
                            "modèle": model_label,
                            "params_total_m": param_info["params_total_m"],
                            "params_trainable_m": param_info["params_trainable_m"],
                        }
                    )

                for run_key, info in registry["runs"].items():
                    run_rows.append(
                        {
                            "run_key": run_key,
                            "modèle": model_title(info["model_key"]),
                            "régime": regime_title(info["regime"]),
                            "best_mb": checkpoint_size_mb(info["best"]),
                            "last_mb": checkpoint_size_mb(info["last"]),
                        }
                    )

                size_payload = {
                    "architecture": architecture_rows,
                    "runs": run_rows,
                }
                save_json(size_payload, SIZE_CACHE)

            df_size_arch = pd.DataFrame(size_payload["architecture"]).sort_values("modèle")
            df_size_runs = pd.DataFrame(size_payload["runs"]).sort_values(["régime", "modèle"])

            display(df_size_arch)
            display(df_size_runs)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            df_size_arch.plot(
                kind="bar",
                x="modèle",
                y=["params_total_m"],
                ax=axes[0],
                rot=0,
                color=["C4"],
                edgecolor="black",
                legend=False,
            )
            axes[0].set_title("Nombre de paramètres")
            axes[0].set_ylabel("millions de paramètres")
            axes[0].grid(True, alpha=0.3, axis="y")

            checkpoint_plot = (
                df_size_runs.groupby("modèle")[["best_mb", "last_mb"]]
                .mean()
                .rename(columns={"best_mb": "best.pt", "last_mb": "last.pt"})
            )
            checkpoint_plot.plot(
                kind="bar",
                ax=axes[1],
                rot=0,
                edgecolor="black",
                width=0.7,
            )
            axes[1].set_title("Taille des checkpoints")
            axes[1].set_ylabel("Mo sur disque")
            axes[1].grid(True, alpha=0.3, axis="y")
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(FIGS / "model_size_compare.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            VOC_PATH = DATASETS_DIR / "VOC"
            VAL_SAMPLE = ROOT / "data" / "voc_sample"
            VAL_SAMPLE.mkdir(parents=True, exist_ok=True)

            src = VOC_PATH / "images" / "test2007"
            if src.exists() and len(list(VAL_SAMPLE.glob("*.jpg"))) < 120:
                for path in sorted(src.glob("*.jpg"))[:120]:
                    target = VAL_SAMPLE / path.name
                    if not target.exists():
                        shutil.copy(path, target)

            print({"voc_path_exists": VOC_PATH.exists(), "sample_images": len(list(VAL_SAMPLE.glob('*.jpg')))})
            """
        ),
        code_cell(
            """
            EVAL_CACHE = ANALYSIS / f"eval_summary_{WEIGHT_KIND}.json"
            if EVAL_CACHE.exists():
                eval_results = load_json(EVAL_CACHE)
            else:
                eval_results = {}
                for regime in REGIME_NAMES:
                    eval_results[make_run_key("v10", regime)] = evaluate(get_weights(make_run_key("v10", regime)))
                    eval_results[make_run_key("v8", regime)] = evaluate(get_weights(make_run_key("v8", regime)))
                save_json(eval_results, EVAL_CACHE)

            eval_rows = []
            for regime in REGIME_NAMES:
                eval_rows.append({"modèle": "YOLOv10n", "régime": regime_title(regime), **{k: eval_results[make_run_key("v10", regime)][k] for k in ["map50", "map50_95", "precision", "recall"]}})
                eval_rows.append({"modèle": "YOLOv8n", "régime": regime_title(regime), **{k: eval_results[make_run_key("v8", regime)][k] for k in ["map50", "map50_95", "precision", "recall"]}})
            df_eval = pd.DataFrame(eval_rows)
            df_eval
            """
        ),
        code_cell(
            """
            metrics = ["map50", "map50_95", "precision", "recall"]
            fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.5), sharey=False)

            for ax, metric in zip(axes, metrics):
                pivot = df_eval.pivot(index="régime", columns="modèle", values=metric)
                pivot.plot(kind="bar", ax=ax, rot=0, edgecolor="black", width=0.7)
                ax.set_title(metric)
                ax.set_xlabel("régime")
                ax.grid(True, alpha=0.3, axis="y")
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(FIGS / f"final_metrics_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, len(REGIME_NAMES), figsize=(6 * len(REGIME_NAMES), 4.2), sharey=True)
            axes = ensure_axes_list(axes)

            for ax, regime in zip(axes, REGIME_NAMES):
                for run_key, label, color in [
                    (make_run_key("v10", regime), "YOLOv10n", "C0"),
                    (make_run_key("v8", regime), "YOLOv8n", "C1"),
                ]:
                    df = load_train_csv(run_key)
                    if df is None:
                        continue
                    epoch_col = find_metric_col(df, "epoch")
                    map_col = find_metric_col(df, "mAP50-95")
                    if epoch_col is None or map_col is None:
                        continue
                    ax.plot(df[epoch_col], df[map_col], marker="o", label=label, color=color)
                ax.set_title(f"Régime {regime_title(regime)}")
                ax.set_xlabel("epoch")
                ax.set_ylabel("mAP50-95 val")
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(FIGS / f"learning_curves_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            ABL_CACHE = ANALYSIS / f"ablation_fixed_{WEIGHT_KIND}.json"
            if ABL_CACHE.exists():
                ablation_rows = load_json(ABL_CACHE)
            else:
                ablation_rows = []
                for regime in REGIME_NAMES:
                    ablation_rows.append({
                        "régime": regime,
                        "modèle": "v10_end2end",
                        **evaluate(get_weights(make_run_key("v10", regime)), conf=0.001, iou=0.7, end2end=True),
                    })
                    ablation_rows.append({
                        "régime": regime,
                        "modèle": "v10_nms_forced",
                        **evaluate(get_weights(make_run_key("v10", regime)), conf=0.001, iou=0.7, end2end=False),
                    })
                    ablation_rows.append({
                        "régime": regime,
                        "modèle": "v8",
                        **evaluate(get_weights(make_run_key("v8", regime)), conf=0.001, iou=0.7),
                    })
                save_json(ablation_rows, ABL_CACHE)

            df_ab = pd.DataFrame(ablation_rows)
            df_ab
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, len(REGIME_NAMES), figsize=(5.5 * len(REGIME_NAMES), 4), sharey=True)
            axes = ensure_axes_list(axes)
            for ax, regime in zip(axes, REGIME_NAMES):
                sub = df_ab[df_ab["régime"] == regime].set_index("modèle")
                sub["map50_95"].plot(kind="bar", ax=ax, edgecolor="black", color=["C2", "C0", "C1"])
                ax.set_title(f"Ablation fixe - régime {regime_title(regime)}")
                ax.set_ylabel("mAP50-95")
                ax.grid(True, alpha=0.3, axis="y")
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.savefig(FIGS / f"ablation_fixed_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            DIST_CACHE = ANALYSIS / f"detection_dist_{WEIGHT_KIND}.json"
            if DIST_CACHE.exists():
                cached = load_json(DIST_CACHE)
                dist = {k: (np.array(v["counts"]), np.array(v["scores"])) for k, v in cached.items()}
            else:
                dist = {}
                for regime in REGIME_NAMES:
                    dist[f"v10_end2end_{regime}"] = predict_boxes(get_weights(make_run_key("v10", regime)), VAL_SAMPLE, conf=0.25, n=100, end2end=True)
                    dist[f"v10_nms_forced_{regime}"] = predict_boxes(get_weights(make_run_key("v10", regime)), VAL_SAMPLE, conf=0.25, n=100, end2end=False)
                    dist[f"v8_{regime}"] = predict_boxes(get_weights(make_run_key("v8", regime)), VAL_SAMPLE, conf=0.25, n=100)
                serializable = {
                    k: {"counts": v[0].tolist(), "scores": v[1].tolist()}
                    for k, v in dist.items()
                }
                save_json(serializable, DIST_CACHE)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, len(REGIME_NAMES), figsize=(5.5 * len(REGIME_NAMES), 4), sharey=True)
            axes = ensure_axes_list(axes)
            for ax, regime in zip(axes, REGIME_NAMES):
                keys = [f"v10_end2end_{regime}", f"v10_nms_forced_{regime}", f"v8_{regime}"]
                data = [dist[k][0] for k in keys]
                ax.boxplot(data, labels=["v10_end2end", "v10_nms_forced", "v8"])
                ax.set_title(f"Régime {regime_title(regime)}")
                ax.set_ylabel("nb détections / image")
                ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(FIGS / f"detection_counts_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            def per_class_df(regime: str) -> pd.DataFrame:
                v10 = eval_results[make_run_key("v10", regime)]["per_class_map50_95"]
                v8 = eval_results[make_run_key("v8", regime)]["per_class_map50_95"]
                rows = [{"classe": k, "v10": v10[k], "v8": v8[k], "écart": v10[k] - v8[k]} for k in v10]
                return pd.DataFrame(rows).sort_values("écart")

            fig, axes = plt.subplots(1, len(REGIME_NAMES), figsize=(6.5 * len(REGIME_NAMES), 5), sharey=True)
            axes = ensure_axes_list(axes)
            for ax, regime in zip(axes, REGIME_NAMES):
                d = per_class_df(regime)
                y = np.arange(len(d))
                ax.barh(y - 0.2, d["v10"], 0.4, label="v10", edgecolor="black")
                ax.barh(y + 0.2, d["v8"], 0.4, label="v8", edgecolor="black")
                ax.set_yticks(y)
                ax.set_yticklabels(d["classe"])
                ax.set_xlabel("mAP50-95 par classe")
                ax.set_title(f"Régime {regime_title(regime)}")
                ax.grid(True, alpha=0.3, axis="x")
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(FIGS / f"per_class_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            LAT_CACHE = ANALYSIS / f"latency_{WEIGHT_KIND}.json"
            if LAT_CACHE.exists():
                lat = load_json(LAT_CACHE)
            else:
                lat = {}
                for regime in REGIME_NAMES:
                    for model_key in ["v10", "v8"]:
                        run_key = make_run_key(model_key, regime)
                        lat[run_key] = measure_latency(get_weights(run_key), VAL_SAMPLE)
                save_json(lat, LAT_CACHE)

            df_lat = pd.DataFrame(
                [
                    {"modèle": model_title(k.split("_")[0]), "régime": regime_title(k.split("_")[1]), **v}
                    for k, v in lat.items()
                ]
            )
            df_lat
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(6, 3.5))
            pivot = df_lat.pivot(index="régime", columns="modèle", values="ms")
            pivot.plot(kind="bar", ax=ax, rot=0, edgecolor="black", width=0.6)
            ax.set_ylabel("ms / image")
            ax.set_title("Latence d'inférence")
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(FIGS / f"latency_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            efficiency_rows = []
            for regime in REGIME_NAMES:
                for model_label, eval_key, lat_key in [
                    ("YOLOv10n", make_run_key("v10", regime), make_run_key("v10", regime)),
                    ("YOLOv8n", make_run_key("v8", regime), make_run_key("v8", regime)),
                ]:
                    ref = OFFICIAL_BENCH[model_label]
                    efficiency_rows.append(
                        {
                            "modèle": model_label,
                            "régime": regime_title(regime),
                            "Param.(M)": ref["params_m"],
                            "FLOPs(G)": ref["flops_g"],
                            "APval_ref(%)": ref["ap_val_ref_pct"],
                            "Latency_ref(ms)": ref["latency_ref_ms"],
                            "Latencyf_ref(ms)": ref["latencyf_ref_ms"],
                            "APval_VOC(%)": 100 * eval_results[eval_key]["map50_95"],
                            "Latency_local(ms)": lat[lat_key]["ms"],
                        }
                    )

            df_eff = pd.DataFrame(efficiency_rows).sort_values(["régime", "modèle"]).reset_index(drop=True)
            df_eff_rounded = df_eff.copy()
            for col in ["Param.(M)", "FLOPs(G)", "APval_ref(%)", "Latency_ref(ms)", "Latencyf_ref(ms)", "APval_VOC(%)", "Latency_local(ms)"]:
                df_eff_rounded[col] = df_eff_rounded[col].map(lambda x: round(float(x), 2))
            df_eff_rounded
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

            axes[0].axis("off")
            table = axes[0].table(
                cellText=df_eff_rounded.values,
                colLabels=df_eff_rounded.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.15, 1.35)
            axes[0].set_title("Résumé efficacité / performance")

            color_map = {"YOLOv10n": "C0", "YOLOv8n": "C1"}
            for _, row in df_eff.iterrows():
                axes[1].scatter(
                    row["Latency_local(ms)"],
                    row["APval_VOC(%)"],
                    s=row["Param.(M)"] * 260,
                    color=color_map[row["modèle"]],
                    alpha=0.75,
                    edgecolors="black",
                )
                axes[1].annotate(
                    f"{row['modèle']} {row['régime']}\\n{row['Param.(M)']:.1f}M | {row['FLOPs(G)']:.1f}G",
                    (row["Latency_local(ms)"], row["APval_VOC(%)"]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=8,
                )

            axes[1].set_title("AP local vs latence locale")
            axes[1].set_xlabel("Latency_local (ms)")
            axes[1].set_ylabel("APval_VOC = mAP50-95 val (%)")
            axes[1].grid(True, alpha=0.3)

            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label="YOLOv10n", markerfacecolor="C0", markeredgecolor="black", markersize=8),
                plt.Line2D([0], [0], marker="o", color="w", label="YOLOv8n", markerfacecolor="C1", markeredgecolor="black", markersize=8),
            ]
            axes[1].legend(handles=handles, fontsize=8, loc="best")

            plt.tight_layout()
            plt.savefig(FIGS / f"efficiency_summary_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            model_map = {}
            for regime in REGIME_NAMES:
                model_map[run_title("v10", regime)] = YOLO(get_weights(make_run_key("v10", regime)))
                model_map[run_title("v8", regime)] = YOLO(get_weights(make_run_key("v8", regime)))

            img_paths = sorted(VAL_SAMPLE.glob("*.jpg"))[:4]
            fig, axes = plt.subplots(len(img_paths), len(model_map), figsize=(3.5 * len(model_map), 3.2 * len(img_paths)))

            for row_idx, img_path in enumerate(img_paths):
                for col_idx, (name, model) in enumerate(model_map.items()):
                    result = model.predict(
                        str(img_path),
                        conf=0.25,
                        imgsz=registry["config"]["imgsz"],
                        device=DEVICE,
                        verbose=False,
                    )[0]
                    axes[row_idx, col_idx].imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
                    axes[row_idx, col_idx].set_title(name if row_idx == 0 else "", fontsize=10)
                    axes[row_idx, col_idx].axis("off")

            plt.tight_layout()
            plt.savefig(FIGS / f"qualitative_grid_{WEIGHT_KIND}.png", dpi=130)
            plt.show()
            """
        ),
        md_cell(
            """
            ## Résultat

            Toutes les figures sont enregistrées dans :

            - `results/mps_runs/analysis/figures/`

            Si tu veux comparer les derniers checkpoints au lieu des meilleurs :

            - change `WEIGHT_KIND = "last"` dans la première cellule,
            - puis relance le notebook.
            """
        ),
    ]
)


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    write_notebook(NOTEBOOKS_DIR / "01_train_yolov10_mps.ipynb", TRAIN_NOTEBOOK)
    write_notebook(NOTEBOOKS_DIR / "02_viz_compare_yolov10_mps.ipynb", VIZ_NOTEBOOK)
    print("Notebooks générés dans", NOTEBOOKS_DIR)


if __name__ == "__main__":
    main()
