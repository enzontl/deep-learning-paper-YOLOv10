#!/usr/bin/env python3
"""Build a fresh comparison notebook wired to explicit run directories."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_viz_compare_yolov10_mps.ipynb"


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


NB = notebook(
    [
        md_cell(
            f"""
            # 02. Visualisations YOLOv10 vs YOLOv8 [v3 explicite]

            Ce notebook compare **explicitement** les deux runs suivants :

            - `YOLOv10n` : `{ROOT / 'results' / 'mps_runs' / 'yolov10n_main_mps_sgd_main'}`
            - `YOLOv8n` : `{ROOT / 'results' / 'mps_runs' / 'yolov8n_main_mps_sgd_main'}`

            Il ne dépend pas d'un ancien registre. Si ces chemins existent, c'est eux qui sont utilisés.
            """
        ),
        code_cell(
            f"""
            import os
            import json
            import time
            import shutil
            from pathlib import Path

            import cv2
            import yaml
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import torch
            from ultralytics import YOLO
            from ultralytics import settings as ul_settings

            plt.rcParams.update({{"figure.dpi": 110, "font.size": 10}})

            ROOT = Path(r"{ROOT}")
            os.chdir(ROOT)

            RUN_V10 = ROOT / "results" / "mps_runs" / "yolov10n_main_mps_sgd_main"
            RUN_V8 = ROOT / "results" / "mps_runs" / "yolov8n_main_mps_sgd_main"
            RUNS = {{
                "YOLOv10n": RUN_V10,
                "YOLOv8n": RUN_V8,
            }}

            for name, path in RUNS.items():
                if not path.exists():
                    raise FileNotFoundError(f"Run introuvable pour {{name}}: {{path}}")

            RESULTS_ROOT = ROOT / "results" / "mps_runs"
            ANALYSIS = RESULTS_ROOT / "analysis_explicit"
            FIGS = ANALYSIS / "figures"
            DATASETS_DIR = ROOT / "data" / "datasets"
            for path in [ANALYSIS, FIGS, DATASETS_DIR]:
                path.mkdir(parents=True, exist_ok=True)

            DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
            ul_settings.update({{"datasets_dir": str(DATASETS_DIR), "runs_dir": str(RESULTS_ROOT)}})

            WEIGHT_KIND = "best"  # ou "last"
            DATA_YAML = "VOC.yaml"

            OFFICIAL_BENCH = {{
                "YOLOv8n": {{
                    "params_m": 3.2,
                    "flops_g": 8.7,
                    "ap_val_ref_pct": 37.3,
                    "latency_ref_ms": 6.16,
                    "latencyf_ref_ms": 1.77,
                }},
                "YOLOv10n": {{
                    "params_m": 2.3,
                    "flops_g": 6.7,
                    "ap_val_ref_pct": 39.5,
                    "latency_ref_ms": 1.84,
                    "latencyf_ref_ms": 1.79,
                }},
            }}

            print({{
                "device": DEVICE,
                "RUN_V10": str(RUN_V10),
                "RUN_V8": str(RUN_V8),
                "WEIGHT_KIND": WEIGHT_KIND,
            }})
            """
        ),
        code_cell(
            """
            def save_json(obj, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

            def load_json(path: Path):
                return json.loads(path.read_text(encoding="utf-8"))

            def load_yaml(path: Path):
                return yaml.safe_load(path.read_text(encoding="utf-8"))

            def load_train_csv(run_dir: Path) -> pd.DataFrame:
                path = run_dir / "results.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                return df

            def get_weights(run_dir: Path, kind: str = WEIGHT_KIND) -> str:
                path = run_dir / "weights" / f"{kind}.pt"
                if not path.exists():
                    raise FileNotFoundError(path)
                return str(path)

            def model_title_from_run(run_dir: Path) -> str:
                return "YOLOv10n" if "yolov10" in run_dir.name else "YOLOv8n"

            def maybe_set_end2end(model: YOLO, end2end):
                if end2end is None:
                    return
                head = None
                if hasattr(model.model, "model") and len(model.model.model):
                    head = model.model.model[-1]
                if head is not None and hasattr(head, "end2end"):
                    head.end2end = end2end

            def safe_float(value):
                return round(float(value), 6)

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
                    data=DATA_YAML,
                    imgsz=416,
                    batch=16,
                    conf=conf,
                    iou=iou,
                    split="val",
                    device=DEVICE,
                    workers=0,
                    plots=False,
                    verbose=False,
                )
                return summarize_metrics(metrics)

            def sync_device():
                if DEVICE == "mps" and hasattr(torch, "mps"):
                    torch.mps.synchronize()
                elif DEVICE == "cuda":
                    torch.cuda.synchronize()

            def measure_latency(weights: str, image_dir: Path, n_iter: int = 30, warmup: int = 10) -> dict:
                model = YOLO(weights)
                imgs = sorted(image_dir.glob("*.jpg"))[: warmup + n_iter]
                for path in imgs[:warmup]:
                    model.predict(str(path), imgsz=416, device=DEVICE, verbose=False)
                sync_device()
                t0 = time.perf_counter()
                for path in imgs[warmup : warmup + n_iter]:
                    model.predict(str(path), imgsz=416, device=DEVICE, verbose=False)
                sync_device()
                dt = time.perf_counter() - t0
                return {"ms": 1000 * dt / n_iter, "fps": n_iter / dt, "device": DEVICE}

            def predict_boxes(weights: str, image_dir: Path, conf: float = 0.25, n: int = 100, end2end=None):
                model = YOLO(weights)
                maybe_set_end2end(model, end2end)
                counts, scores = [], []
                for path in sorted(image_dir.glob("*.jpg"))[:n]:
                    result = model.predict(str(path), conf=conf, imgsz=416, device=DEVICE, verbose=False)[0]
                    counts.append(len(result.boxes))
                    if len(result.boxes):
                        scores.extend(result.boxes.conf.detach().cpu().numpy().tolist())
                return np.array(counts), np.array(scores)

            def count_model_parameters(weights: str) -> dict:
                model = YOLO(weights)
                total = sum(p.numel() for p in model.model.parameters())
                trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                return {
                    "params_total_m": total / 1e6,
                    "params_trainable_m": trainable / 1e6,
                }

            def checkpoint_size_mb(path: str | Path) -> float:
                return Path(path).stat().st_size / (1024 ** 2)
            """
        ),
        code_cell(
            """
            run_rows = []
            for model_name, run_dir in RUNS.items():
                args = load_yaml(run_dir / "args.yaml")
                run_rows.append(
                    {
                        "modèle": model_name,
                        "run_dir": str(run_dir),
                        "best": get_weights(run_dir, "best"),
                        "last": get_weights(run_dir, "last"),
                        "epochs": args["epochs"],
                        "fraction": args["fraction"],
                        "optimizer": args["optimizer"],
                        "batch": args["batch"],
                        "imgsz": args["imgsz"],
                    }
                )
            df_runs = pd.DataFrame(run_rows)
            df_runs
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            for model_name, run_dir in RUNS.items():
                df = load_train_csv(run_dir)
                ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], marker="o", label=model_name)
            ax.set_title("Courbes d'apprentissage")
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
            EVAL_CACHE = ANALYSIS / f"eval_summary_{WEIGHT_KIND}.json"
            if EVAL_CACHE.exists():
                eval_results = load_json(EVAL_CACHE)
            else:
                eval_results = {
                    "YOLOv10n": evaluate(get_weights(RUN_V10)),
                    "YOLOv8n": evaluate(get_weights(RUN_V8)),
                }
                save_json(eval_results, EVAL_CACHE)

            df_eval = pd.DataFrame(
                [
                    {"modèle": name, **{k: vals[k] for k in ["map50", "map50_95", "precision", "recall"]}}
                    for name, vals in eval_results.items()
                ]
            )
            df_eval
            """
        ),
        code_cell(
            """
            metrics = ["map50", "map50_95", "precision", "recall"]
            fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.5))
            for ax, metric in zip(axes, metrics):
                ax.bar(df_eval["modèle"], df_eval[metric], edgecolor="black", color=["C0", "C1"])
                ax.set_title(metric)
                ax.grid(True, alpha=0.3, axis="y")
                ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            plt.savefig(FIGS / f"final_metrics_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            ABL_CACHE = ANALYSIS / f"ablation_{WEIGHT_KIND}.json"
            if ABL_CACHE.exists():
                df_ab = pd.DataFrame(load_json(ABL_CACHE))
            else:
                rows = [
                    {"modèle": "YOLOv10 end2end", **evaluate(get_weights(RUN_V10), end2end=True)},
                    {"modèle": "YOLOv10 NMS forcée", **evaluate(get_weights(RUN_V10), end2end=False)},
                    {"modèle": "YOLOv8", **evaluate(get_weights(RUN_V8))},
                ]
                save_json(rows, ABL_CACHE)
                df_ab = pd.DataFrame(rows)
            df_ab
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_ab["modèle"], df_ab["map50_95"], edgecolor="black", color=["C2", "C0", "C1"])
            ax.set_title("Ablation fixe")
            ax.set_ylabel("mAP50-95")
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="x", rotation=20)
            plt.tight_layout()
            plt.savefig(FIGS / f"ablation_{WEIGHT_KIND}.png", dpi=150)
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
            LAT_CACHE = ANALYSIS / f"latency_{WEIGHT_KIND}.json"
            if LAT_CACHE.exists():
                lat = load_json(LAT_CACHE)
            else:
                lat = {
                    "YOLOv10n": measure_latency(get_weights(RUN_V10), VAL_SAMPLE),
                    "YOLOv8n": measure_latency(get_weights(RUN_V8), VAL_SAMPLE),
                }
                save_json(lat, LAT_CACHE)
            df_lat = pd.DataFrame([{"modèle": k, **v} for k, v in lat.items()])
            df_lat
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_lat["modèle"], df_lat["ms"], edgecolor="black", color=["C0", "C1"])
            ax.set_title("Latence locale")
            ax.set_ylabel("ms / image")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(FIGS / f"latency_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            size_rows = []
            for model_name, run_dir in RUNS.items():
                ref = OFFICIAL_BENCH[model_name]
                params = count_model_parameters(get_weights(run_dir))
                size_rows.append(
                    {
                        "modèle": model_name,
                        "Param.(M)": ref["params_m"],
                        "FLOPs(G)": ref["flops_g"],
                        "APval_ref(%)": ref["ap_val_ref_pct"],
                        "Latency_ref(ms)": ref["latency_ref_ms"],
                        "Latencyf_ref(ms)": ref["latencyf_ref_ms"],
                        "APval_VOC(%)": 100 * eval_results[model_name]["map50_95"],
                        "Latency_local(ms)": lat[model_name]["ms"],
                        "Checkpoint_best(MB)": checkpoint_size_mb(get_weights(run_dir, "best")),
                        "Checkpoint_last(MB)": checkpoint_size_mb(get_weights(run_dir, "last")),
                        "Params_recounted(M)": params["params_total_m"],
                    }
                )
            df_eff = pd.DataFrame(size_rows)
            for col in df_eff.columns[1:]:
                df_eff[col] = df_eff[col].map(lambda x: round(float(x), 2))
            df_eff
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

            axes[0].axis("off")
            table = axes[0].table(
                cellText=df_eff.values,
                colLabels=df_eff.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.35)
            axes[0].set_title("Résumé efficacité / performance")

            for _, row in df_eff.iterrows():
                axes[1].scatter(
                    row["Latency_local(ms)"],
                    row["APval_VOC(%)"],
                    s=row["Param.(M)"] * 260,
                    alpha=0.75,
                    edgecolors="black",
                    label=row["modèle"],
                )
                axes[1].annotate(
                    f"{row['modèle']}\\n{row['Param.(M)']:.1f}M | {row['FLOPs(G)']:.1f}G",
                    (row["Latency_local(ms)"], row["APval_VOC(%)"]),
                    textcoords="offset points",
                    xytext=(8, 6),
                    fontsize=8,
                )

            axes[1].set_title("AP local vs latence locale")
            axes[1].set_xlabel("Latency_local (ms)")
            axes[1].set_ylabel("APval_VOC = mAP50-95 val (%)")
            axes[1].grid(True, alpha=0.3)
            handles, labels = axes[1].get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            axes[1].legend(uniq.values(), uniq.keys(), fontsize=8)

            plt.tight_layout()
            plt.savefig(FIGS / f"efficiency_summary_{WEIGHT_KIND}.png", dpi=150)
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
                dist = {
                    "YOLOv10 end2end": predict_boxes(get_weights(RUN_V10), VAL_SAMPLE, end2end=True),
                    "YOLOv10 NMS forcée": predict_boxes(get_weights(RUN_V10), VAL_SAMPLE, end2end=False),
                    "YOLOv8": predict_boxes(get_weights(RUN_V8), VAL_SAMPLE),
                }
                save_json({k: {"counts": v[0].tolist(), "scores": v[1].tolist()} for k, v in dist.items()}, DIST_CACHE)
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(
                [dist["YOLOv10 end2end"][0], dist["YOLOv10 NMS forcée"][0], dist["YOLOv8"][0]],
                labels=["v10_end2end", "v10_nms_forced", "v8"],
            )
            ax.set_title("Nombre de détections par image")
            ax.set_ylabel("nb détections / image")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(FIGS / f"detection_counts_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(12, 4))
            compare = pd.DataFrame(
                {
                    "classe": list(eval_results["YOLOv10n"]["per_class_map50_95"].keys()),
                    "YOLOv10n": list(eval_results["YOLOv10n"]["per_class_map50_95"].values()),
                    "YOLOv8n": list(eval_results["YOLOv8n"]["per_class_map50_95"].values()),
                }
            )
            compare["écart"] = compare["YOLOv10n"] - compare["YOLOv8n"]
            compare = compare.sort_values("écart")
            y = np.arange(len(compare))
            ax.barh(y - 0.2, compare["YOLOv10n"], 0.4, label="YOLOv10n", edgecolor="black")
            ax.barh(y + 0.2, compare["YOLOv8n"], 0.4, label="YOLOv8n", edgecolor="black")
            ax.set_yticks(y)
            ax.set_yticklabels(compare["classe"])
            ax.set_xlabel("mAP50-95 par classe")
            ax.set_title("Comparaison par classe")
            ax.grid(True, alpha=0.3, axis="x")
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGS / f"per_class_{WEIGHT_KIND}.png", dpi=150)
            plt.show()
            """
        ),
        code_cell(
            """
            model_map = {
                "YOLOv10n": YOLO(get_weights(RUN_V10)),
                "YOLOv8n": YOLO(get_weights(RUN_V8)),
            }
            img_paths = sorted(VAL_SAMPLE.glob("*.jpg"))[:4]
            fig, axes = plt.subplots(len(img_paths), len(model_map), figsize=(7, 3.2 * len(img_paths)))
            if len(img_paths) == 1:
                axes = np.array([axes])

            for row_idx, img_path in enumerate(img_paths):
                for col_idx, (name, model) in enumerate(model_map.items()):
                    result = model.predict(str(img_path), conf=0.25, imgsz=416, device=DEVICE, verbose=False)[0]
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

            Les figures sont enregistrées dans `results/mps_runs/analysis_explicit/figures/`.

            Si tu veux comparer `last.pt` au lieu de `best.pt`, change `WEIGHT_KIND = "last"` dans la première cellule puis relance.
            """
        ),
    ]
)


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(NB, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Notebook écrit dans {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
