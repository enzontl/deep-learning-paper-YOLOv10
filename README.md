# YOLOv10 Reduced Reproduction

This repository contains a lightweight reproduction of *YOLOv10: Real-Time End-to-End Object Detection* on **Pascal VOC**, adapted for a **Mac Mini / Apple Silicon** workflow.

The project is intentionally simple:

- training is done from a notebook
- comparison and analysis are done from a second notebook
- heavy artifacts such as datasets, checkpoints, and figures stay outside Git

## Goal

The main question is whether **YOLOv10 in its native end-to-end / NMS-free mode** remains competitive against a similarly sized **YOLOv8n** baseline in a reduced experimental setting.

This is not meant to reproduce the full paper at COCO scale.  
It is meant to:

- isolate the main idea of the paper
- test it on a smaller setup
- compare it to a clean baseline
- analyze where it works or fails

## Dataset Choice

We keep **Pascal VOC** because it is the most relevant dataset in the allowed list for an object detection paper such as YOLOv10.

To keep the project feasible locally, the setup uses:

- a reduced training fraction
- an MPS-friendly configuration for Apple Silicon
- a longer training schedule

## Notebooks

The repository is organized around two notebooks:

- [01_train_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/01_train_yolov10_mps.ipynb)
- [02_viz_compare_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/02_viz_compare_yolov10_mps.ipynb)

### 1. Training notebook

The training notebook:

- uses `mps` when available
- uses an MPS-friendly setup
- trains `YOLOv10n` and `YOLOv8n`
- stores `best.pt`, `last.pt`, and `results.csv`

Current default training setup:

- `imgsz = 640`
- `batch = 8`
- `epochs = 100`
- `fraction = 0.10`
- `optimizer = SGD`
- `cos_lr = True`
- `device = mps`
- `workers = 0`
- `plots = False`

### 2. Visualization notebook

The visualization notebook reads the two run folders directly and produces:

- learning curves
- final metrics
- end-to-end vs forced-NMS ablation
- local latency comparison
- efficiency summary table
- detection count distribution
- per-class comparison
- qualitative examples

## Recommended Usage

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Open [01_train_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/01_train_yolov10_mps.ipynb) and run all cells.

3. Open [02_viz_compare_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/02_viz_compare_yolov10_mps.ipynb) and run all cells.

## Main Experimental Logic

The most important comparison is:

- `YOLOv10 end2end`: the native NMS-free inference behavior
- `YOLOv10 NMS forced`: the same checkpoint evaluated with a more classical post-processing path
- `YOLOv8n`: the baseline

This makes it possible to separate two questions:

1. Is YOLOv10 better than a standard baseline?
2. Is the NMS-free branch itself strong enough in this reduced setting?

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_train_yolov10_mps.ipynb
│   ├── 02_viz_compare_yolov10_mps.ipynb
│   └── yolov10_reproduction.ipynb
├── data/
├── results/
├── tasks/
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

- Large files in `data/` and `results/` are intentionally ignored by Git.
- The repository keeps only the code and notebooks needed to reproduce the analysis.
- The final discussion should focus first on `mAP50-95`, then on the end-to-end vs forced-NMS ablation, then on latency and qualitative behavior.
