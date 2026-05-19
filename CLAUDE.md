# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# YOLOv10 Reproducibility Study

Reproduction et étude critique du papier *YOLOv10: Real-Time End-to-End Object Detection* dans le cadre du cours d'introduction au Deep Learning (Albert School × Mines Paris PSL). Référence : https://arxiv.org/abs/2405.14458

## Tech Stack

- Python 3.10+
- PyTorch (CUDA obligatoire pour l'entraînement)
- Ultralytics (baselines YOLO + DETR) — https://github.com/ultralytics/ultralytics
- YOLOv10 officiel (optionnel / comparaison) — https://github.com/THU-MIG/yolov10
- OpenCV, Pillow, NumPy, Pandas
- <!-- TODO: fill in --> logging (Weights & Biases ou TensorBoard — à confirmer)
- <!-- TODO: fill in --> dataset (COCO 2017 ou sous-ensemble — à confirmer)

## Project Structure

```
deep-learning-paper-YOLOv10/
├── tasks/
│   ├── todo.md          # plan de travail courant (créer si absent)
│   └── lessons.md       # apprentissages accumulés (créer si absent)
├── configs/             # <!-- TODO: fill in --> configs YAML (modèle, hyperparams, seeds)
├── data/                # <!-- TODO: fill in --> scripts de préparation du dataset
├── notebooks/           # analyse, visualisations, rapport final
├── scripts/             # train.py, eval.py, infer.py — entry points principaux
├── results/             # checkpoints, métriques, figures (gitignore les gros fichiers)
├── requirements.txt     # <!-- TODO: fill in -->
└── CLAUDE.md
```

## Development Commands

**Installation**
```bash
# <!-- TODO: fill in après création de requirements.txt -->
pip install -r requirements.txt
```

**Entraînement**
```bash
# <!-- TODO: fill in -->
python scripts/train.py --config configs/yolov10n.yaml --seed 42
```

**Évaluation**
```bash
# <!-- TODO: fill in -->
python scripts/eval.py --weights results/best.pt --data data/coco.yaml
```

**Inférence**
```bash
# <!-- TODO: fill in -->
python scripts/infer.py --source path/to/image --weights results/best.pt
```

**Lint / Format**
```bash
# <!-- TODO: fill in -->
ruff check . && ruff format .
```

**Tests**
```bash
# <!-- TODO: fill in -->
pytest tests/
```

## Architecture & Key Conventions

### Entry points
- `scripts/train.py` — lancement entraînement, charge config YAML, initialise seed
- `scripts/eval.py` — calcule mAP50, mAP50-95 sur validation set
- `scripts/infer.py` — inférence image/vidéo, NMS-free (YOLOv10 ne requiert pas de post-NMS)

### Conventions de nommage
- Configs : `configs/{model_variant}_{dataset}_{resolution}.yaml` (ex. `yolov10n_coco_640.yaml`)
- Checkpoints : `results/{run_name}/weights/best.pt` et `last.pt`
- Runs : nommés `{model}_{date}_{seed}` pour reproductibilité

### Pipeline data
- Téléchargement et préparation dans `data/` avec scripts versionnés
- Splits COCO standard (train2017 / val2017) — ne pas modifier les splits
- Augmentations définies dans la config YAML, pas dans le code Python

### Reproductibilité (priorité absolue)
- Fixer `seed=42` partout (`torch.manual_seed`, `numpy.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic=True`)
- Versionner les configs YAML et les requirements (`pip freeze > requirements.txt` après chaque install)
- Logger hardware, CUDA version, et hyperparams à chaque run
- Tout écart avec les métriques du papier doit être documenté dans le rapport

### Gotchas YOLOv10 / Ultralytics
- YOLOv10 utilise un **dual label assignment** (one-to-one + one-to-many) : l'inférence utilise uniquement la branche one-to-one → **NMS-free**
- Ne pas activer NMS en post-processing pour YOLOv10 (contrairement à YOLOv8/v9)
- La branche one-to-many sert uniquement pendant l'entraînement (supervision auxiliaire)
- Comparer les métriques avec le papier sur COCO val2017, IoU=0.50:0.95, toutes catégories

---

## DÉMARRAGE DE SESSION

1. Lire tasks/lessons.md — appliquer toutes les leçons avant de toucher quoi que ce soit
2. Lire tasks/todo.md — comprendre l'état actuel
3. Si aucun des deux n'existe, les créer avant de commencer

## WORKFLOW

### 1. Planifier d'abord

- Passer en mode plan pour toute tâche non triviale (3+ étapes)
- Écrire le plan dans tasks/todo.md avant d'implémenter
- Si quelque chose ne va pas, STOP et re-planifier — ne jamais forcer

### 2. Stratégie sous-agents

- Utiliser des sous-agents pour garder le contexte principal propre
- Une tâche par sous-agent
- Investir plus de compute sur les problèmes difficiles

### 3. Boucle d'auto-amélioration

- Après toute correction : mettre à jour tasks/lessons.md
- Format : [date] | ce qui a mal tourné | règle pour l'éviter
- Relire les leçons à chaque démarrage de session

### 4. Standard de vérification

- Se demander : « Est-ce qu'un staff engineer validerait ça ? »

### 5. Exiger l'élégance

- Pour les changements non triviaux : existe-t-il une solution plus élégante ?
- Si un fix semble bricolé : le reconstruire proprement
- Ne pas sur-ingénieriser les choses simples

### 6. Correction de bugs autonome

- Quand on reçoit un bug : le corriger directement
- Aller dans les logs, trouver la cause racine, résoudre
- Pas besoin d'être guidé étape par étape

## PRINCIPES FONDAMENTAUX

- Simplicité d'abord — toucher un minimum de code
- Pas de paresse — causes racines uniquement, pas de fixes temporaires
- Ne jamais supposer — vérifier chemins, APIs, variables avant utilisation
- Demander une seule fois — une question en amont si nécessaire, ne jamais interrompre en cours de tâche

## GESTION DES TÂCHES

1. Planifier → tasks/todo.md
2. Vérifier → confirmer avant d'implémenter
3. Suivre → marquer comme terminé au fur et à mesure
4. Expliquer → résumé de haut niveau à chaque étape
5. Apprendre → tasks/lessons.md après corrections

## APPRENTISSAGES

(Claude remplit cette section au fil du temps)
