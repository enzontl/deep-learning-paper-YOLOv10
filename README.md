# YOLOv10 Reproducibility Study

Reproduction réduite du papier *YOLOv10: Real-Time End-to-End Object Detection*
(Wang et al., 2024, [arXiv:2405.14458](https://arxiv.org/abs/2405.14458))
adaptée à un **Mac Mini Apple Silicon**.

## Angle de reproduction

L'idée centrale conservée est la suivante : comparer un modèle YOLOv10
NMS-free à une baseline simple de même taille (`YOLOv8n`) dans un cadre plus
léger que le papier original.

Comme votre projet doit rester faisable localement, on garde une vraie tâche
de détection mais on réduit le coût de calcul :

- dataset : **Pascal VOC**, mais entraînement sur une **fraction déterministe**
  du train set ;
- budget raisonnable : **environ 40 epochs** ;
- optimisation rapprochée du papier : **SGD** ;
- matériel : **`mps`** sur macOS, pas `cuda`.

## Pourquoi garder VOC

Dans la liste proposée (`CIFAR-10/100`, `STL-10`, `Oxford-IIIT Pet`,
`Pascal VOC`, `Tiny-ImageNet`, `Fashion-MNIST`), **Pascal VOC est le seul
dataset directement adapté à la détection d'objets**, donc le plus cohérent
pour une reproduction de YOLOv10.

La réduction de complexité se fait donc en travaillant sur une **petite
fraction de VOC** plutôt qu'en changeant de tâche.

## Nouveau workflow notebook

Le dépôt contient maintenant deux notebooks pensés pour un usage local sur Mac :

- [01_train_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/01_train_yolov10_mps.ipynb)
- [02_viz_compare_yolov10_mps.ipynb](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/notebooks/02_viz_compare_yolov10_mps.ipynb)

Le premier notebook :

- détecte `mps` automatiquement ;
- bascule sur `cpu` si `mps` n'est pas disponible ;
- entraîne `yolov10n.pt` et `yolov8n.pt` ;
- utilise **SGD** ;
- applique une **cosine decay** avec un LR initial plus élevé puis plus faible
  en fin d'entraînement ;
- sauvegarde `best.pt`, `last.pt`, `results.csv` et un registre JSON dans `results/mps_runs/`.

Le second notebook recharge ensuite ces runs pour produire les comparaisons,
les courbes et les visualisations.

## Lancement recommandé

Installer les dépendances :

```bash
python3 -m pip install -r requirements.txt
```

Ordre recommandé :

1. Ouvrir `notebooks/01_train_yolov10_mps.ipynb`
2. Faire **Run All**
3. Ouvrir `notebooks/02_viz_compare_yolov10_mps.ipynb`
4. Faire **Run All**

Réglages par défaut du notebook d'entraînement :

- Régime principal : `fraction=0.20`, `epochs=40`
- `imgsz=416`
- `batch=8`
- `optimizer=SGD`
- `lr0=0.01`
- `lrf=0.01`
- `device=mps`

## Script optionnel

Le script suivant reste disponible si vous voulez lancer l'entraînement hors notebook :

- [scripts/train_yolov10_mps.py](/Users/guillaumerabeau/deep-learning-paper-YOLOv10/scripts/train_yolov10_mps.py)

Exemple :

```bash
python3 scripts/train_yolov10_mps.py --epochs 40 --fraction 0.20 --batch 8 --imgsz 416
```

Si la mémoire est trop juste sur votre machine :

```bash
python3 scripts/train_yolov10_mps.py --epochs 30 --fraction 0.15 --batch 4 --imgsz 416
```

## Réglages choisis

Le notebook et le script utilisent les mêmes principes :

- `device=mps`
- `optimizer=SGD`
- `epochs=40`
- `fraction=0.20`
- `imgsz=416`
- `batch=8`
- `lr0=0.01`
- `lrf=0.01`
- `cos_lr=True`
- `workers=0`
- `amp=False`

Ce choix est volontairement conservateur pour la stabilité sur macOS.

## Proposition d'expérience pour le rapport

Vous pouvez structurer la reproduction ainsi :

- **Ce qui est reproduit** : comparaison d'un YOLOv10 NMS-free contre une
  baseline YOLOv8n sur une tâche de détection.
- **Ce qui est simplifié** : petit modèle (`n`), fine-tuning court,
  fraction de VOC, matériel local MPS.
- **Baseline** : `YOLOv8n`.
- **Changement par rapport au papier** : matériel `MPS`, dataset plus petit et
  budget réduit.
- **Ablation simple** : comparer `YOLOv10 end2end` à `YOLOv10 NMS forcée`, ou
  faire varier `epochs` (30 vs 50) / `fraction` (0.15 vs 0.20).

## Structure

```text
.
├── notebooks/
│   ├── 01_train_yolov10_mps.ipynb
│   ├── 02_viz_compare_yolov10_mps.ipynb
│   └── yolov10_reproduction.ipynb
├── scripts/
│   ├── generate_mps_notebooks.py
│   └── train_yolov10_mps.py
├── data/
├── results/
├── tasks/
├── requirements.txt
└── README.md
```

## Limites

- Le dataset complet VOC est quand même téléchargé par Ultralytics.
- Ce n'est pas une reproduction exhaustive du papier, mais une reproduction
  réduite et rigoureuse.
- Les résultats dépendront beaucoup du budget mémoire et du temps
  d'entraînement disponibles sur votre Mac Mini.
