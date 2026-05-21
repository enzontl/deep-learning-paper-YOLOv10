# YOLOv10 Reproducibility Study

Reproduction réduite du papier *YOLOv10: Real-Time End-to-End Object Detection*
(Wang et al., 2024, [arXiv:2405.14458](https://arxiv.org/abs/2405.14458))
dans le cadre du cours d'introduction au Deep Learning
(Albert School x Mines Paris PSL).

## Angle de reproduction

Idée centrale étudiée : la suppression de la NMS à l'inférence, obtenue par
l'assignation duale (one-to-many + one-to-one) et la métrique de matching
cohérente.

Question expérimentale : la branche one-to-one suffit-elle à produire des
détections sans doublons sur un dataset plus simple que COCO, et à quel
coût en précision ?

## Méthode B : compute-parity entre deux régimes

La one-to-one head reçoit un seul positif par objet par epoch, contre N pour
la one-to-many. Elle pourrait demander plus de passages par objet pour
converger. On teste sous deux régimes équivalents en nombre de vues d'images
total :

- **Régime A** : 30% de VOC 2007+2012 (~5000 images), 10 epochs.
- **Régime B** : 10% de VOC 2007+2012 (~1655 images), 30 epochs.

Compute identique (~50000 vues), mais en B chaque ground truth est vu 30
fois au lieu de 10 (ratio 3x). Permet de tester si la one-to-one souffrait
de sous-entraînement.

Optimisé T4 : imgsz=416, batch=32, cache RAM. Notebook complet en ~15 min.

## Structure

```
.
├── notebooks/
│   └── yolov10_reproduction.ipynb   # tout le code et toutes les expériences
├── data/                            # échantillon d'images pour latence et qualitatif
├── results/                         # checkpoints, JSON, figures
├── tasks/
│   ├── todo.md
│   └── lessons.md
├── requirements.txt
├── CLAUDE.md
└── README.md
```

Pas de scripts externes. Le notebook est self-contained : toutes les
fonctions (training, évaluation, ablation, latence, visualisations) sont
définies en cellules.

## Utilisation sur Colab (recommandé)

1. Copier le dossier sur Google Drive.
2. Ouvrir `notebooks/yolov10_reproduction.ipynb` depuis Drive.
3. Runtime > Change runtime type > GPU (T4 minimum).
4. Adapter `PROJECT_DIR` dans la cellule de configuration si besoin.
5. Run all.

Le dataset VOC (~2.8 Go) est placé sur le disque local Colab
(`/content/datasets`, volatil) pour ne pas saturer Drive. Checkpoints, JSON
et figures restent sur Drive. Le dataset est retéléchargé à chaque session
(~3 min).

## Budget de calcul

Sur T4 (Colab gratuit) avec les réglages par défaut du notebook (imgsz=416,
batch=32, cache RAM, sous-échantillonnage VOC) :

- 4 entraînements : ~2 min chacun = ~8 min
- Évaluation finale + ablation (~30 evals) : ~3 min
- Latence + figures + qualitatif : ~2 min
- Total notebook : **~15 min**

Le notebook saute toute étape déjà cachée (`best.pt`, JSON existants), donc
on peut interrompre et reprendre sans rejouer ce qui est fini.

## Reproductibilité

- `seed=42` partout (torch, numpy, random, cudnn deterministic).
- Hyperparamètres d'entraînement fixés dans `train_if_needed` dans le
  notebook.
- Chaque étape sérialise ses résultats en JSON dans `results/`.

## Limites

Détaillées en section 11 du notebook. En résumé : fine-tuning court depuis
COCO, VOC moins dense que COCO, latence PyTorch et pas TensorRT, un seul
seed.
