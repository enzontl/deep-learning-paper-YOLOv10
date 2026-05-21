# Plan de travail (révisé)

## Choix de reproduction
- Papier : YOLOv10 (Wang et al., 2024), https://arxiv.org/abs/2405.14458
- Idée centrale reproduite : suppression de la NMS via assignation duale + métrique cohérente
- Composants efficacité/précision : non re-entraînés, hérités du checkpoint

## Méthode B : compute-parity entre deux régimes
- Régime A : VOC 2007+2012 (16551 images), 15 epochs
- Régime B : VOC 2007 trainval (5011 images), 50 epochs
- Compute équivalent (~15500 pas) mais 3.3x plus de passages par objet en B
- Permet de tester si la one-to-one head souffrait de sous-entraînement en A

## Modèles
- YOLOv10n (~2.3M paramètres), fine-tune depuis checkpoint COCO
- Baseline YOLOv8n, fine-tune depuis checkpoint COCO
- Quatre runs au total

## Expériences (toutes dans le notebook)
1. Courbes d'apprentissage (mAP val par epoch, 4 runs)
2. Évaluation finale (mAP50, mAP50:95, precision, recall)
3. Ablation NMS : end2end vs nms_forced (sweep IoU et conf), pour chaque régime
4. Écart end2end vs nms_forced : indicateur direct de la qualité de la one-to-one
5. Distribution des détections par image et des scores de confiance
6. mAP par classe et écart v10 vs v8 par classe
7. Latence d'inférence batch 1
8. Grille qualitative

## Livrable
- Notebook unique self-contained : notebooks/yolov10_reproduction.ipynb
- Pas de scripts externes
- Résultats sérialisés en JSON dans results/, figures en PNG

## Hors scope
- Training from scratch sur COCO
- TensorRT/ONNX
- Variantes s, m, l, x
