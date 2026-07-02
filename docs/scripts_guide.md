# Scripts Guide

## dropout_search.py
Grid search automatique sur les configurations de state dropout.
Lance 10-22 mini-trainings en séquence, mesure le ratio vision/proprioception pour chaque config, sauvegarde les résultats en JSON.

```bash
python3 scripts/dropout_search.py
# Résultats dans: /home/soriano/lerobot/dropout_search_results.json
```

## auto_pipeline.py
Pipeline automatique : attend la fin de la grid search → sélectionne la meilleure config → lance le training final.

```bash
nohup python3 scripts/auto_pipeline.py > /tmp/auto_pipeline_log.txt 2>&1 &
```

## visualize_attention.py
Génère des heatmaps d'attention du transformer (cross-attention decoder) superposées sur les images des caméras.
Permet de visualiser où le modèle "regarde" pour prendre ses décisions.

```bash
# Frame 20, modele baseline
python3 scripts/visualize_attention.py Louislerikiki/act_sort_by_weight_multicam_v1 20
# Output: /tmp/attention_top_rgb_frame20.jpg etc.
```

## chunk_fidelity_test.py
Mesure la fidélité du chunk d'actions prédit par rapport aux actions réelles du dataset.
Donne un score L1 moyen sur N échantillons aléatoires.
Note: cette métrique ne corrèle pas de façon fiable avec la qualité physique réelle du grasp.

```bash
python3 scripts/chunk_fidelity_test.py Louislerikiki/act_sort_by_weight_multicam_v1 100
```

## annotation_server.py
Serveur web Flask pour annoter manuellement les phases de la tâche sur les épisodes du dataset.
Utilisé pour préparer les labels d'importance des caméras pour BFA (Best-Feature-Aware Fusion).

```bash
python3 scripts/annotation_server.py
# Ouvre http://[IP]:5050 dans le navigateur
# Touches: T=top view, D=wrist droit, G=wrist gauche, flèches=navigation
```

## annotate_phases.py
Version CLI de l'outil d'annotation (alternative à annotation_server.py).

## build_relative_dataset.py
Pré-calcule les états proprioceptifs de départ pour chaque épisode.
Utilisé pour l'expérience de coordonnées relatives (soustraire la position initiale du state).

```bash
python3 scripts/build_relative_dataset.py
# Output: /home/soriano/lerobot/episode_start_states.pt
```
