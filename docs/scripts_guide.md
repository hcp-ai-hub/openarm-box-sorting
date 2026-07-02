# Scripts Guide

## dropout_search.py
Automatic grid search over state dropout configurations.
Runs 10-22 mini-trainings in sequence, measures vision/proprioception ratio for each config, saves results to JSON.

```bash
python3 scripts/dropout_search.py
# Results saved to: /home/soriano/lerobot/dropout_search_results.json
```

## auto_pipeline.py
Automated pipeline: waits for grid search to finish → selects best config → launches final training.

```bash
nohup python3 scripts/auto_pipeline.py > /tmp/auto_pipeline_log.txt 2>&1 &
```

## visualize_attention.py
Generates transformer attention heatmaps (decoder cross-attention) overlaid on camera images.
Allows visualizing where the model "looks" when making decisions.

```bash
# Frame 20, baseline model
python3 scripts/visualize_attention.py Louislerikiki/act_sort_by_weight_multicam_v1 20
# Output: /tmp/attention_top_rgb_frame20.jpg etc.
```

## chunk_fidelity_test.py
Measures fidelity of predicted action chunk vs real dataset actions.
Returns mean L1 score over N random samples.
Note: this metric does not reliably correlate with physical grasp quality.

```bash
python3 scripts/chunk_fidelity_test.py Louislerikiki/act_sort_by_weight_multicam_v1 100
```

## annotation_server.py
Flask web server for manually annotating task phases on dataset episodes.
Used to prepare camera importance labels for BFA (Best-Feature-Aware Fusion).

```bash
python3 scripts/annotation_server.py
# Open http://[IP]:5050 in browser
# Keys: T=top view, D=right wrist, G=left wrist, arrows=navigation
```

## annotate_phases.py
CLI version of the annotation tool (alternative to annotation_server.py).

## build_relative_dataset.py
Pre-computes per-episode starting proprioceptive states.
Used for the relative coordinates experiment (subtract initial state from observations).

```bash
python3 scripts/build_relative_dataset.py
# Output: /home/soriano/lerobot/episode_start_states.pt
```
