# OpenArm Box Sorting — Bimanual Imitation Learning

## Task
Bimanual sorting of cardboard boxes by weight using imitation learning (ACT).
When the target bin is on the opposite side, a hand-to-hand transfer is performed between the two arms.

## Hardware
- Robot: 2x OpenArm follower, 7 DOF + gripper (can0=right, can1=left)
- Cameras: ZED top view (1280x720), 2x RealSense wrist (640x480 rgb+depth)
- Weight detection: via joint torque (no dedicated sensor)

## Dataset
- HuggingFace: `Louislerikiki/bimanual_sort_by_weight_v2`
- 100 episodes, ~249k frames, 30fps
- 6 camera streams: top_rgb, top_depth, wrist_right_rgb, wrist_right_depth, wrist_left_rgb, wrist_left_depth

## Models (HuggingFace)
| Model | Description | Notes |
|-------|-------------|-------|
| `act_sort_by_weight_multicam_v1` | ACT baseline, 6 cams, 1 decoder layer | Reference model |
| `act_sort_by_weight_dropout_v1` | ACT + 80% uniform state dropout | Best hand-to-hand transfer observed |
| `act_sort_by_weight_decoder4_louis` | ACT, 6 cams, 4 decoder layers | Improved attention focus on workspace |
| `act_sort_by_weight_no_depth` | ACT, RGB only (no depth) | Similar to full model |

## Stack
- LeRobot 0.5.2, Python 3.12, PyTorch
- Eval: `--policy.n_action_steps=100 --interpolation_multiplier=4`

## Scripts
- `dropout_search.py` — Grid search over state dropout configurations
- `auto_pipeline.py` — Automated pipeline: search → select best → train final
- `visualize_attention.py` — Transformer attention heatmap visualization
- `chunk_fidelity_test.py` — Chunk-level action fidelity measurement
- `annotation_server.py` — Web tool for phase annotation (BFA preparation)
- `annotate_phases.py` — CLI annotation tool
- `build_relative_dataset.py` — Compute per-episode relative state offsets

## Demo Videos
Available on Google Drive: https://drive.google.com/drive/u/1/folders/1PsBtCrXj4AE_7-ZICalc3jr7PMX_zbeD

## HuggingFace
Dataset: https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v2
Models: https://huggingface.co/Louislerikiki

## See also
- [Project History](docs/project_history.md) — previous single-camera setup and experiments
