# Project History

## Phase 1 — Single Camera Setup

### Hardware
- Robot: 2x OpenArm follower, 7 DOF + gripper (can0=right, can1=left)
- Single top-down camera: `observation.images.up` (640x480, 30fps, /dev/video0)
- No weight sensor — detection via joint torque
- Gravity compensation always active

### Task
Bimanual sorting of cardboard boxes by weight into two bins on either side of the workspace.
Boxes in random positions. Hand-to-hand transfer when target bin is on the opposite side.

### Dataset
- `Louislerikiki/bimanual_sort_by_weight_v1` — 101 episodes, 280854 frames, 30fps
- Local copy: `/home/soriano/lerobot/data/bimanual_sort_by_weight_v1_full`

### Models trained
| Model | Description | Result |
|-------|-------------|--------|
| v1 | Diffusion Policy, camera mishandled, too few steps | Abandoned |
| v2 | Bug: normalization stats computed on 5 episodes instead of 101 | Explosive actions in eval |
| v3 | Fix normalization bug, 200k steps, DDPM training | Partially working |

### Eval notes (Phase 1)
- Always use `--policy.num_inference_steps=10 --policy.noise_scheduler_type=DDIM`
- `--interpolation_multiplier=2`
- `--robot.right_arm_config.max_relative_target=2`
- `--dataset.video_backend=pyav` (torchcodec broken)

---

## Phase 2 — Multi-Camera Setup (current)

### Hardware
- Same robot
- 6 cameras: ZED top rgb+depth (1280x720), 2x RealSense wrist rgb+depth (640x480)

### Dataset
- `Louislerikiki/bimanual_sort_by_weight_v2` — 100 episodes, ~249k frames, 30fps
- HuggingFace: https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v2
- Local copy: `/home/soriano/lerobot/data/sort_by_weight_v2_multicam`

### Models trained

#### ACT baseline and ablations
| Model | HuggingFace | Config | Steps | Physical result |
|-------|-------------|--------|-------|-----------------|
| `act_sort_by_weight_multicam_v1` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_multicam_v1) | 6 cams, 1 layer, no dropout | 200k | Grasps boxes, performs hand-to-hand transfer |
| `act_sort_by_weight_dropout_v1` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_dropout_v1) | 6 cams, 1 layer, 80% uniform state dropout | 1M | Best result: grasps + occasional hand-to-hand |
| `act_sort_by_weight_decoder4_louis` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_decoder4_louis) | 6 cams, 4 decoder layers | 150k | Improved attention on workspace vs borders |
| `act_sort_by_weight_no_depth` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_no_depth) | RGB only (no depth), 4 layers | 150k | Similar to full model |
| `act_sort_by_weight_wrist_rgb_only` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_wrist_rgb_only) | Wrist RGB only (no top cam), 4 layers | 150k | Fails: hand-to-hand in empty air, hits support |
| `act_sort_by_weight_wrist_only` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_wrist_only) | Wrist rgb+depth only, 4 layers | 150k | Same failure as wrist_rgb_only |

#### State dropout grid search (all 1 layer, 15k steps each, not pushed to HF)
Searched over posvel_dropout ∈ {0.70, 0.85, 0.95, 0.98, 0.99, 0.995, 1.0} × torque_dropout ∈ {0.1..0.7}.
Best config: posvel=0.98, torque=0.40 → 5.5% vision sensitivity (vs 0.1% baseline).
See `scripts/dropout_search.py` for implementation.

#### Longer dropout runs (all pushed to HF)
| Model | HuggingFace | Config | Result |
|-------|-------------|--------|--------|
| `act_sort_by_weight_final_v1` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_final_v1) | posvel=0.98, torque=0.40, 1 layer | 250k | Worse than short run — vision use collapsed |
| `act_sort_by_weight_full_dropout` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_full_dropout) | posvel=1.0, torque=0.0, 1 layer | 100k | Poor — temporal ensembling bug + slow |
| `act_sort_by_weight_remove_posvel` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_remove_posvel) | pos/vel permanently zeroed, 1 layer | 100k | Poor — same issue |

#### Diffusion Policy attempts
| Model | Config | Result |
|-------|--------|--------|
| `diffusion_sort_by_weight_v4` | 2D Diffusion, 6 cams, resize 480x640 | Violent behavior (hits support), abandoned |

### Key findings
- Top RGB camera is essential for global localization — wrist-only models fail completely
- Depth cameras: comparable contribution to RGB (contrary to initial intuition from supervisor)
- State dropout helps short-term but vision use collapses with longer training
- 4 decoder layers: attention shifts from workspace borders to actual objects (confirmed via attention heatmaps)
- Diffusion Policy 2D: consistently violent behavior across datasets, abandoned definitively
- No automatic metric found that correlates with physical grasp success (tested: L1 loss, chunk fidelity, attention sensitivity)
- The only reliable quality measure remains physical evaluation on the robot

### Eval command (Phase 2)
```bash
lerobot-record \
  --policy.n_action_steps=100 \
  --interpolation_multiplier=4 \
  --robot.right_arm_config.max_relative_target=8 \
  --dataset.video_backend=pyav
```

---

## Previous Tasks (Phase 0) — Single Camera, Fully Working

These tasks were completed successfully before moving to the sort-by-weight challenge.
Demo videos available on Google Drive: https://drive.google.com/drive/u/1/folders/1PsBtCrXj4AE_7-ZICalc3jr7PMX_zbeD

### Hardware (same for all)
- Robot: 2x OpenArm follower, 7 DOF + gripper (can0=right, can1=left)
- Single top-down camera: `observation.images.up` (640x480, 30fps)
- LeRobot 0.5.2, `--dataset.video_backend=pyav`

### Task 1 — Pick and Drop (single arm, left)
- **Model**: `Louislerikiki/pick_place_left_policy`
- **Description**: Left arm only. Picks a box from random position, drops it into a bin. Box always same orientation.
- **Result**: 100% functional ✅
- **Eval command**:
```bash
lerobot-record \
  --policy.path=Louislerikiki/pick_place_left_policy \
  --robot.type=bi_openarm_follower \
  --robot.cameras="{up: {type: opencv, index_or_path: /dev/video0, fps: 30, width: 640, height: 480}}" \
  --policy.n_action_steps=100 \
  --interpolation_multiplier=2 \
  --robot.right_arm_config.max_relative_target=2
```

### Task 2 — Hand-to-Hand Transfer (bimanual)
- **Model**: `Louislerikiki/act_pick_and_drop_bimanual_v1`
- **Description**: Bimanual task. Fixed initial box position. One arm picks, transfers to the other arm mid-air.
- **Result**: 100% functional ✅
- **Eval command**:
```bash
lerobot-record \
  --policy.path=Louislerikiki/act_pick_and_drop_bimanual_v1 \
  --robot.type=bi_openarm_follower \
  --robot.cameras="{up: {type: opencv, index_or_path: /dev/video0, fps: 30, width: 640, height: 480}}" \
  --policy.n_action_steps=100 \
  --interpolation_multiplier=2 \
  --robot.right_arm_config.max_relative_target=2
```

### Task 3 — Sort by Weight (single arm, fixed position)
- **Model**: `Louislerikiki/act_sort_by_weight_v2`
- **Description**: Single arm. Fixed box position. Sorts into correct bin based on weight (detected via joint torque).
- **Result**: 100% functional ✅
- **Eval command**:
```bash
lerobot-record \
  --policy.path=Louislerikiki/act_sort_by_weight_v2 \
  --robot.type=bi_openarm_follower \
  --robot.cameras="{up: {type: opencv, index_or_path: /dev/video0, fps: 30, width: 640, height: 480}}" \
  --policy.n_action_steps=100 \
  --interpolation_multiplier=2 \
  --robot.right_arm_config.max_relative_target=8
```
