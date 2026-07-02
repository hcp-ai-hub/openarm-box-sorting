# Project History

> **Note**: All models in this project are proof-of-concept demonstrations tied to a specific hardware setup (robot, camera positions, workspace layout). Any change in setup will require retraining from scratch.

---

## Phase 0 — Preliminary Tasks (Single Camera, Fully Working)

These simpler tasks were completed successfully before tackling the sort-by-weight challenge.
They demonstrate that the hardware and pipeline work correctly end-to-end.
Demo videos: https://drive.google.com/drive/u/1/folders/1PsBtCrXj4AE_7-ZICalc3jr7PMX_zbeD

### Hardware
- Robot: 2x OpenArm follower, 7 DOF + gripper (can0=right, can1=left)
- Single top-down camera: `observation.images.up` (640x480, 30fps, /dev/video0)
- No weight sensor — detection via joint torque
- Gravity compensation always active

### Task 1 — Pick and Drop (single arm, left)
- **Model**: `Louislerikiki/pick_place_left_policy`
- Left arm only. Picks a box from random position (fixed orientation), drops into bin.
- 100k steps, ACT, n_action_steps=100
- **Result**: Fully working ✅

### Task 2 — Hand-to-Hand Transfer (bimanual)
- **Model**: `Louislerikiki/act_pick_and_drop_bimanual_v1`
- Fixed initial box position. One arm picks, transfers to the other arm mid-air.
- **Result**: Fully working ✅

### Task 3 — Sort by Weight (single arm, fixed position)
- **Model**: `Louislerikiki/act_sort_by_weight_v2`
- Single arm. Fixed box position. Sorts into correct bin based on weight (via joint torque).
- 100k steps, ACT, n_action_steps=100
- **Result**: Fully working ✅

### Eval parameters (Phase 0)
```bash
--policy.n_action_steps=100
--interpolation_multiplier=2
--robot.right_arm_config.max_relative_target=2
--dataset.video_backend=pyav
```

---

## Phase 1 — Sort by Weight Bimanual, Single Camera

### Hardware
- Same robot as Phase 0
- Single top-down camera: `observation.images.up` (640x480, 30fps, /dev/video0)

### Task
Bimanual sorting of cardboard boxes by weight. Random box positions. Hand-to-hand transfer when target bin is on the opposite side.

### Dataset
- [`Louislerikiki/bimanual_sort_by_weight_v1`](https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v1) — 101 episodes, 280854 frames, 30fps
- Local copy: `/home/soriano/lerobot/data/bimanual_sort_by_weight_v1_full`

### Models trained
| Version | Policy | Description | Result |
|---------|--------|-------------|--------|
| v1 | Diffusion Policy | Camera mishandled, too few steps | Abandoned |
| v2 | Diffusion Policy | Bug: normalization stats computed on 5 episodes instead of 101 | Explosive actions in eval |
| v3 | Diffusion Policy | Fixed normalization, 200k steps, DDPM training | Partially working |

### Eval parameters (Phase 1 — Diffusion Policy only)
```bash
--policy.num_inference_steps=10
--policy.noise_scheduler_type=DDIM
--interpolation_multiplier=2
--robot.right_arm_config.max_relative_target=2
--dataset.video_backend=pyav
```

---

## Phase 2 — Sort by Weight Bimanual, Multi-Camera (current)

### Hardware
- Same robot
- 6 cameras: ZED top rgb+depth (1280x720), 2x RealSense wrist rgb+depth (640x480)

### Task
Same as Phase 1, with random box positions and hand-to-hand transfer capability.

### Dataset
- [`Louislerikiki/bimanual_sort_by_weight_v2`](https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v2) — 100 episodes, ~249k frames, 30fps
- Local copy: `/home/soriano/lerobot/data/sort_by_weight_v2_multicam`

### Models trained

#### Main ACT models
| Model | Config | Steps | Physical result |
|-------|--------|-------|-----------------|
| [`act_sort_by_weight_multicam_v1`](https://huggingface.co/Louislerikiki/act_sort_by_weight_multicam_v1) | 6 cams, 1 decoder layer | 200k | Best overall compromise: good grasping + reliable hand-to-hand transfer |
| [`act_sort_by_weight_dropout_v1`](https://huggingface.co/Louislerikiki/act_sort_by_weight_dropout_v1) | 6 cams, 1 layer, 80% uniform state dropout | 1M | More precise grasping but slightly jerky, less reliable hand-to-hand |
| [`act_sort_by_weight_decoder4_louis`](https://huggingface.co/Louislerikiki/act_sort_by_weight_decoder4_louis) | 6 cams, 4 decoder layers | 150k | Improved attention on workspace vs borders |
| [`act_sort_by_weight_no_depth`](https://huggingface.co/Louislerikiki/act_sort_by_weight_no_depth) | RGB only (no depth), 4 layers | 150k | Similar to full model |
| [`act_sort_by_weight_wrist_rgb_only`](https://huggingface.co/Louislerikiki/act_sort_by_weight_wrist_rgb_only) | Wrist RGB only (no top cam), 4 layers | 150k | Fails: hand-to-hand in empty air, hits support |
| [`act_sort_by_weight_wrist_only`](https://huggingface.co/Louislerikiki/act_sort_by_weight_wrist_only) | Wrist rgb+depth only, 4 layers | 150k | Same failure as wrist_rgb_only |

#### State dropout experiments (not pushed to HuggingFace)
Grid search over posvel_dropout ∈ {0.70..1.0} × torque_dropout ∈ {0.1..0.7}.
Best config found: posvel=0.98, torque=0.40 → 5.5% vision sensitivity (vs 0.1% baseline).
Longer runs with this config degraded — vision use collapsed back toward zero.
See `scripts/dropout_search.py`.

#### Other experiments (pushed to HuggingFace, not recommended)
| Model | Description | Why not recommended |
|-------|-------------|---------------------|
| [`act_sort_by_weight_final_v1`](https://huggingface.co/Louislerikiki/act_sort_by_weight_final_v1) | Dropout 98%/40%, 250k steps | Vision use collapsed with longer training |
| [`act_sort_by_weight_full_dropout`](https://huggingface.co/Louislerikiki/act_sort_by_weight_full_dropout) | 100% posvel dropout, 100k steps | Poor results |
| [`act_sort_by_weight_remove_posvel`](https://huggingface.co/Louislerikiki/act_sort_by_weight_remove_posvel) | pos/vel permanently zeroed | Poor results |
| [`diffusion_sort_by_weight_v4`](https://huggingface.co/Louislerikiki/diffusion_sort_by_weight_v4) | Diffusion Policy 2D, 6 cams | Violent behavior (hits support), abandoned |

### Key findings
- Top RGB camera is essential — wrist-only models fail completely
- Depth cameras: comparable contribution to RGB (tested via attention analysis)
- State dropout helps short-term but vision use collapses with longer training
- 4 decoder layers: attention shifts from workspace borders to actual objects (confirmed via heatmaps)
- Diffusion Policy 2D: consistently violent across both datasets, definitively abandoned
- No automatic metric found correlating with physical grasp quality

### Eval parameters (Phase 2)
```bash
--policy.n_action_steps=100
--interpolation_multiplier=4
--robot.right_arm_config.max_relative_target=8
--dataset.video_backend=pyav
```
