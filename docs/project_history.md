# Project History

## Phase 1 — Single Camera Setup (before v2 dataset)

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

### Models trained
| Model | HuggingFace | Description | Physical eval result |
|-------|-------------|-------------|----------------------|
| `act_sort_by_weight_multicam_v1` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_multicam_v1) | ACT baseline, 6 cams, 1 decoder layer, 200k steps | Grasps boxes, no hand-to-hand |
| `act_sort_by_weight_dropout_v1` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_dropout_v1) | ACT + 80% uniform state dropout, 1M steps | Best result: grasps + occasional hand-to-hand |
| `act_sort_by_weight_decoder4_louis` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_decoder4_louis) | ACT, 6 cams, 4 decoder layers, 150k steps | Improved attention on workspace |
| `act_sort_by_weight_no_depth` | [link](https://huggingface.co/Louislerikiki/act_sort_by_weight_no_depth) | ACT, RGB only (no depth streams), 150k steps | Similar to full model |

### Eval command (Phase 2)
```bash
lerobot-record \
  --policy.n_action_steps=100 \
  --interpolation_multiplier=4 \
  --robot.right_arm_config.max_relative_target=8
```

### Key findings
- Top RGB camera essential for global localization — wrist-only models fail completely
- Depth cameras: comparable contribution to RGB (contrary to initial intuition)
- State dropout forces more vision use but degrades at longer training
- 4 decoder layers: attention shifts from workspace borders to actual objects
- Diffusion Policy 2D: violent behavior, abandoned
- No automatic metric found that correlates with physical grasp success
