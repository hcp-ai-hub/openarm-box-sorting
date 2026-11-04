# OpenArm Box Sorting — Bimanual Imitation Learning

Real-robot imitation learning on a bimanual 7-DoF OpenArm: a full record → train → evaluate pipeline built from a bare, unwired kit, and a head-to-head comparison of **ACT, Diffusion Policy, VQ-BeT and π0** on the same hardware and the same task.

**▶ Demo videos:** [ACT policy (main result)](https://huggingface.co/Louislerikiki/act_sort_by_weight_multicam_v1) · [VQ-BeT failure mode](https://huggingface.co/Louislerikiki/vqbet_sort_by_weight_topview_v1) · [full video archive](https://drive.google.com/drive/folders/1PsBtCrXj4AE_7-ZICalc3jr7PMX_zbeD)

## Author

**Louis Barthez** ([@LouisBarthez](https://github.com/LouisBarthez)): system setup, data collection, training and evaluation code, developed during a 16-week internship at AIST Tokyo Waterfront (Apr–Jul 2026), supervised by Shunki Itadera.

*Commits were pushed from a shared lab workstation (em-sor account).*

## Task
Bimanual sorting of cardboard boxes by weight using imitation learning.
When the target bin is on the opposite side, a hand-to-hand transfer is performed between the two arms. Weight is inferred from joint torque during the grasp (no dedicated sensor).

## What was built
- **Hardware bring-up from scratch:** CAN FD wiring (4 buses, 1 / 5 Mbit/s), motor ID assignment, calibration of 2 leader + 2 follower arms.
- **Teleoperation:** unilateral and bilateral (force-feedback) leader–follower control, first with low-level CAN scripts, then ported to LeRobot.
- **Sensing:** 6 synchronized streams (ZED top RGB-D + 2 RealSense wrist RGB-D) on custom 3D-printed camera mounts.
- **Data:** 100 teleoperated episodes, ~249k frames at 30 fps, published on Hugging Face.
- **Training & deployment** of 4 policy families on 4× RTX 6000 Ada, including π0 (~4B params) in reduced precision, which required patching a chain of dtype mismatches in the forward pass.
- **Offline diagnostic tooling** to test a policy without the robot (see below).

## Results

| Policy | Mechanism | Behaviour on the real robot |
|--------|-----------|-----------------------------|
| **ACT** | CVAE, action chunking | ✅ Completes the full task: finds the box, hand-to-hand transfer when needed, correct bin in most trials. Grasp alignment remains imprecise. |
| Diffusion Policy | Iterative denoising | ❌ Erratic, bursty motion, no directed approach |
| VQ-BeT | Discrete action codes (VQ-VAE) | ❌ Same erratic pattern |
| π0 (~4B, fine-tuned) | VLA, flow matching | ❌ Same erratic pattern |

Pipeline validation tasks (single camera, ACT, 100k steps) all worked reliably: single-arm pick & drop, fixed-position hand-to-hand transfer, single-arm sort by weight.

### ACT variants (Hugging Face)
| Model | Description | Physical result |
|-------|-------------|-----------------|
| `act_sort_by_weight_multicam_v1` | ACT baseline, 6 cams, 1 decoder layer | Best overall compromise: good grasping + reliable hand-to-hand transfer |
| `act_sort_by_weight_dropout_v1` | ACT + 80% uniform state dropout | More precise grasping but slightly jerky, less reliable hand-to-hand transfer |
| `act_sort_by_weight_decoder4_louis` | ACT, 6 cams, 4 decoder layers | Improved attention focus on workspace objects vs borders |
| `act_sort_by_weight_no_depth` | ACT, RGB only (no depth) | Similar behaviour to full model |
| `act_sort_by_weight_25ep / 50ep / 75ep` | Episode-count ablation | Fewer episodes → more erratic, less reliable transfer |

## Failure analysis

Three architecturally very different generative policies failing *the same way* pointed to a shared cause, so hypotheses were tested one by one:

| Hypothesis | Test | Result |
|------------|------|--------|
| Too much / too varied data | ACT retrained on 25 / 50 / 75 of 100 episodes | ❌ Ruled out: fewer episodes made it worse |
| Normalization mismatch | Policy stats vs. dataset stats | ❌ Ruled out: exact match |
| Camera drift between sessions | Pixel diff of first frames (early / mid / late episodes) | ❌ Ruled out: negligible |
| Robot latency / control loop | **Dry-run replay**: real recorded episode fed offline to the policy, no robot | ❌ Ruled out: the jumps happen inside the model |
| Weights not actually loaded | Generic loader vs. policy-specific `from_pretrained` | ⚠️ Real bug: generic helper silently skipped trained weights; all diagnostics re-run after the fix |

**Dry-run frame-to-frame coherence** on the same 150-frame real episode: ACT never exceeds the jump threshold; VQ-BeT exceeds it on **18 / 150 frames (12%)**, so the erratic behaviour originates in the model, not the hardware.

For ACT, the predicted reach target correlates only weakly (and with inconsistent sign across joints) with the true target before the grasp, so the imprecision is a visual-to-target mapping problem, not noise around a correct direction.

**Open leads:** explicit camera exposure control (auto-exposure was never fixed), adding force/torque to the policy input, and more constrained action representations to handle multimodal demonstrations.

## Dataset
- Hugging Face: [`Louislerikiki/bimanual_sort_by_weight_v2`](https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v2)
- 100 episodes, ~249k frames, 30 fps
- 6 camera streams: top_rgb, top_depth, wrist_right_rgb, wrist_right_depth, wrist_left_rgb, wrist_left_depth

## Hardware
- Robot: 2× OpenArm follower (+ 2 leaders), 7 DoF + gripper (can0=right, can1=left)
- Cameras: ZED top view (1280×720), 2× RealSense wrist (640×480 RGB + depth)

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

## Docs
- [Teleoperation](docs/teleoperation.md)
- [Scripts guide](docs/scripts_guide.md)
- [Operational notes](docs/operational_notes.md)
- [Project history](docs/project_history.md) — previous single-camera setup and experiments

## Links
- Models: https://huggingface.co/Louislerikiki
- Dataset: https://huggingface.co/datasets/Louislerikiki/bimanual_sort_by_weight_v2
