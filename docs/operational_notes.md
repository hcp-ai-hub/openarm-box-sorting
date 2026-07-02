# Operational Notes — Things to Know Before Running the Robot

## Critical: Start with closed grippers
Always start evaluation with grippers physically closed (pinch them manually before launching).
The robot uses relative encoders that reset at power-on — starting position must be reproduced physically each time.

## Camera setup
- ZED top camera: shared instance between rgb and depth streams (same physical camera)
- RealSense wrist cameras: identified by serial number, not device path
  - Right wrist: `218622278645`
  - Left wrist: `218622270482`
- If ZED fails with `CAMERA NOT DETECTED`: unplug and replug the USB cable, wait 5 seconds

## CAN bus
- can0 = right follower arm
- can1 = left follower arm
- can2 = right leader arm
- can3 = left leader arm
- If CAN bus fails after reboot: `sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on` (same for can1/2/3)

## Joint offsets
- Right arm j3 has a slight offset at start — empirically improves grasp success
- Never touch grippers in reset scripts (manage manually)

## Eval parameters (Phase 2 — multi-camera ACT)
```bash
--policy.n_action_steps=100
--interpolation_multiplier=4
--robot.right_arm_config.max_relative_target=8
--dataset.video_backend=pyav
```

## Eval parameters (Phase 1 — single camera Diffusion Policy)
```bash
--policy.num_inference_steps=10
--policy.noise_scheduler_type=DDIM
--interpolation_multiplier=2
--robot.right_arm_config.max_relative_target=2
--dataset.video_backend=pyav
```

## Known issues
- `max_relative_target` must be omitted or set high enough for faithful evaluation — too low slows the robot excessively
- Diffusion Policy 2D: violent behavior observed (hits support structure), not recommended
- `HF_HUB_DISABLE_XET=1` required for HuggingFace uploads
- A residual warmup patch in `lerobot_record.py` hardcoded `/dev/video0` — removed, do not re-add

## Demo videos (Google Drive)
Previous models and task demonstrations: https://drive.google.com/drive/u/1/folders/1PsBtCrXj4AE_7-ZICalc3jr7PMX_zbeD

Videos include:
- Phase 1 models (single camera, Diffusion Policy v1/v2/v3)
- Phase 2 models (multi-camera ACT, various configurations)
- Hand-to-hand transfer demonstrations
