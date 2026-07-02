# Teleoperation & Data Collection

> This workflow follows the official LeRobot tutorial: https://huggingface.co/docs/lerobot/en/il_robots

## Hardware
- **Leader arms**: 2x OpenArm leader (can2=right, can3=left)
- **Follower arms**: 2x OpenArm follower (can0=right, can1=left)
- The leader arms are held by the human operator and mirrored to the follower arms in real-time

## Recording episodes (lerobot-record with teleop)
```bash
lerobot-record \
  --robot.type=bi_openarm_follower \
  --robot.right_arm_config.port=can0 --robot.right_arm_config.side=right \
  --robot.right_arm_config.can_data_bitrate=5000000 \
  --robot.left_arm_config.port=can1 --robot.left_arm_config.side=left \
  --robot.left_arm_config.can_data_bitrate=5000000 \
  --robot.cameras="{ top_rgb: {type: zed, serial_number: 30357600, fps: 30, width: 1280, height: 720, return_depth: false}, top_depth: {type: zed, serial_number: 30357600, fps: 30, width: 1280, height: 720, return_depth: true}, wrist_right_rgb: {type: intelrealsense, serial_number_or_name: 218622278645, fps: 30, width: 640, height: 480, use_depth: false}, wrist_right_depth: {type: realsense_depth, serial_number_or_name: 218622278645, fps: 30, width: 640, height: 480}, wrist_left_rgb: {type: intelrealsense, serial_number_or_name: 218622270482, fps: 30, width: 640, height: 480, use_depth: false}, wrist_left_depth: {type: realsense_depth, serial_number_or_name: 218622270482, fps: 30, width: 640, height: 480}}" \
  --robot.id=my_bimanual_follower \
  --teleop.type=bi_openarm_leader \
  --teleop.right_arm_config.port=can2 \
  --teleop.right_arm_config.can_data_bitrate=5000000 \
  --teleop.left_arm_config.port=can3 \
  --teleop.left_arm_config.can_data_bitrate=5000000 \
  --teleop.id=my_bimanual_leader \
  --dataset.repo_id=Louislerikiki/bimanual_sort_by_weight_v2 \
  --dataset.root=/home/soriano/lerobot/data/sort_by_weight_v2_multicam \
  --dataset.single_task="Sort cardboard boxes by weight into the correct bin" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.streaming_encoding=false \
  --interpolation_multiplier=2 \
  --resume=true
```

## Keyboard controls during recording
- **Right arrow** → validate episode, move to next
- **Left arrow** → discard episode, redo
- **Escape** → stop recording session

## Important tips
- Always start with **closed grippers** (physically pinch them before launching)
- Motors use **relative encoders** — zero resets at every power-on based on physical joint position
- Starting position must be physically reproduced before each session
- Right arm j3 has a slight empirical offset that improves grasp success
- Never include gripper movement in reset scripts — manage manually
- Aim for **consistent demonstrations**: same trajectory, same speed, same starting position
- Bad episodes (collision, drop, wrong bin) should be discarded with left arrow immediately

## Uploading to HuggingFace
```bash
HF_HUB_DISABLE_XET=1 python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    repo_id='Louislerikiki/bimanual_sort_by_weight_v2',
    repo_type='dataset',
    folder_path='/home/soriano/lerobot/data/sort_by_weight_v2_multicam',
)
"
```

## Training after data collection
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=Louislerikiki/bimanual_sort_by_weight_v2 \
  --dataset.root=/home/soriano/lerobot/data/sort_by_weight_v2_multicam \
  --dataset.video_backend=pyav \
  --output_dir=/home/soriano/lerobot/outputs/train/my_model \
  --policy.device=cuda \
  --policy.repo_id=Louislerikiki/my_model \
  --steps=200000 \
  --batch_size=2 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.use_amp=true \
  --policy.push_to_hub=true \
  --wandb.enable=false
```

## State vector structure
48 dimensions = (pos, vel, torque) × 8 joints × 2 arms
- Torque indices: every 3rd element starting at offset 2 (indices 2,5,8,11,...,47)
- right arm: indices 0-23, left arm: indices 24-47
- Gripper: indices 21 (right) and 45 (left)

## Technical notes
- Stack: LeRobot 0.5.2, Python 3.12
- `torchcodec` is broken — always use `--dataset.video_backend=pyav`
- HuggingFace upload requires `HF_HUB_DISABLE_XET=1` prefix
- Gravity compensation is always active on the follower arms
- `--dataset.streaming_encoding=false` recommended for stability
