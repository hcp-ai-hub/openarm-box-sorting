import subprocess
import time
import json
import torch
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CONFIGS = [
    (0.98, 0.40),
    (0.98, 0.50),
    (0.99, 0.30),
    (0.99, 0.40),
    (0.99, 0.50),
    (0.995, 0.30),
    (0.995, 0.50),
    (1.0, 0.30),
    (1.0, 0.50),
    (1.0, 0.70),
]

STEPS_PER_RUN = 15000
RESULTS_FILE = "/home/soriano/lerobot/dropout_search_results.json"

def run_training(posvel_p, torque_p, run_id):
    output_dir = f"/home/soriano/lerobot/outputs/train/dropout_search_{run_id}"
    repo_id = f"Louislerikiki/dropout_search_{run_id}"
    cmd = [
        "lerobot-train",
        "--policy.type=act",
        "--dataset.repo_id=Louislerikiki/bimanual_sort_by_weight_v2",
        "--dataset.root=/home/soriano/lerobot/data/sort_by_weight_v2_multicam",
        "--dataset.video_backend=pyav",
        f"--output_dir={output_dir}",
        f"--job_name=dropout_search_{run_id}",
        "--policy.device=cuda",
        f"--policy.repo_id={repo_id}",
        f"--steps={STEPS_PER_RUN}",
        "--batch_size=2",
        "--save_freq=1000000",
        "--policy.chunk_size=100",
        "--policy.n_action_steps=100",
        "--policy.use_amp=true",
        f"--policy.state_posvel_dropout_prob={posvel_p}",
        f"--policy.state_torque_dropout_prob={torque_p}",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
    ]
    print(f"\n{'='*60}")
    print(f"RUN {run_id}: posvel_dropout={posvel_p}, torque_dropout={torque_p}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"Termine en {elapsed:.0f}s")
    if result.returncode != 0:
        print("ERREUR pendant le training:")
        print(result.stderr[-2000:])
        return None
    return output_dir

def measure_influence(checkpoint_dir, n_samples=100):
    policy = ACTPolicy.from_pretrained(f"{checkpoint_dir}/checkpoints/last/pretrained_model")
    policy.eval()
    policy.to("cuda")

    ds = LeRobotDataset(
        repo_id="Louislerikiki/bimanual_sort_by_weight_v2",
        root="/home/soriano/lerobot/data/sort_by_weight_v2_multicam",
        video_backend="pyav",
    )

    np.random.seed(42)
    indices = np.random.randint(0, len(ds), n_samples)

    groups = {
        "state": ["observation.state"],
        "top": ["observation.images.top_rgb", "observation.images.top_depth"],
        "wrist_right": ["observation.images.wrist_right_rgb", "observation.images.wrist_right_depth"],
        "wrist_left": ["observation.images.wrist_left_rgb", "observation.images.wrist_left_depth"],
    }
    results = {name: [] for name in groups}

    for idx in indices:
        sample = ds[int(idx)]
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            policy.reset()
            action_normal = policy.select_action(batch)
        for group_name, keys in groups.items():
            batch_noise = {k: v.clone() for k, v in batch.items()}
            for key in keys:
                tensor = batch_noise[key]
                batch_noise[key] = tensor + torch.randn_like(tensor) * tensor.std() * 0.20
            with torch.no_grad():
                policy.reset()
                action_noise = policy.select_action(batch_noise)
            diff = (action_normal - action_noise).abs().mean().item()
            results[group_name].append(diff)

    total = sum(np.mean(v) for v in results.values())
    pct = {name: 100 * np.mean(diffs) / total for name, diffs in results.items()}

    del policy
    torch.cuda.empty_cache()
    return pct

all_results = []
for i, (posvel_p, torque_p) in enumerate(CONFIGS):
    run_id = f"run{i}_pv{int(posvel_p*100)}_tq{int(torque_p*100)}"
    output_dir = run_training(posvel_p, torque_p, run_id)
    if output_dir is None:
        continue
    print(f"Mesure de l'influence pour {run_id}...")
    pct = measure_influence(output_dir)
    vision_pct = pct["top"] + pct["wrist_right"] + pct["wrist_left"]
    print(f"  state: {pct['state']:.1f}%  top: {pct['top']:.1f}%  wrist_r: {pct['wrist_right']:.1f}%  wrist_l: {pct['wrist_left']:.1f}%")
    print(f"  TOTAL VISION: {vision_pct:.1f}%")
    all_results.append({
        "run_id": run_id,
        "posvel_dropout": posvel_p,
        "torque_dropout": torque_p,
        "state_pct": pct["state"],
        "top_pct": pct["top"],
        "wrist_right_pct": pct["wrist_right"],
        "wrist_left_pct": pct["wrist_left"],
        "vision_total_pct": vision_pct,
    })
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

print("\n\n=== RESUME FINAL ===")
for r in sorted(all_results, key=lambda x: -x["vision_total_pct"]):
    print(f"{r['run_id']}: vision={r['vision_total_pct']:.1f}%  (state={r['state_pct']:.1f}%)")
