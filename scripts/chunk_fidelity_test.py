import torch
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import sys

MODEL_REPO = sys.argv[1]
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
CHUNK_SIZE = 100

ds = LeRobotDataset(
    repo_id='Louislerikiki/bimanual_sort_by_weight_v2',
    root='/home/soriano/lerobot/data/sort_by_weight_v2_multicam',
    video_backend='pyav',
)

policy = ACTPolicy.from_pretrained(MODEL_REPO)
policy.eval()
policy.to('cuda')

np.random.seed(42)
max_start = ds.num_frames - CHUNK_SIZE - 1
indices = np.random.randint(0, max_start, N_SAMPLES)

all_l1 = []
for start_idx in indices:
    sample = ds[int(start_idx)]
    batch = {k: v.unsqueeze(0).to('cuda') for k, v in sample.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        policy.reset()
        predicted_chunk = policy.predict_action_chunk(batch).squeeze(0).cpu()

    real_chunk = []
    for i in range(predicted_chunk.shape[0]):
        real_chunk.append(ds[int(start_idx) + i]['action'])
    real_chunk = torch.stack(real_chunk)

    l1 = torch.nn.functional.l1_loss(predicted_chunk, real_chunk).item()
    all_l1.append(l1)

print(f'=== {MODEL_REPO} ===')
print(f'Chunk fidelity L1 (sur {N_SAMPLES} echantillons, chunk de {predicted_chunk.shape[0]}): {np.mean(all_l1):.4f} (std: {np.std(all_l1):.4f})')
