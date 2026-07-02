import torch
import numpy as np
import cv2
import sys
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MODEL_REPO = "Louislerikiki/act_sort_by_weight_multicam_v1"

CAMERA_BLOCKS = [
    ("observation.images.top_rgb", 23, 40, 2),
    ("observation.images.top_depth", 23, 40, 922),
    ("observation.images.wrist_right_rgb", 15, 20, 1842),
    ("observation.images.wrist_right_depth", 15, 20, 2142),
    ("observation.images.wrist_left_rgb", 15, 20, 2442),
    ("observation.images.wrist_left_depth", 15, 20, 2742),
]

policy = ACTPolicy.from_pretrained(MODEL_REPO)
policy.eval()
policy.to("cuda")

attention_weights = {}

def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
        attention_weights["weights"] = output[1].detach().cpu()

hook = policy.model.decoder.layers[0].multihead_attn.register_forward_hook(hook_fn)

ds = LeRobotDataset(
    repo_id="Louislerikiki/bimanual_sort_by_weight_v2",
    root="/home/soriano/lerobot/data/sort_by_weight_v2_multicam",
    video_backend="pyav",
)

frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 500
sample = ds[frame_idx]
batch = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if isinstance(v, torch.Tensor)}

with torch.no_grad():
    policy.reset()
    action = policy.select_action(batch)

hook.remove()

attn = attention_weights["weights"][0]  # (n_queries=100, n_tokens=3042)
attn_mean = attn.mean(dim=0).numpy()  # moyenne sur les 100 queries d'action -> (3042,)

print(f"Attention totale captee: {attn_mean.sum():.4f} (devrait etre proche de 1.0)")
print()

total_attn_by_cam = {}
for name, h, w, start in CAMERA_BLOCKS:
    n_tok = h * w
    block = attn_mean[start:start+n_tok]
    total_attn_by_cam[name] = block.sum()
    print(f"{name}: attention totale = {block.sum():.4f} ({100*block.sum():.1f}%)")

print()
print(f"Attention sur state (token 1): {attn_mean[1]:.4f} ({100*attn_mean[1]:.1f}%)")
print(f"Attention sur latent (token 0): {attn_mean[0]:.4f} ({100*attn_mean[0]:.1f}%)")

# Genere les heatmaps superposees pour chaque camera
for name, h, w, start in CAMERA_BLOCKS:
    n_tok = h * w
    block = attn_mean[start:start+n_tok].reshape(h, w)
    block_norm = (block - block.min()) / (block.max() - block.min() + 1e-8)
    heatmap = cv2.resize(block_norm, (640, 480) if "wrist" in name else (1280, 720))
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    img_tensor = sample[name]
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (heatmap_color.shape[1], heatmap_color.shape[0]))

    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
    cam_short = name.split(".")[-1]
    out_path = f"/tmp/attention_{cam_short}_frame{frame_idx}.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"Sauvegarde: {out_path}")
