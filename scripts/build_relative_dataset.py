import pandas as pd
import glob
import numpy as np
import torch
import shutil
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE_ROOT = "/home/soriano/lerobot/data/sort_by_weight_v2_multicam"
DEST_ROOT = "/home/soriano/lerobot/data/sort_by_weight_v2_relative"

print("Lecture des metadonnees d'episodes...")
ep_files = sorted(glob.glob(f"{SOURCE_ROOT}/meta/episodes/chunk-000/*.parquet"))
episode_lengths = []
for f in ep_files:
    df = pd.read_parquet(f)
    episode_lengths.extend(df["length"].tolist())

# Calcule l'index global de la PREMIERE frame de chaque episode
episode_start_idx = {}
cumulative = 0
for ep_idx, length in enumerate(episode_lengths):
    episode_start_idx[ep_idx] = cumulative
    cumulative += length

print(f"Nombre d'episodes: {len(episode_lengths)}")
print(f"Total frames: {cumulative}")

print("Chargement du dataset source (lecture seule)...")
ds = LeRobotDataset(
    repo_id="Louislerikiki/bimanual_sort_by_weight_v2",
    root=SOURCE_ROOT,
    video_backend="pyav",
)

print("Pre-calcul des etats de depart par episode...")
episode_start_state = {}
for ep_idx, start_idx in episode_start_idx.items():
    sample = ds[start_idx]
    episode_start_state[ep_idx] = sample["observation.state"].clone()

print("Sauvegarde des offsets de depart par episode dans un fichier...")
torch.save(episode_start_state, "/home/soriano/lerobot/episode_start_states.pt")
print("Termine. Fichier sauvegarde: episode_start_states.pt")
print(f"Exemple episode 0, state de depart: {episode_start_state[0]}")
