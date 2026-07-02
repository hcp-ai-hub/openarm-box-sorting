import subprocess
import time
import json
import torch
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SEARCH_LOG = "/tmp/dropout_search_log2.txt"
RESULTS_FILE = "/home/soriano/lerobot/dropout_search_results.json"
BATCH1_FILE = "/home/soriano/lerobot/dropout_search_results_batch1.json"
FINAL_TRAIN_LOG = "/tmp/final_train_log.txt"

def wait_for_search_to_finish():
    print("En attente de la fin de la recherche de dropout (batch2)...")
    while True:
        result = subprocess.run(["pgrep", "-f", "dropout_search.py"], capture_output=True, text=True)
        if result.stdout.strip() == "":
            print("Recherche terminee.")
            break
        time.sleep(60)

def compute_trajectory_fidelity(checkpoint_dir, n_samples=100):
    """Mesure la fidelite de trajectoire : pour des frames aleatoires du dataset,
    on donne au modele l'observation de depart (vision + state) et on compare
    son action predite a l'action reellement enregistree (verite terrain)."""
    policy = ACTPolicy.from_pretrained(f"{checkpoint_dir}/checkpoints/last/pretrained_model")
    policy.eval()
    policy.to("cuda")

    ds = LeRobotDataset(
        repo_id="Louislerikiki/bimanual_sort_by_weight_v2",
        root="/home/soriano/lerobot/data/sort_by_weight_v2_multicam",
        video_backend="pyav",
    )

    np.random.seed(123)
    indices = np.random.randint(0, len(ds), n_samples)

    errors = []
    for idx in indices:
        sample = ds[int(idx)]
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            policy.reset()
            pred_action = policy.select_action(batch)
        true_action = batch.get("action")
        if true_action is not None:
            err = (pred_action - true_action).abs().mean().item()
            errors.append(err)

    del policy
    torch.cuda.empty_cache()
    return float(np.mean(errors)) if errors else None

def select_best_config():
    with open(RESULTS_FILE) as f:
        batch2 = json.load(f)

    # Batch1 checkpoints were deleted to free disk space, so we only consider
    # batch2 configs for which checkpoints are actually still on disk.
    all_results = batch2
    all_results.sort(key=lambda x: -x["vision_total_pct"])

    print("\n=== TOP 5 candidats par vision% ===")
    for r in all_results[:5]:
        print(f"  {r['run_id']}: vision={r['vision_total_pct']:.2f}%  state={r['state_pct']:.2f}%")

    print("\nEvaluation de la loss sur frames de transfert pour le top 5...")
    candidates_with_transfer = []
    for r in all_results[:5]:
        checkpoint_dir = f"/home/soriano/lerobot/outputs/train/dropout_search_{r['run_id']}"
        import os
        if not os.path.exists(checkpoint_dir):
            print(f"  {r['run_id']}: checkpoint introuvable (deja nettoye), skip")
            continue
        transfer_loss = compute_trajectory_fidelity(checkpoint_dir)
        print(f"  {r['run_id']}: vision={r['vision_total_pct']:.2f}%  transfer_loss={transfer_loss}")
        r["transfer_loss"] = transfer_loss
        candidates_with_transfer.append(r)

    valid_candidates = [r for r in candidates_with_transfer if r.get("transfer_loss") is not None]
    if not valid_candidates:
        print("Aucun candidat valide avec transfer_loss mesurable, on prend le meilleur vision% brut.")
        return all_results[0]

    # Score combine : on veut vision haute ET transfer_loss basse.
    # Normalisation simple puis somme ponderee.
    max_vision = max(r["vision_total_pct"] for r in valid_candidates)
    max_tloss = max(r["transfer_loss"] for r in valid_candidates)
    for r in valid_candidates:
        norm_vision = r["vision_total_pct"] / max_vision if max_vision > 0 else 0
        norm_tloss = 1 - (r["transfer_loss"] / max_tloss if max_tloss > 0 else 0)
        r["combined_score"] = 0.5 * norm_vision + 0.5 * norm_tloss

    valid_candidates.sort(key=lambda x: -x["combined_score"])
    print("\n=== CLASSEMENT FINAL (score combine vision + qualite transfert) ===")
    for r in valid_candidates:
        print(f"  {r['run_id']}: score={r['combined_score']:.3f}  vision={r['vision_total_pct']:.2f}%  transfer_loss={r['transfer_loss']:.4f}")

    return valid_candidates[0]

def launch_final_training(best_config):
    import os
    import shutil
    output_dir = "/home/soriano/lerobot/outputs/train/act_sort_by_weight_final_v1"
    if os.path.exists(output_dir):
        print(f"Suppression de l'ancien dossier residuel: {output_dir}")
        shutil.rmtree(output_dir)

    posvel_p = best_config["posvel_dropout"]
    torque_p = best_config["torque_dropout"]
    print(f"\n=== LANCEMENT TRAINING FINAL ===")
    print(f"posvel_dropout={posvel_p}  torque_dropout={torque_p}")

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        "--dataset.repo_id=Louislerikiki/bimanual_sort_by_weight_v2",
        "--dataset.root=/home/soriano/lerobot/data/sort_by_weight_v2_multicam",
        "--dataset.video_backend=pyav",
        "--output_dir=/home/soriano/lerobot/outputs/train/act_sort_by_weight_final_v1",
        "--job_name=act_sort_by_weight_final_v1",
        "--policy.device=cuda",
        "--policy.repo_id=Louislerikiki/act_sort_by_weight_final_v1",
        "--steps=250000",
        "--batch_size=2",
        "--save_freq=50000",
        "--policy.chunk_size=100",
        "--policy.n_action_steps=1",
        "--policy.use_amp=true",
        f"--policy.state_posvel_dropout_prob={posvel_p}",
        f"--policy.state_torque_dropout_prob={torque_p}",
        "--policy.temporal_ensemble_coeff=0.01",
        "--wandb.enable=false",
        "--policy.push_to_hub=true",
    ]

    import os
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(FINAL_TRAIN_LOG, "w") as logf:
        subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    print(f"Training final lance. Suivre avec: tail -f {FINAL_TRAIN_LOG}")

if __name__ == "__main__":
    wait_for_search_to_finish()
    time.sleep(10)
    best = select_best_config()
    with open("/home/soriano/lerobot/best_config_chosen.json", "w") as f:
        json.dump(best, f, indent=2)
    launch_final_training(best)
    print("\nPipeline complet. Le training final tourne maintenant en arriere-plan.")
