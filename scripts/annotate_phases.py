import cv2
import json
import glob
import pandas as pd
import sys

DATASET_ROOT = "/home/soriano/lerobot/data/sort_by_weight_v2_multicam"
OUTPUT_FILE = "/home/soriano/lerobot/phase_annotations.json"

ep_files = sorted(glob.glob(f"{DATASET_ROOT}/meta/episodes/chunk-000/*.parquet"))
episode_lengths = []
for f in ep_files:
    df = pd.read_parquet(f)
    episode_lengths.extend(df["length"].tolist())

print(f"Nombre d'episodes a annoter: {len(episode_lengths)}")
print()
print("INSTRUCTIONS:")
print("  T = phase TOP dominante (approche globale)")
print("  D = phase WRIST DROIT dominante (grasp main droite)")
print("  G = phase WRIST GAUCHE dominante (grasp main gauche)")
print("  ESPACE = pause/lecture")
print("  Q = episode suivant (sans terminer celui-ci)")
print("  ECHAP = tout arreter et sauvegarder")
print()

annotations = {}
try:
    with open(OUTPUT_FILE) as f:
        annotations = json.load(f)
    print(f"Annotations existantes: {len(annotations)} episodes deja faits")
except FileNotFoundError:
    pass

start_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0

for ep_idx in range(start_ep, len(episode_lengths)):
    if str(ep_idx) in annotations:
        continue

    video_path = f"{DATASET_ROOT}/videos/observation.images.top_rgb/chunk-000/file-{ep_idx:03d}.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Episode {ep_idx}: video introuvable a {video_path}, skip")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    transitions = []
    frame_idx = 0
    paused = False
    current_phase = "top"

    print(f"\n=== Episode {ep_idx} ===")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        display = frame.copy()
        cv2.putText(display, f"Ep{ep_idx} Frame{frame_idx} Phase:{current_phase}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, "T=top D=droit G=gauche SPACE=pause Q=skip ESC=quit", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Annotation", display)

        key = cv2.waitKey(int(1000/fps) if not paused else 0) & 0xFF
        if key == ord('t'):
            current_phase = "top"
            transitions.append({"frame": frame_idx, "phase": "top"})
            print(f"  -> TOP a la frame {frame_idx}")
        elif key == ord('d'):
            current_phase = "wrist_right"
            transitions.append({"frame": frame_idx, "phase": "wrist_right"})
            print(f"  -> WRIST DROIT a la frame {frame_idx}")
        elif key == ord('g'):
            current_phase = "wrist_left"
            transitions.append({"frame": frame_idx, "phase": "wrist_left"})
            print(f"  -> WRIST GAUCHE a la frame {frame_idx}")
        elif key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            with open(OUTPUT_FILE, "w") as f:
                json.dump(annotations, f, indent=2)
            print("Sauvegarde et arret.")
            sys.exit(0)

    annotations[str(ep_idx)] = transitions
    cap.release()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Episode {ep_idx} termine, {len(transitions)} transitions enregistrees")

cv2.destroyAllWindows()
print("Toutes les annotations terminees.")
