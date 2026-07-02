from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import glob
import subprocess
import os
import json

app = Flask(__name__)

DATASET_ROOT = "/home/soriano/lerobot/data/sort_by_weight_v2_multicam"
FRAMES_DIR = "/tmp/episode_frames"
OUTPUT_FILE = "/home/soriano/lerobot/phase_annotations.json"

ep_files = sorted(glob.glob(f"{DATASET_ROOT}/meta/episodes/chunk-000/*.parquet"))
all_episodes = []
for f in ep_files:
    df = pd.read_parquet(f)
    for _, row in df.iterrows():
        all_episodes.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "chunk_index": int(row["videos/observation.images.top_rgb/chunk_index"]),
            "file_index": int(row["videos/observation.images.top_rgb/file_index"]),
            "from_ts": float(row["videos/observation.images.top_rgb/from_timestamp"]),
            "to_ts": float(row["videos/observation.images.top_rgb/to_timestamp"]),
        })

def ensure_frames_extracted(ep_idx):
    ep_out_dir = f"{FRAMES_DIR}/ep_{ep_idx:03d}"
    if os.path.exists(ep_out_dir) and len(glob.glob(f"{ep_out_dir}/*.jpg")) > 0:
        return ep_out_dir
    os.makedirs(ep_out_dir, exist_ok=True)
    ep = all_episodes[ep_idx]
    video_path = f"{DATASET_ROOT}/videos/observation.images.top_rgb/chunk-{ep['chunk_index']:03d}/file-{ep['file_index']:03d}.mp4"
    duration = ep['to_ts'] - ep['from_ts']
    cmd = ["ffmpeg", "-ss", str(ep['from_ts']), "-i", video_path, "-t", str(duration),
           "-vf", "select='not(mod(n,5))'", "-vsync", "0", f"{ep_out_dir}/frame_%04d.jpg", "-y"]
    subprocess.run(cmd, capture_output=True)
    return ep_out_dir

def load_annotations():
    try:
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_annotations(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head><title>Annotation phases</title>
<style>
body { font-family: Arial; background: #222; color: white; text-align: center; }
img { max-width: 90vw; max-height: 60vh; border: 3px solid #555; }
button { font-size: 20px; padding: 10px 20px; margin: 5px; cursor: pointer; }
.top { background: #4488ff; }
.right { background: #ff8844; }
.left { background: #44ff88; }
#status { font-size: 18px; margin: 10px; }
#timeline { display: flex; flex-wrap: wrap; gap: 2px; margin: 10px; justify-content: center; }
.seg { width: 8px; height: 20px; }
</style>
</head>
<body>
<h2 id="title">Episode 0</h2>
<div id="status">Frame: <span id="frameNum">0</span> / <span id="frameMax">0</span> | Phase actuelle: <b id="curPhase">top</b></div>
<img id="img" src="">
<br>
<button class="top" onclick="setPhase('top')">T - TOP</button>
<button class="right" onclick="setPhase('wrist_right')">D - WRIST DROIT</button>
<button class="left" onclick="setPhase('wrist_left')">G - WRIST GAUCHE</button>
<br>
<button onclick="prevFrame()">&lt; Frame precedente</button>
<button onclick="nextFrame()">Frame suivante &gt;</button>
<br>
<button onclick="prevEp()">&lt;&lt; Episode precedent</button>
<button onclick="nextEp()">Episode suivant &gt;&gt;</button>
<br><br>
<div id="timeline"></div>
<script>
let epIdx = 0, frameIdx = 0, maxFrame = 0;
let transitions = [];

function loadEp(idx) {
  epIdx = idx;
  fetch(`/episode/${idx}`).then(r => r.json()).then(data => {
    maxFrame = data.n_frames - 1;
    transitions = data.transitions || [];
    frameIdx = 0;
    document.getElementById('title').innerText = `Episode ${idx}`;
    document.getElementById('frameMax').innerText = maxFrame;
    updateImg();
    updateTimeline();
  });
}

function updateImg() {
  document.getElementById('img').src = `/frame/${epIdx}/${frameIdx}?t=${Date.now()}`;
  document.getElementById('frameNum').innerText = frameIdx;
  let phase = 'top';
  for (const t of transitions) { if (t.frame <= frameIdx) phase = t.phase; }
  document.getElementById('curPhase').innerText = phase;
}

function updateTimeline() {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  const colors = {top: '#4488ff', wrist_right: '#ff8844', wrist_left: '#44ff88'};
  for (let i = 0; i <= maxFrame; i++) {
    let phase = 'top';
    for (const t of transitions) { if (t.frame <= i) phase = t.phase; }
    const d = document.createElement('div');
    d.className = 'seg';
    d.style.background = colors[phase];
    tl.appendChild(d);
  }
}

let lastPhase = 'top';
function setPhase(phase) {
  if (phase !== lastPhase) {
    transitions.push({frame: frameIdx, phase: phase});
    transitions.sort((a,b) => a.frame - b.frame);
    fetch(`/annotate/${epIdx}`, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(transitions)});
    lastPhase = phase;
    updateTimeline();
  }
  nextFrame();
}

function nextFrame() { if (frameIdx < maxFrame) { frameIdx++; updateImg(); } }
function prevFrame() { if (frameIdx > 0) { frameIdx--; updateImg(); } }
function nextEp() { loadEp(epIdx + 1); }
function prevEp() { if (epIdx > 0) loadEp(epIdx - 1); }

document.addEventListener('keydown', (e) => {
  if (e.key === 't' || e.key === 'T') setPhase('top');
  if (e.key === 'd' || e.key === 'D') setPhase('wrist_right');
  if (e.key === 'g' || e.key === 'G') setPhase('wrist_left');
  if (e.key === 'ArrowRight') nextFrame();
  if (e.key === 'ArrowLeft') prevFrame();
  if (e.key === 'PageDown') nextEp();
  if (e.key === 'PageUp') prevEp();
});

document.body.tabIndex = 0;
document.body.focus();
window.onclick = () => document.body.focus();
document.body.tabIndex = 0;
document.body.focus();
window.onclick = () => document.body.focus();
loadEp(0);
</script>
</body>
</html>
"""

@app.route("/episode/<int:idx>")
def get_episode(idx):
    ep_dir = ensure_frames_extracted(idx)
    n_frames = len(glob.glob(f"{ep_dir}/*.jpg"))
    annotations = load_annotations()
    transitions = annotations.get(str(idx), [])
    return jsonify({"n_frames": n_frames, "transitions": transitions})

@app.route("/frame/<int:ep_idx>/<int:frame_idx>")
def get_frame(ep_idx, frame_idx):
    ep_dir = f"{FRAMES_DIR}/ep_{ep_idx:03d}"
    return send_from_directory(ep_dir, f"frame_{frame_idx+1:04d}.jpg")

@app.route("/annotate/<int:idx>", methods=["POST"])
def annotate(idx):
    annotations = load_annotations()
    annotations[str(idx)] = request.json
    save_annotations(annotations)
    return jsonify({"ok": True})

if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"Ouvre: http://{ip}:5050")
    app.run(host="0.0.0.0", port=5050, threaded=True)
