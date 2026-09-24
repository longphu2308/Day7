"""
Ambulance Chaser — Real Image Pipeline (CLIP + BDD100K)
C4: Long-tail / Rare Scenario Mining

Chạy trên Google Colab (GPU T4 free) hoặc local GTX 1650 Ti.
Dùng CLIP thật encode ảnh driving, sau đó chạy pipeline đầy đủ.

Usage LOCAL (GTX 1650 Ti):
    python demo_real.py --data_dir ./sample_images --device cuda

Usage COLAB:
    Copy toàn bộ file này vào cell Colab, hoặc upload rồi chạy:
    !python demo_real.py --data_dir /content/images --device cuda
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# PARSE ARGS
# ============================================================
parser = argparse.ArgumentParser(description="Ambulance Chaser — Real CLIP Pipeline")
parser.add_argument("--data_dir", type=str, default="./sample_images",
                    help="Thư mục chứa ảnh driving (jpg/png)")
parser.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"], help="Device cho CLIP")
parser.add_argument("--budget", type=float, default=0.05, help="Budget ratio (default 5%%)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size cho CLIP encode (giảm nếu OOM)")
parser.add_argument("--k_nn", type=int, default=50, help="k cho k-NN density")
parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
parser.add_argument("--skip_tsne", action="store_true", help="Bỏ qua t-SNE (tiết kiệm thời gian)")
args = parser.parse_args()

OUT = args.output_dir
os.makedirs(OUT, exist_ok=True)

# ============================================================
# CHECK DEPENDENCIES
# ============================================================
def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package, "-q"])

print("Checking dependencies...")
install_if_missing("open_clip", "open-clip-torch")
install_if_missing("torch")
install_if_missing("faiss", "faiss-cpu")
install_if_missing("sklearn", "scikit-learn")
install_if_missing("matplotlib")
install_if_missing("PIL", "Pillow")
install_if_missing("tqdm")
install_if_missing("tabulate")

import torch
import open_clip
from PIL import Image
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize as sk_normalize
from tabulate import tabulate

np.random.seed(42)
torch.manual_seed(42)

DEVICE = args.device if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# VIETNAM TRAFFIC ONTOLOGY — Zero-shot prompts
# ============================================================
SCENARIO_PROMPTS = {
    "S1_xe_qua_kho": [
        "a three-wheeled cargo bike carrying oversized load on the road",
        "an overloaded truck with goods sticking out on a highway",
        "a bicycle carrying a huge pile of goods blocking traffic",
    ],
    "S2_dong_vat": [
        "a water buffalo crossing a highway in Vietnam",
        "cows walking on a national road",
        "a flock of ducks crossing a road",
    ],
    "S3_su_kien_duong": [
        "a funeral procession blocking an intersection with a hearse",
        "a wedding tent set up on the side of a road spilling into traffic",
        "a street market with vendors and stalls extending into the road",
    ],
    "S4_ngap_lut": [
        "a flooded road with only the tops of vehicles visible",
        "cars driving through deep water on a flooded street",
        "a motorcycle navigating a heavily flooded road",
    ],
    "S5_xe_khan_cap": [
        "an ambulance trying to navigate through a swarm of motorbikes",
        "a fire truck stuck in dense motorcycle traffic",
        "an emergency vehicle with sirens in heavy traffic",
    ],
    "S6_dem_compound": [
        "a dark night road with a person sleeping on the road median",
        "motorcycles at night with helmets obscuring faces",
        "poorly lit road with construction workers at night",
    ],
}

# ============================================================
# STEP 1: LOAD & ENCODE IMAGES
# ============================================================
print("\n" + "=" * 60)
print("  [1/5] Loading CLIP model & encoding images...")
print("=" * 60)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

data_dir = Path(args.data_dir)
if not data_dir.exists():
    print(f"\n⚠️  Thư mục '{data_dir}' không tồn tại!")
    print("Tạo sample images từ random noise để demo...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(500):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(data_dir / f"frame_{i:05d}.jpg")
    print(f"   ✓ Đã tạo 500 ảnh random tại {data_dir}")

image_paths = sorted([
    p for p in data_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
])
print(f"   Tìm thấy {len(image_paths)} ảnh")

# Encode images in batches
all_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="   Encoding"):
        batch_paths = image_paths[i:i + args.batch_size]
        images = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
            except Exception:
                images.append(preprocess(Image.new("RGB", (224, 224))))
        images = torch.stack(images).to(DEVICE)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())

all_embeddings = np.vstack(all_embeddings).astype(np.float32)
N = len(all_embeddings)
print(f"   ✓ Encoded {N} images → shape {all_embeddings.shape}")

# ============================================================
# STEP 2: ZERO-SHOT SCENARIO LABELING
# ============================================================
print("\n" + "=" * 60)
print("  [2/5] Zero-shot scenario labeling via CLIP...")
print("=" * 60)

# Encode all scenario prompts
scenario_embeddings = {}
with torch.no_grad():
    for key, prompts in SCENARIO_PROMPTS.items():
        tokens = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scenario_embeddings[key] = text_features.cpu().numpy().mean(axis=0)  # Average

# Compute similarity to each scenario
scenario_scores = {}
for key, text_emb in scenario_embeddings.items():
    sims = all_embeddings @ text_emb
    scenario_scores[key] = sims

# Determine "rare" frames: top-N highest scoring for each scenario
RARE_THRESHOLD_PERCENTILE = 95
is_rare = np.zeros(N, dtype=bool)
group_ids = np.full(N, -1, dtype=int)
group_labels = ["normal"] * N

for gid, (key, scores) in enumerate(scenario_scores.items()):
    threshold = np.percentile(scores, RARE_THRESHOLD_PERCENTILE)
    mask = scores >= threshold
    is_rare |= mask
    for idx in np.where(mask)[0]:
        if group_ids[idx] == -1:  # First assignment wins
            group_ids[idx] = gid
            group_labels[idx] = key

n_rare = is_rare.sum()
print(f"   ✓ Identified {n_rare} rare frames ({n_rare/N:.1%}) across {len(SCENARIO_PROMPTS)} groups")
for gid, key in enumerate(SCENARIO_PROMPTS):
    count = (np.array(group_ids) == gid).sum()
    print(f"      {key}: {count} frames")

# ============================================================
# STEP 3: PIPELINE (same as synthetic)
# ============================================================
print("\n" + "=" * 60)
print("  [3/5] Running Ambulance Chaser pipeline...")
print("=" * 60)

# k-NN Density
print("   k-NN density estimation...")
index = faiss.IndexFlatIP(all_embeddings.shape[1])
index.add(all_embeddings)
distances, _ = index.search(all_embeddings, args.k_nn + 1)
knn_dist = distances[:, -1]
novelty_score = 1.0 - knn_dist

# Ensemble Uncertainty
print("   Ensemble uncertainty...")
n_train = min(2000, N // 3)
train_idx = np.random.choice(N, n_train, replace=False)
y_train = is_rare[train_idx].astype(int)

classifiers = [
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
]
probas = []
for clf in classifiers:
    clf.fit(all_embeddings[train_idx], y_train)
    probas.append(clf.predict_proba(all_embeddings)[:, 1])
uncertainty = np.std(probas, axis=0)

# Temporal diversity
print("   Temporal diversity...")
STRIDE = 30
temporal_div = np.ones(N)
for i in range(1, N):
    lookback = max(0, i - STRIDE)
    window = all_embeddings[lookback:i]
    sims = window @ all_embeddings[i]
    temporal_div[i] = 1.0 - sims.max()

# Compound score
eps = 1e-8
def norm(x):
    return (x - x.min()) / (x.max() - x.min() + eps)

ALPHA, BETA, GAMMA = 0.40, 0.35, 0.25
compound_score = ALPHA * norm(novelty_score) + BETA * norm(uncertainty) + GAMMA * norm(temporal_div)

# Select
budget = int(N * args.budget)
sorted_idx = np.argsort(compound_score)[::-1]
selected = []
for idx in sorted_idx:
    if len(selected) >= budget:
        break
    too_close = any(abs(int(idx) - int(s)) < STRIDE for s in selected[-STRIDE:])
    if not too_close:
        selected.append(idx)
our_selection = np.array(selected)

# Baselines
random_sel = np.random.choice(N, budget, replace=False)
uncertainty_sel = np.argsort(uncertainty)[-budget:]
diversity_sel = np.argsort(norm(novelty_score))[-budget:]

print(f"   ✓ Selected {len(our_selection)} frames @ {args.budget:.0%} budget")

# ============================================================
# STEP 4: EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("  [4/5] Evaluation...")
print("=" * 60)

def recall(sel):
    return is_rare[sel].sum() / max(is_rare.sum(), 1)

def coverage(sel):
    g = set(group_ids[i] for i in sel if group_ids[i] >= 0)
    return len(g)

def redundancy(sel):
    sub = np.random.choice(sel, min(2000, len(sel)), replace=False)
    emb = all_embeddings[sub]
    sims = emb @ emb.T
    np.fill_diagonal(sims, 0)
    return sims.mean()

methods = {
    "Random": random_sel,
    "Uncertainty-only": uncertainty_sel,
    "Diversity-only": diversity_sel,
    "Ours (Ambulance Chaser)": our_selection,
}

print(f"\n  Kết quả @ Budget {args.budget:.0%} ({budget} frames):\n")
rows = []
for name, sel in methods.items():
    r = recall(sel)
    c = coverage(sel)
    red = redundancy(sel)
    rows.append({"Method": name, "Rare Recall": f"{r:.1%}", "Coverage": f"{c}/6", "Redundancy": f"{red:.3f}"})

print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))

# ============================================================
# STEP 5: SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("  [5/5] Saving outputs...")
print("=" * 60)

# Save selected frame paths
selected_paths = [str(image_paths[i]) for i in our_selection if i < len(image_paths)]
with open(os.path.join(OUT, "selected_frames.json"), "w") as f:
    json.dump({"budget": args.budget, "n_selected": len(selected_paths), "frames": selected_paths[:100]}, f, indent=2)

# Save results
with open(os.path.join(OUT, "results_real.json"), "w", encoding="utf-8") as f:
    json.dump({"comparison": rows}, f, indent=2, ensure_ascii=False)

print(f"   ✓ selected_frames.json")
print(f"   ✓ results_real.json")
print(f"\n✅ Done! Check {OUT}/")
