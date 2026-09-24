"""
Ambulance Chaser — Demo Pipeline (Synthetic Data)
C4: Long-tail / Rare Scenario Mining

Chạy KHÔNG CẦN GPU, không cần dataset thật.
Sinh dữ liệu synthetic mô phỏng embedding space của CLIP,
chạy toàn bộ pipeline, so sánh baselines, xuất charts.

Usage:
    python demo_synthetic.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from tabulate import tabulate

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
N_TOTAL       = 10_000        # Tổng số frames giả lập
EMBED_DIM     = 768           # CLIP ViT-B/32 output dim
BUDGET_RATIO  = 0.05          # 5% budget
K_NN          = 50            # k cho k-NN density
ALPHA, BETA, GAMMA = 0.40, 0.35, 0.25   # Trọng số compound
TEMPORAL_STRIDE = 30          # Min stride giữa các frame chọn

# 6 Scenario Groups — Vietnam Traffic Ontology
SCENARIO_GROUPS = {
    "S1_xe_qua_kho":     {"label": "🚐 Xe tải/ba gác quá khổ",  "n": 80,  "center_offset": np.array([3.5, -2.0])},
    "S2_dong_vat":       {"label": "🐃 Động vật trên đường",     "n": 30,  "center_offset": np.array([-3.0, 3.5])},
    "S3_su_kien_duong":  {"label": "⛺ Sự kiện chiếm đường",     "n": 50,  "center_offset": np.array([2.0, 4.0])},
    "S4_ngap_lut":       {"label": "🌊 Ngập lụt/thời tiết",      "n": 40,  "center_offset": np.array([-4.0, -1.5])},
    "S5_xe_khan_cap":    {"label": "🚑 Xe khẩn cấp + xe máy",   "n": 60,  "center_offset": np.array([0.5, -4.5])},
    "S6_dem_compound":   {"label": "🌙 Đêm + Che khuất",         "n": 40,  "center_offset": np.array([-2.5, -3.5])},
}
N_RARE = sum(g["n"] for g in SCENARIO_GROUPS.values())  # 300

print("=" * 60)
print("  🚗  AMBULANCE CHASER — Synthetic Demo Pipeline")
print("=" * 60)
print(f"  Tổng frames: {N_TOTAL:,}")
print(f"  Rare frames: {N_RARE} ({N_RARE/N_TOTAL:.1%})")
print(f"  Budget: {BUDGET_RATIO:.0%} = {int(N_TOTAL * BUDGET_RATIO):,} frames")
print(f"  Scenario groups: {len(SCENARIO_GROUPS)}")
print("=" * 60)

# ============================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================
print("\n[1/6] Sinh dữ liệu synthetic...")

# Normal frames: clustered around origin with moderate spread
normal_embeddings = np.random.randn(N_TOTAL - N_RARE, EMBED_DIM) * 0.5

# Rare frames: placed in low-density regions (offset from main cluster)
rare_embeddings = []
rare_labels = []
rare_group_ids = []

for gid, (key, group) in enumerate(SCENARIO_GROUPS.items()):
    n = group["n"]
    offset = group["center_offset"]
    # Create embeddings far from the normal cluster
    emb = np.random.randn(n, EMBED_DIM) * 0.3
    # Apply offset in first 2 dims (for visualization) + random offset in other dims
    emb[:, 0] += offset[0]
    emb[:, 1] += offset[1]
    emb[:, 2:10] += np.random.randn(8) * 1.5  # spread in higher dims
    rare_embeddings.append(emb)
    rare_labels.extend([key] * n)
    rare_group_ids.extend([gid] * n)

rare_embeddings = np.vstack(rare_embeddings)

# Combine
all_embeddings = np.vstack([normal_embeddings, rare_embeddings])
all_embeddings = normalize(all_embeddings)  # L2 normalize (like CLIP)

is_rare = np.zeros(N_TOTAL, dtype=bool)
is_rare[N_TOTAL - N_RARE:] = True

group_labels = ["normal"] * (N_TOTAL - N_RARE) + rare_labels
group_ids = [-1] * (N_TOTAL - N_RARE) + rare_group_ids

# Simulate temporal order (frame indices)
frame_indices = np.arange(N_TOTAL)

print(f"   ✓ {N_TOTAL - N_RARE:,} normal + {N_RARE} rare frames generated")
print(f"   ✓ Embedding shape: {all_embeddings.shape}")

# ============================================================
# 2. k-NN DENSITY ESTIMATION
# ============================================================
print("\n[2/6] k-NN density estimation (k={})...".format(K_NN))

try:
    import faiss
    # Use faiss for fast k-NN
    index = faiss.IndexFlatIP(EMBED_DIM)  # cosine sim on L2-normalized = dot product
    index.add(all_embeddings.astype(np.float32))
    distances, neighbors = index.search(all_embeddings.astype(np.float32), K_NN + 1)
    # Distance to k-th nearest neighbor (skip self at index 0)
    knn_dist = distances[:, -1]
except ImportError:
    print("   ⚠ faiss not installed, using sklearn (slower)...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=K_NN + 1, metric="cosine")
    nn.fit(all_embeddings)
    distances, neighbors = nn.kneighbors(all_embeddings)
    knn_dist = distances[:, -1]

# Novelty score: low density = high novelty
eps = 1e-8
novelty_score = 1.0 / (knn_dist + eps)
# Invert for cosine: higher knn_dist (cosine) = farther neighbors = more novel
# For faiss inner product: lower value = farther = more novel
novelty_score = 1.0 - knn_dist  # for IP, lower sim = more novel

print(f"   ✓ Novelty score range: [{novelty_score.min():.4f}, {novelty_score.max():.4f}]")
print(f"   ✓ Mean novelty (rare): {novelty_score[is_rare].mean():.4f} vs (normal): {novelty_score[~is_rare].mean():.4f}")

# ============================================================
# 3. ENSEMBLE UNCERTAINTY
# ============================================================
print("\n[3/6] Ensemble uncertainty (3 classifiers)...")

# Train 3 classifiers to predict rare vs normal
# Use a small labeled subset as "proxy labels" (simulating pre-existing weak labels)
n_train = 2000
train_idx = np.random.choice(N_TOTAL, n_train, replace=False)
y_train = is_rare[train_idx].astype(int)

classifiers = [
    ("RF", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)),
    ("GBT", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
]

probas = []
for name, clf in classifiers:
    clf.fit(all_embeddings[train_idx], y_train)
    p = clf.predict_proba(all_embeddings)[:, 1]  # P(rare)
    probas.append(p)
    print(f"   ✓ {name} trained — mean P(rare) on rare: {p[is_rare].mean():.3f}, normal: {p[~is_rare].mean():.3f}")

probas = np.array(probas)  # (3, N)
uncertainty = np.std(probas, axis=0)  # High std = high disagreement

print(f"   ✓ Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

# ============================================================
# 4. TEMPORAL DIVERSITY
# ============================================================
print("\n[4/6] Temporal diversity calculation...")

# Compute similarity with preceding frames
temporal_div = np.ones(N_TOTAL)
for i in range(1, N_TOTAL):
    lookback = max(0, i - TEMPORAL_STRIDE)
    window = all_embeddings[lookback:i]
    sims = window @ all_embeddings[i]
    temporal_div[i] = 1.0 - sims.max()  # 1 - max_similarity = diversity

print(f"   ✓ Temporal diversity range: [{temporal_div.min():.4f}, {temporal_div.max():.4f}]")

# ============================================================
# 5. COMPOUND SCORING & SELECTION
# ============================================================
print("\n[5/6] Compound scoring & budget selection...")

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + eps)

novelty_norm = min_max_norm(novelty_score)
uncertainty_norm = min_max_norm(uncertainty)
temporal_norm = min_max_norm(temporal_div)

compound_score = (
    ALPHA * novelty_norm +
    BETA * uncertainty_norm +
    GAMMA * temporal_norm
)

budget = int(N_TOTAL * BUDGET_RATIO)

# --- BASELINES ---
def random_select(n):
    return np.random.choice(N_TOTAL, n, replace=False)

def uncertainty_only_select(n):
    return np.argsort(uncertainty)[-n:]

def diversity_only_select(n):
    return np.argsort(novelty_norm)[-n:]

def our_select(n):
    # Select top scores with temporal stride constraint
    sorted_idx = np.argsort(compound_score)[::-1]
    selected = []
    selected_set = set()
    for idx in sorted_idx:
        if len(selected) >= n:
            break
        # Temporal stride check
        too_close = False
        for s in selected[-TEMPORAL_STRIDE:]:
            if abs(int(idx) - int(s)) < TEMPORAL_STRIDE:
                too_close = True
                break
        if not too_close:
            selected.append(idx)
            selected_set.add(idx)
    return np.array(selected)

selections = {
    "Random Sampling":   random_select(budget),
    "Uncertainty-only":  uncertainty_only_select(budget),
    "Diversity-only":    diversity_only_select(budget),
    "Ours (Ambulance Chaser)": our_select(budget),
}

# --- METRICS ---
def rare_case_recall(selected, is_rare):
    """% of all rare cases that are selected"""
    return is_rare[selected].sum() / is_rare.sum()

def scenario_coverage(selected, group_ids, n_groups=6):
    """How many scenario groups are represented"""
    g = [group_ids[i] for i in selected if group_ids[i] >= 0]
    return len(set(g))

def redundancy_ratio(selected, embeddings):
    """Mean pairwise cosine similarity in selected set"""
    if len(selected) > 2000:
        sub = np.random.choice(selected, 2000, replace=False)
    else:
        sub = selected
    emb = embeddings[sub]
    sims = emb @ emb.T
    np.fill_diagonal(sims, 0)
    return sims.mean()

def per_group_recall(selected, is_rare, group_labels, scenario_groups):
    """Recall per scenario group"""
    results = {}
    for key, group in scenario_groups.items():
        mask = np.array([gl == key for gl in group_labels])
        total = mask.sum()
        hit = sum(1 for i in selected if i < len(group_labels) and group_labels[i] == key)
        results[group["label"]] = hit / total if total > 0 else 0
    return results

print(f"\n{'='*70}")
print(f"  📊 KẾT QUẢ SO SÁNH @ Budget {BUDGET_RATIO:.0%} ({budget:,} frames)")
print(f"{'='*70}")

results = []
for name, sel in selections.items():
    r = rare_case_recall(sel, is_rare)
    c = scenario_coverage(sel, group_ids)
    red = redundancy_ratio(sel, all_embeddings)
    results.append({
        "Method": name,
        "Rare-Case Recall": f"{r:.1%}",
        "Coverage": f"{c}/6",
        "Redundancy": f"{red:.3f}",
        "Recall_raw": r,
        "Coverage_raw": c,
        "Redundancy_raw": red,
    })

print(tabulate(
    [{k: v for k, v in r.items() if not k.endswith("_raw")} for r in results],
    headers="keys", tablefmt="rounded_outline"
))

# Per-group recall for our method
print(f"\n  📋 Per-group Recall (Ours):")
our_sel = selections["Ours (Ambulance Chaser)"]
pgr = per_group_recall(our_sel, is_rare, group_labels, SCENARIO_GROUPS)
for label, recall in pgr.items():
    bar = "█" * int(recall * 30) + "░" * (30 - int(recall * 30))
    print(f"   {label:30s} {bar} {recall:.1%}")

# ============================================================
# 6. ABLATION STUDY
# ============================================================
print(f"\n{'='*70}")
print("  🔬 ABLATION STUDY")
print(f"{'='*70}")

ablations = {
    "Full system": compound_score,
    "− Temporal filter": ALPHA * novelty_norm + BETA * uncertainty_norm,
    "− Uncertainty": ALPHA * novelty_norm + GAMMA * temporal_norm,
    "− Density (CLIP)": BETA * uncertainty_norm + GAMMA * temporal_norm,
    "Random baseline": np.random.rand(N_TOTAL),
}

ablation_results = []
for name, scores in ablations.items():
    sel = np.argsort(scores)[-budget:]
    r = rare_case_recall(sel, is_rare)
    ablation_results.append({"Component": name, "Rare-Case Recall": f"{r:.1%}", "Recall_raw": r})

print(tabulate(
    [{k: v for k, v in r.items() if not k.endswith("_raw")} for r in ablation_results],
    headers="keys", tablefmt="rounded_outline"
))

# ============================================================
# 7. BUDGET SWEEP
# ============================================================
print(f"\n{'='*70}")
print("  📈 BUDGET SWEEP (1% → 20%)")
print(f"{'='*70}")

budgets = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
sweep_data = {"Budget": [], "Random": [], "Ours": [], "Uplift": []}

for b in budgets:
    n = int(N_TOTAL * b)
    r_random = rare_case_recall(random_select(n), is_rare)
    r_ours = rare_case_recall(np.argsort(compound_score)[-n:], is_rare)
    uplift = r_ours / max(r_random, eps)
    sweep_data["Budget"].append(f"{b:.0%}")
    sweep_data["Random"].append(f"{r_random:.1%}")
    sweep_data["Ours"].append(f"{r_ours:.1%}")
    sweep_data["Uplift"].append(f"{uplift:.1f}×")

print(tabulate(sweep_data, headers="keys", tablefmt="rounded_outline"))

# ============================================================
# 8. GENERATE CHARTS
# ============================================================
print(f"\n[7/7] Xuất biểu đồ...")

plt.style.use("dark_background")
COLORS = {
    "bg": "#0a0e1a",
    "card": "#1a2035",
    "text": "#f1f5f9",
    "muted": "#64748b",
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "purple": "#8b5cf6",
    "rose": "#f43f5e",
}

# --- Chart 1: Main Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

methods = [r["Method"] for r in results]
recalls = [r["Recall_raw"] for r in results]
colors = [COLORS["muted"], COLORS["amber"], COLORS["blue"], COLORS["green"]]

bars = ax.barh(methods, recalls, color=colors, height=0.6, edgecolor="none")
ax.axvline(x=1.0, color=COLORS["muted"], linestyle="--", alpha=0.3)

for bar, val in zip(bars, recalls):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=13, fontweight="bold", color=COLORS["text"])

ax.set_xlabel("Rare-Case Recall @ 5% Budget", fontsize=12, color=COLORS["text"])
ax.set_title("So sánh phương pháp chọn mẫu", fontsize=16, fontweight="bold", color=COLORS["text"], pad=20)
ax.set_xlim(0, max(recalls) * 1.25)
ax.tick_params(colors=COLORS["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_comparison.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 01_comparison.png")

# --- Chart 2: t-SNE Visualization ---
print("   Computing t-SNE (this may take 30-60s)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
coords = tsne.fit_transform(all_embeddings[:5000])  # Subsample for speed

fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS["bg"])

# Left: ground truth
ax = axes[0]
ax.set_facecolor(COLORS["bg"])
is_rare_sub = is_rare[:5000]
group_ids_sub = np.array(group_ids[:5000])

ax.scatter(coords[~is_rare_sub, 0], coords[~is_rare_sub, 1],
           c=COLORS["muted"], alpha=0.15, s=8, label="Normal")

group_colors = [COLORS["red"], COLORS["green"], COLORS["amber"],
                COLORS["cyan"], COLORS["rose"], COLORS["purple"]]
for gid, (key, group) in enumerate(SCENARIO_GROUPS.items()):
    mask = group_ids_sub == gid
    if mask.any():
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=group_colors[gid], s=40, alpha=0.9, label=group["label"], edgecolors="white", linewidths=0.5)

ax.set_title("Ground Truth — Rare Scenarios", fontsize=14, fontweight="bold", color=COLORS["text"])
ax.legend(loc="upper left", fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["muted"])
ax.tick_params(colors=COLORS["muted"])
ax.set_xlabel("t-SNE dim 1", color=COLORS["muted"])
ax.set_ylabel("t-SNE dim 2", color=COLORS["muted"])

# Right: our selection
ax = axes[1]
ax.set_facecolor(COLORS["bg"])

our_sel_sub = set(our_sel[our_sel < 5000])
selected_mask = np.array([i in our_sel_sub for i in range(5000)])

ax.scatter(coords[~selected_mask, 0], coords[~selected_mask, 1],
           c=COLORS["muted"], alpha=0.1, s=5, label="Not selected")
ax.scatter(coords[selected_mask & ~is_rare_sub, 0], coords[selected_mask & ~is_rare_sub, 1],
           c=COLORS["blue"], alpha=0.5, s=15, label="Selected (normal)")
ax.scatter(coords[selected_mask & is_rare_sub, 0], coords[selected_mask & is_rare_sub, 1],
           c=COLORS["green"], s=50, alpha=0.9, label="Selected (rare) ✓",
           edgecolors="white", linewidths=0.8)

# Highlight missed rare
missed = is_rare_sub & ~selected_mask
ax.scatter(coords[missed, 0], coords[missed, 1],
           c=COLORS["red"], s=30, alpha=0.7, marker="x", label="Missed rare ✗", linewidths=1.5)

ax.set_title("Ambulance Chaser — Selection @ 5%", fontsize=14, fontweight="bold", color=COLORS["text"])
ax.legend(loc="upper left", fontsize=8, facecolor=COLORS["card"], edgecolor=COLORS["muted"])
ax.tick_params(colors=COLORS["muted"])
ax.set_xlabel("t-SNE dim 1", color=COLORS["muted"])
ax.set_ylabel("t-SNE dim 2", color=COLORS["muted"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_tsne.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 02_tsne.png")

# --- Chart 3: Per-Group Recall ---
fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

labels = list(pgr.keys())
values = list(pgr.values())
colors_pg = [COLORS["red"] if v < 0.5 else COLORS["amber"] if v < 0.8 else COLORS["green"] for v in values]

bars = ax.barh(labels, values, color=colors_pg, height=0.6)
ax.axvline(x=0.95, color=COLORS["text"], linestyle="--", alpha=0.4, label="Target 95%")
ax.axvline(x=0.50, color=COLORS["red"], linestyle=":", alpha=0.3)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=12, fontweight="bold", color=COLORS["text"])

ax.set_xlabel("Recall @ 5% Budget", fontsize=12, color=COLORS["text"])
ax.set_title("Recall theo từng nhóm kịch bản", fontsize=16, fontweight="bold", color=COLORS["text"], pad=20)
ax.set_xlim(0, 1.15)
ax.legend(fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["muted"])
ax.tick_params(colors=COLORS["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_per_group.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 03_per_group.png")

# --- Chart 4: Ablation Study ---
fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

abl_names = [r["Component"] for r in ablation_results]
abl_vals = [r["Recall_raw"] for r in ablation_results]
abl_colors = [COLORS["green"], COLORS["amber"], COLORS["amber"], COLORS["red"], COLORS["muted"]]

bars = ax.barh(abl_names, abl_vals, color=abl_colors, height=0.55)
for bar, val in zip(bars, abl_vals):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=12, fontweight="bold", color=COLORS["text"])

ax.set_xlabel("Rare-Case Recall", fontsize=12, color=COLORS["text"])
ax.set_title("Ablation Study — Đóng góp từng component", fontsize=15, fontweight="bold", color=COLORS["text"], pad=15)
ax.set_xlim(0, max(abl_vals) * 1.25)
ax.tick_params(colors=COLORS["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_ablation.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 04_ablation.png")

# --- Chart 5: Budget Sweep ---
fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

budgets_pct = [b * 100 for b in budgets]
random_recalls = []
ours_recalls = []
for b in budgets:
    n = int(N_TOTAL * b)
    random_recalls.append(rare_case_recall(random_select(n), is_rare))
    ours_recalls.append(rare_case_recall(np.argsort(compound_score)[-n:], is_rare))

ax.plot(budgets_pct, random_recalls, "o--", color=COLORS["muted"], linewidth=2, markersize=8, label="Random")
ax.plot(budgets_pct, ours_recalls, "o-", color=COLORS["green"], linewidth=3, markersize=10, label="Ours")
ax.fill_between(budgets_pct, random_recalls, ours_recalls, alpha=0.15, color=COLORS["green"])

for i, (r, o) in enumerate(zip(random_recalls, ours_recalls)):
    uplift = o / max(r, eps)
    ax.annotate(f"{uplift:.1f}×", (budgets_pct[i], (r + o) / 2),
                fontsize=10, fontweight="bold", color=COLORS["cyan"], ha="center")

ax.set_xlabel("Budget (%)", fontsize=12, color=COLORS["text"])
ax.set_ylabel("Rare-Case Recall", fontsize=12, color=COLORS["text"])
ax.set_title("Rare-Case Recall vs Budget", fontsize=15, fontweight="bold", color=COLORS["text"], pad=15)
ax.legend(fontsize=12, facecolor=COLORS["card"], edgecolor=COLORS["muted"])
ax.tick_params(colors=COLORS["text"])
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(COLORS["muted"])
ax.spines["left"].set_color(COLORS["muted"])

plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_budget_sweep.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 05_budget_sweep.png")

# --- Chart 6: Score Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLORS["bg"])

components = [
    ("Novelty Score", novelty_norm, COLORS["blue"]),
    ("Uncertainty", uncertainty_norm, COLORS["amber"]),
    ("Compound Score", compound_score, COLORS["green"]),
]

for ax, (title, scores, color) in zip(axes, components):
    ax.set_facecolor(COLORS["bg"])
    ax.hist(scores[~is_rare], bins=50, alpha=0.5, color=COLORS["muted"], label="Normal", density=True)
    ax.hist(scores[is_rare], bins=50, alpha=0.8, color=color, label="Rare", density=True)
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.legend(fontsize=9, facecolor=COLORS["card"], edgecolor=COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.spines["left"].set_color(COLORS["muted"])

plt.suptitle("Phân bố điểm — Rare vs Normal", fontsize=15, fontweight="bold", color=COLORS["text"], y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_distributions.png"), dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
plt.close()
print(f"   ✓ 06_distributions.png")

# ============================================================
# SAVE RESULTS JSON
# ============================================================
output_json = {
    "config": {
        "n_total": N_TOTAL,
        "n_rare": N_RARE,
        "budget_ratio": BUDGET_RATIO,
        "budget_frames": budget,
        "k_nn": K_NN,
        "weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA},
    },
    "comparison": [{k: v for k, v in r.items() if not k.endswith("_raw")} for r in results],
    "ablation": [{k: v for k, v in r.items() if not k.endswith("_raw")} for r in ablation_results],
    "per_group_recall": {k: f"{v:.1%}" for k, v in pgr.items()},
    "budget_sweep": sweep_data,
}

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)
print(f"   ✓ results.json")

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*60}")
print(f"  ✅ HOÀN TẤT!")
print(f"     Tất cả output đã lưu vào: {OUT}/")
print(f"     - 01_comparison.png    : So sánh phương pháp")
print(f"     - 02_tsne.png          : t-SNE visualization")
print(f"     - 03_per_group.png     : Recall theo nhóm")
print(f"     - 04_ablation.png      : Ablation study")
print(f"     - 05_budget_sweep.png  : Budget sweep")
print(f"     - 06_distributions.png : Score distributions")
print(f"     - results.json         : Raw results")
print(f"{'='*60}")
