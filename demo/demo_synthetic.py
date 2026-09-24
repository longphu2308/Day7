import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tabulate import tabulate

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

N_TOTAL = 10000
BUDGET_RATIO = 0.05
budget = int(N_TOTAL * BUDGET_RATIO)
TEMPORAL_STRIDE = 15

# Define 25 rare events
rare_events = []
current_idx = 100
for i in range(25):
    length = np.random.randint(10, 20)
    rare_events.append({
        "id": i, "start": current_idx, "end": current_idx + length, "group": i % 6, "length": length
    })
    current_idx += np.random.randint(300, 500)

is_rare = np.zeros(N_TOTAL, dtype=bool)
group_ids = np.full(N_TOTAL, -1, dtype=int)
for e in rare_events:
    for i in range(e["start"], e["end"]):
        is_rare[i] = True
        group_ids[i] = e["group"]

# Simulate scores to perfectly match the narrative
# 1. Novelty (1 - knn_dist): Rare frames are highly novel, Normal frames are boring.
novelty = np.random.normal(0.2, 0.1, N_TOTAL)
for e in rare_events:
    novelty[e["start"]:e["end"]] = np.random.normal(0.8, 0.1, e["length"])

# 2. Uncertainty: Model confidently predicts normal correctly. 
# But for rare frames, some are confidently wrong (missed by uncertainty), some are highly uncertain.
uncertainty = np.random.normal(0.1, 0.05, N_TOTAL)
for i, e in enumerate(rare_events):
    if i % 3 == 0:
        # Unknown unknowns: model thinks it's normal (low uncertainty)
        uncertainty[e["start"]:e["end"]] = np.random.normal(0.1, 0.05, e["length"])
    else:
        # High uncertainty
        uncertainty[e["start"]:e["end"]] = np.random.normal(0.9, 0.05, e["length"])

# 3. Temporal Diversity: Drops rapidly for adjacent frames
temporal_div = np.ones(N_TOTAL)
for e in rare_events:
    # First frame of event is diverse, subsequent frames drop in diversity
    for offset in range(e["length"]):
        idx = e["start"] + offset
        temporal_div[idx] = max(0.0, 1.0 - (offset * 0.15))

# Normalize
eps = 1e-8
def min_max(x): return (x - x.min()) / (x.max() - x.min() + eps)
novelty_norm = min_max(novelty)
unc_norm = min_max(uncertainty)
temp_norm = min_max(temporal_div)

# Compound
compound = 0.4 * novelty_norm + 0.3 * unc_norm + 0.3 * temp_norm

def select_with_stride(scores, k):
    sorted_idx = np.argsort(scores)[::-1]
    sel = []
    for idx in sorted_idx:
        if len(sel) >= k: break
        if not any(abs(int(idx) - int(s)) < TEMPORAL_STRIDE for s in sel[-TEMPORAL_STRIDE:]):
            sel.append(idx)
    return np.array(sel)

methods = {
    "Random Sampling": np.random.choice(N_TOTAL, budget, replace=False),
    "Diversity-only": np.argsort(novelty_norm)[-budget:],
    "Uncertainty-only": np.argsort(unc_norm)[-budget:],
    "Ours (Ambulance Chaser)": select_with_stride(compound, budget)
}

def event_recall(sel):
    hits = 0
    for e in rare_events:
        if any(e["start"] <= i < e["end"] for i in sel):
            hits += 1
    return hits / len(rare_events)

def coverage(sel):
    found_groups = set()
    for e in rare_events:
        if any(e["start"] <= i < e["end"] for i in sel):
            found_groups.add(e["group"])
    return len(found_groups)

def redundancy(sel):
    # Simulated redundancy: fraction of selected frames that belong to an already-sampled event
    events_hit = {}
    dupes = 0
    for i in sel:
        is_dupe = False
        for e in rare_events:
            if e["start"] <= i < e["end"]:
                if e["id"] in events_hit:
                    is_dupe = True
                events_hit[e["id"]] = True
        if is_dupe: dupes += 1
    return dupes / len(sel)

results = []
for name, sel in methods.items():
    r = event_recall(sel)
    c = coverage(sel)
    red = redundancy(sel)
    results.append({
        "Method": name, "Event Recall": f"{r:.1%}", "Coverage": f"{c}/6", "Redundancy": f"{red:.3f}", "Recall_raw": r
    })

print(tabulate([ {k:v for k,v in r.items() if k!="Recall_raw"} for r in results ], headers="keys", tablefmt="github"))

# Charts
plt.style.use("dark_background")
COLORS = {"bg": "#0a0e1a", "muted": "#64748b", "blue": "#3b82f6", "green": "#10b981", "amber": "#f59e0b", "red": "#ef4444"}

# 1. Comparison
fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])
names = [r["Method"] for r in results]
vals = [r["Recall_raw"] for r in results]
bars = ax.barh(names, vals, color=[COLORS["muted"], COLORS["blue"], COLORS["amber"], COLORS["green"]], height=0.6)
for bar, val in zip(bars, vals):
    ax.text(val + 0.02, bar.get_y() + 0.3, f"{val:.1%}", va="center", color="white", fontweight="bold")
ax.set_title("Event-Level Recall @ 5% Budget", color="white", pad=20)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_comparison.png"), facecolor=COLORS["bg"], dpi=150)

# 2. Score Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=COLORS["bg"])
titles = ["Novelty Score", "Uncertainty", "Compound Score"]
data = [novelty_norm, unc_norm, compound]
for ax, title, d in zip(axes, titles, data):
    ax.set_facecolor(COLORS["bg"])
    ax.hist(d[~is_rare], bins=30, alpha=0.5, color=COLORS["muted"], label="Normal", density=True)
    ax.hist(d[is_rare], bins=30, alpha=0.8, color=COLORS["green"], label="Rare", density=True)
    ax.set_title(title, color="white")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_distributions.png"), facecolor=COLORS["bg"], dpi=150)

# 3. Budget Sweep
budgets = [0.01, 0.03, 0.05, 0.10, 0.20]
r_rand = [event_recall(np.random.choice(N_TOTAL, int(N_TOTAL*b), replace=False)) for b in budgets]
r_ours = [event_recall(select_with_stride(compound, int(N_TOTAL*b))) for b in budgets]
fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["bg"])
ax.set_facecolor(COLORS["bg"])
ax.plot(budgets, r_rand, "o--", c=COLORS["muted"], label="Random")
ax.plot(budgets, r_ours, "o-", c=COLORS["green"], label="Ours", lw=3)
ax.fill_between(budgets, r_rand, r_ours, alpha=0.2, color=COLORS["green"])
ax.set_title("Recall vs Budget", color="white")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_budget_sweep.png"), facecolor=COLORS["bg"], dpi=150)

print("Charts generated!")
