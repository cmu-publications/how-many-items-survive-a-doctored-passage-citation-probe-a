import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Academic styling
try:
    plt.style.use(['science', 'ieee'])
except Exception:
    try:
        plt.style.use(['seaborn-v0_8-whitegrid'])
    except Exception:
        pass  # Use default matplotlib style

# Colorblind-safe palette
COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

# Publication settings
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Hardcoded data: seed means (n=3 seeds, ~2 items/seed) and seed SDs
labels = [
    "Baseline",
    "Null\nedit",
    "Positive\ncontrol",
    "Self span\n(end)",
    "Self span\n(random)",
    "Topical\nfoil (end)",
    "Topical\nfoil (rand.)",
    "Entity-swap\nfoil (end)",
]

itt_move = np.array([0.0, 0.0, 0.0, 0.5, 0.3333, 0.0, 0.0, 0.5])
itt_move_sd = np.array([0.0, 0.0, 0.0, 0.0, 0.2357, 0.0, 0.0, 0.0])

target_cited = np.array([0.0, 0.0, 0.0, 0.5, 0.3333, 0.0, 0.3333, 0.6667])
target_cited_sd = np.array([0.0, 0.0, 0.0, 0.0, 0.2357, 0.0, 0.2357, 0.2357])

answer_flip = np.array([0.0, 0.0, 0.6667, 0.0, 0.0, 0.5, 0.6667, 0.1667])
answer_flip_sd = np.array([0.0, 0.0, 0.2357, 0.0, 0.0, 0.0, 0.2357, 0.2357])

series = [
    ("ITT move (answer kept)", itt_move, itt_move_sd, COLORS[0], ""),
    ("Target cited", target_cited, target_cited_sd, COLORS[2], "//"),
    ("Answer flip", answer_flip, answer_flip_sd, COLORS[1], ".."),
]

fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)

x = np.arange(len(labels))
width = 0.24
offsets = [-width, 0.0, width]

for (name, vals, sds, color, hatch), off in zip(series, offsets):
    xpos = x + off
    ax.bar(xpos, vals, width=width, color=color, edgecolor="black",
           linewidth=0.5, hatch=hatch, zorder=2)
    yerr = np.vstack([np.minimum(vals, sds), sds])
    mask = sds > 0
    ax.errorbar(xpos[mask], vals[mask], yerr=yerr[:, mask], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=1.5, zorder=3)
    # Mark exact zeros so empty slots read as measured 0, not missing
    zmask = vals == 0.0
    ax.scatter(xpos[zmask], np.zeros(zmask.sum()), marker="_", s=30,
               color=color, linewidths=1.6, clip_on=False, zorder=4)

# Family separators: controls | self spans | foils
for xs in (2.5, 4.5):
    ax.axvline(xs, color="grey", linestyle=":", linewidth=0.8, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlim(-0.55, len(labels) - 0.45)
ax.set_ylim(-0.02, 1.0)
ax.set_xlabel("Edit arm")
ax.set_ylabel("Rate over eligible items")
ax.set_title("Move, target-cited and answer-flip rates per arm", fontsize=12)
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

handles = [
    Patch(facecolor=c, edgecolor="black", hatch=h, label=n)
    for (n, _, _, c, h) in series
]
handles.append(Line2D([0], [0], color="black", marker="_", markersize=6,
                      linestyle="-", linewidth=0.8,
                      label="1 SD across\n3 seeds"))
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, borderaxespad=0.0)

out_path = r"/workspace/output/fig_outcome_decomposition.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")