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
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Hardcoded data: ALCE NLI-proxy citation scores, seed mean and seed SD (n=3 seeds)
arms = [
    "Baseline",
    "Null edit",
    "Pos. control",
    "Self span (end)",
    "Self span (rand.)",
    "Topical foil (end)",
    "Topical foil (rand.)",
    "Entity-swap (end)",
]
precision = np.array([0.167, 0.167, 0.139, 0.056, 0.694, 0.056, 0.056, 0.167])
precision_sd = np.array([0.0, 0.0, 0.196, 0.079, 0.142, 0.079, 0.079, 0.118])
recall = np.array([0.083, 0.083, 0.079, 0.033, 0.242, 0.028, 0.024, 0.089])
recall_sd = np.array([0.0, 0.0, 0.112, 0.047, 0.054, 0.039, 0.034, 0.068])

# Companion ITT migration rates for the same arms (cross-reference with main figure)
itt_rate = np.array([0.0, 0.0, 0.0, 0.5, 0.3333, 0.0, 0.0, 0.5])

baseline_precision = 0.167

C_PREC = COLORS[0]
C_REC = COLORS[4]

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

y = np.arange(len(arms))
h = 0.38

for off, vals, sds, col, hatch in ((-h / 2, precision, precision_sd, C_PREC, ""),
                                   (h / 2, recall, recall_sd, C_REC, "//")):
    yp = y + off
    ax.barh(yp, vals, height=h, color=col, edgecolor="black", linewidth=0.5,
            hatch=hatch, zorder=2)
    m = sds > 0
    xerr = np.vstack([np.minimum(vals, sds), sds])  # lower whisker clipped at 0
    ax.errorbar(vals[m], yp[m], xerr=xerr[:, m], fmt="none", ecolor="black",
                elinewidth=0.7, capsize=1.8, zorder=3)

# Baseline precision reference
ax.axvline(baseline_precision, color="black", linestyle="--", linewidth=0.8,
           alpha=0.7, zorder=1)

# Family separators: controls | self spans | foils
for ys in (2.5, 4.5):
    ax.axhline(ys, color="grey", linestyle=":", linewidth=0.7, zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(arms, fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.invert_yaxis()
ax.set_xlim(0.0, 0.9)
ax.set_ylim(len(arms) - 0.5, -0.5)
ax.set_xlabel("ALCE citation score", fontsize=8)
ax.set_ylabel("Edit arm", fontsize=8)
ax.set_title("ALCE citation quality under planted spans", fontsize=10)
ax.xaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

handles = [
    Patch(facecolor=C_PREC, edgecolor="black", label="Precision"),
    Patch(facecolor=C_REC, edgecolor="black", hatch="//", label="Recall"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=0.8,
           label="Baseline precision"),
    Line2D([0], [0], color="black", marker="|", markersize=5, linestyle="-",
           linewidth=0.7, label=r"$\pm$1 SD (3 seeds)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.2),
          ncol=2, frameon=False, fontsize=7)

out_path = r"/workspace/output/fig_alce_position_artifact.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")