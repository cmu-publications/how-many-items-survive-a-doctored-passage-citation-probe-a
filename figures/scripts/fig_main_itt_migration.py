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

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Hardcoded data: ITT citation migration rate (primary_metric), mean over seeds 101-103
labels = [
    "Baseline\n(unedited)",
    "Null edit\n(whitespace)",
    "Positive\ncontrol",
    "Self span\n(end)",
    "Self span\n(random)",
    "Topical foil\n(end)",
    "Topical foil\n(random)",
    "Entity-swap\nfoil (end)",
]
means = np.array([0.0, 0.0, 0.0, 0.5, 0.3333, 0.0, 0.0, 0.5])
stds = np.array([0.0, 0.0, 0.0, 0.0, 0.2357, 0.0, 0.0, 0.0])
families = ["control", "control", "control", "self", "self", "foil", "foil", "foil"]

FAMILY_COLORS = {"control": COLORS[6], "self": COLORS[0], "foil": COLORS[1]}
bar_colors = [FAMILY_COLORS[f] for f in families]

# Asymmetric error bars: lower whisker truncated at 0
yerr_low = np.minimum(means, stds)
yerr_high = stds
yerr = np.vstack([yerr_low, yerr_high])

fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)

x = np.arange(len(labels))
ax.bar(x, means, width=0.65, color=bar_colors, edgecolor="black",
       linewidth=0.6, zorder=2)
ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black",
            elinewidth=0.9, capsize=3, zorder=4)

# Gate line drawn above bars so it stays visible across 0.5-high bars
gate = 0.5
ax.axhline(gate, color="black", linestyle="--", linewidth=1.0, zorder=3)

# Value labels above bars / whiskers
for xi, m, s in zip(x, means, stds):
    ax.text(xi, m + s + 0.04, f"{m:.2f}", ha="center", va="bottom",
            fontsize=9, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.0, 0.75)
ax.set_xlim(-0.6, len(labels) - 0.4)
ax.set_xlabel("Edit arm")
ax.set_ylabel(r"ITT migration rate (move $\wedge$ answer kept)")
ax.set_title("ITT citation migration by edit arm", fontsize=12)
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=FAMILY_COLORS["control"], edgecolor="black", label="Controls"),
    Patch(facecolor=FAMILY_COLORS["self"], edgecolor="black", label="Self spans"),
    Patch(facecolor=FAMILY_COLORS["foil"], edgecolor="black", label="Foils"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
           label="Positive-control\nexpected minimum"),
    Line2D([0], [0], color="black", marker="_", markersize=8, linestyle="-",
           linewidth=0.9, label=r"$\pm$1 SD (3 seeds)"),
]
ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False, borderaxespad=0.0)

out_path = r"/workspace/output/fig_main_itt_migration.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")