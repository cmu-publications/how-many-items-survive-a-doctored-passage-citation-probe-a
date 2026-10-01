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

# Hardcoded data: delta_logp_target_id_mean (seed mean and seed SD, n=3), sorted descending
arms = [
    "Pos. control",
    "Self span (end)",
    "Self span (rand.)",
    "Entity-swap (end)",
    "Null edit",
    "Baseline",
    "Topical foil (end)",
    "Topical foil (rand.)",
]
delta = np.array([4.948, 4.803, 1.941, 1.092, 0.639, 0.0, -0.146, -0.345])
delta_sd = np.array([0.501, 0.508, 1.609, 0.859, 0.054, 0.0, 0.031, 0.203])
families = ["control", "self", "self", "foil", "control", "control", "foil", "foil"]

# Companion ITT migration rates for the same arms (cross-reference with main figure)
itt_rate = np.array([0.0, 0.5, 0.3333, 0.5, 0.0, 0.0, 0.0, 0.0])

FAMILY_COLORS = {"control": COLORS[6], "self": COLORS[0], "foil": COLORS[1]}
bar_colors = [FAMILY_COLORS[f] for f in families]
edge_widths = [1.4 if a == "Null edit" else 0.5 for a in arms]

null_floor = 0.639

fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

y = np.arange(len(arms))[::-1]  # largest at top

# Null-edit noise floor: 0 to null-edit mean drift
ax.axvspan(0.0, null_floor, color=COLORS[3], alpha=0.18, zorder=0, linewidth=0)
ax.axvline(0.0, color="black", linewidth=0.7, zorder=1)

ax.barh(y, delta, height=0.65, color=bar_colors, edgecolor="black",
        linewidth=edge_widths, zorder=2)
mask = delta_sd > 0
ax.errorbar(delta[mask], y[mask], xerr=delta_sd[mask], fmt="none",
            ecolor="black", elinewidth=0.8, capsize=2, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(arms, fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.set_xlim(-0.8, 6.0)
ax.set_ylim(-0.6, len(arms) - 0.4)
ax.set_xlabel(r"$\Delta \log p(\mathrm{target\ ID})$ (nats)", fontsize=8)
ax.set_ylabel("Edit arm", fontsize=8)
ax.set_title("Teacher-forced shift in target\ncitation log-probability", fontsize=9)
ax.xaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

handles = [
    Patch(facecolor=FAMILY_COLORS["control"], edgecolor="black", label="Controls"),
    Patch(facecolor=FAMILY_COLORS["self"], edgecolor="black", label="Self spans"),
    Patch(facecolor=FAMILY_COLORS["foil"], edgecolor="black", label="Foils"),
    Patch(facecolor=COLORS[3], alpha=0.35, edgecolor="none",
          label="Null-edit noise floor"),
    Line2D([0], [0], color="black", marker="|", markersize=5, linestyle="-",
           linewidth=0.8, label=r"$\pm$1 SD (3 seeds)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.2),
          ncol=2, frameon=False, fontsize=7)

out_path = r"/workspace/output/fig_delta_logp_target.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")