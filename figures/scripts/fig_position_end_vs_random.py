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

# ---- Hardcoded data: seed means and seed SDs (n=3 seeds, ~2 items/seed) ----
# Panel (a): rates
rate_groups = ["Self span\nITT move", "Self span\ntarget cited",
               "Topical foil\nITT move", "Topical foil\ntarget cited"]
rate_end = np.array([0.5, 0.5, 0.0, 0.0])
rate_end_sd = np.array([0.0, 0.0, 0.0, 0.0])
rate_rand = np.array([0.3333, 0.3333, 0.0, 0.3333])
rate_rand_sd = np.array([0.2357, 0.2357, 0.0, 0.2357])

# Panel (b): delta log p(target ID), nats
lp_groups = ["Self span", "Topical foil"]
lp_end = np.array([4.803, -0.146])
lp_end_sd = np.array([0.508, 0.031])
lp_rand = np.array([1.941, -0.345])
lp_rand_sd = np.array([1.609, 0.203])

C_END = COLORS[0]
C_RAND = COLORS[3]
H_RAND = "//"


def draw_pair(ax, xc, end, end_sd, rnd, rnd_sd, width, clip_low):
    for off, vals, sds, col, hatch in ((-width / 2, end, end_sd, C_END, ""),
                                       (width / 2, rnd, rnd_sd, C_RAND, H_RAND)):
        xp = xc + off
        ax.bar(xp, vals, width=width, color=col, edgecolor="black",
               linewidth=0.6, hatch=hatch, zorder=2)
        low = np.minimum(vals, sds) if clip_low else sds
        m = sds > 0
        if m.any():
            ax.errorbar(xp[m], vals[m], yerr=np.vstack([low, sds])[:, m],
                        fmt="none", ecolor="black", elinewidth=0.9,
                        capsize=2.5, zorder=3)
        z = vals == 0.0
        if z.any():
            ax.scatter(xp[z], np.zeros(z.sum()), marker="_", s=60, color=col,
                       edgecolors="black", linewidths=2.0, clip_on=False,
                       zorder=4)


fig, (ax_rate, ax_lp) = plt.subplots(
    1, 2, figsize=(7.0, 3.0), constrained_layout=True,
    gridspec_kw={"width_ratios": [2, 1]})

width = 0.36

# Panel (a)
xa = np.arange(len(rate_groups))
draw_pair(ax_rate, xa, rate_end, rate_end_sd, rate_rand, rate_rand_sd,
          width, clip_low=True)
ax_rate.axvline(1.5, color="grey", linestyle=":", linewidth=0.8, zorder=1)
ax_rate.set_xticks(xa)
ax_rate.set_xticklabels(rate_groups, fontsize=10)
ax_rate.set_ylim(-0.02, 1.0)
ax_rate.set_xlim(-0.6, len(rate_groups) - 0.4)
ax_rate.set_ylabel("Rate over eligible items")
ax_rate.set_xlabel(r"Span type $\times$ metric")
ax_rate.set_title("(a) Migration and citation rates")
ax_rate.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax_rate.xaxis.grid(False)
ax_rate.set_axisbelow(True)

# Panel (b)
xb = np.arange(len(lp_groups))
draw_pair(ax_lp, xb, lp_end, lp_end_sd, lp_rand, lp_rand_sd,
          width, clip_low=False)
ax_lp.axhline(0.0, color="black", linewidth=0.7, zorder=1)
ax_lp.set_xticks(xb)
ax_lp.set_xticklabels(lp_groups, fontsize=10)
ax_lp.set_ylim(-0.8, 6.0)
ax_lp.set_xlim(-0.6, len(lp_groups) - 0.4)
ax_lp.set_ylabel(r"$\Delta \log p(\mathrm{target\ ID})$ (nats)")
ax_lp.set_xlabel("Span type")
ax_lp.set_title(r"(b) Teacher-forced $\Delta \log p$")
ax_lp.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax_lp.xaxis.grid(False)
ax_lp.set_axisbelow(True)

fig.suptitle("Fixed end vs randomized injection point", fontsize=12)

handles = [
    Patch(facecolor=C_END, edgecolor="black", label="End of passage"),
    Patch(facecolor=C_RAND, edgecolor="black", hatch=H_RAND,
          label="Random internal boundary"),
    Line2D([0], [0], color="black", marker="_", markersize=6, linestyle="-",
           linewidth=0.9, label=r"$\pm$1 SD (3 seeds)"),
]
try:
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False)
except Exception:
    ax_lp.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                 frameon=False)

out_path = r"/workspace/output/fig_position_end_vs_random.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")