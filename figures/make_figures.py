"""Figures 2 to 4 of the paper, from results_extract.json. Run from the paper folder."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ex = json.load(open("results_extract.json", encoding="utf-8"))
ORDER = ["c1_uncovered", "c3_parse", "c4_self_span", "c5_target", "c6_foil", "c7_swap"]
STAGES = ["scanned", "c1", "c3", "c4", "c5", "c6", "c7"]
# Okabe-Ito, validated with the dataviz validator; markers carry identity as well as color.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def funnel() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, pool, title in ((axes[0], "development", "Development pool, 544 items"),
                            (axes[1], "confirmation", "Confirmation pool, 404 items")):
        seeds = ex["yield"][pool]["per_seed"]
        x = list(range(len(STAGES)))
        for i, rec in enumerate(seeds):
            left = rec["n_scanned"]
            ys = [left]
            for key in ORDER:
                left -= rec["exclusions"][key]
                ys.append(left)
            assert ys[-1] == rec["n_included"]
            off = (i - (len(seeds) - 1) / 2) * 0.09
            ax.plot([v + off for v in x], ys, color=PALETTE[i], marker=MARKERS[i], markersize=4.5,
                    linewidth=1.5, label=f"seed {rec['seed']}, {rec['n_included']} admitted")
        target = seeds[0]["design_n_items"]
        ax.axhline(target, color="#595959", linestyle="--", linewidth=1)
        ax.text(6.3, target * 1.12, f"design target {target}", ha="right", va="bottom", fontsize=8,
                color="#404040")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 900)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, fontsize=8)
        ax.set_xlabel("after filter")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axes[1].annotate("404 scanned", xy=(0, 404), xytext=(0.35, 600), fontsize=7.5, color="#404040",
                     arrowprops={"arrowstyle": "-", "color": "#808080", "linewidth": 0.8})
    axes[0].set_yticks([1, 2, 4, 10, 32, 100, 544])
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_yticklabels(["1", "2", "4", "10", "32", "100", "544"])
    axes[0].set_ylabel("items remaining (log scale)")
    fig.tight_layout()
    fig.savefig("figures/eligibility_funnel.pdf")
    fig.savefig("figures/eligibility_funnel.png", dpi=300)


ARM_LABELS = {
    "baseline_unedited_alce_citation": "baseline, unedited",
    "null_random_drop_trailing_whitespace_edit": "null edit",
    "positive_control_planted_evidence_relocated": "positive control",
    "self_span_end_injection": "self span, end",
    "self_span_random_injection": "self span, random",
    "topical_foil_end_injection": "topical foil, end",
    "topical_foil_random_injection": "topical foil, random",
    "entity_swap_foil_end_injection": "entity-swap foil, end",
}


def arms() -> None:
    names = list(ARM_LABELS)
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    blocks = (("development", ex["development"]["conditions"], 0, "development, 4 items, 9 measurements"),
              ("confirmation", ex["conditions"], 1, "confirmation, 2 items, 6 measurements"))
    for pool, conds, i, label in blocks:
        off = 0.17 if i == 0 else -0.17
        # Each admitted item's own move rate is a small open marker and the item mean a filled one.
        # No interval is drawn (revision 2026-10-01): a binomial interval does not fit averaged
        # item means, and with 2 or 4 items the dots show the spread directly.
        for k, n in enumerate(names):
            y = len(names) - 1 - k + off
            vals = [a / b for a, b in conds[n]["primary_metric"]["item_values"]]
            ax.scatter(vals, [y] * len(vals), marker=MARKERS[i], s=22, facecolors="none",
                       edgecolors=PALETTE[i], linewidths=1.0, zorder=2)
        ys = [len(names) - 1 - k + off for k in range(len(names))]
        xs = [conds[n]["primary_metric"]["mean"] for n in names]
        ax.scatter(xs, ys, marker=MARKERS[i], s=34, color=PALETTE[i], zorder=3, label=label)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([ARM_LABELS[n] for n in reversed(names)], fontsize=8)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel("ITT migration rate, item mean (filled) and each item's own rate (open)")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/arm_rates.pdf")
    fig.savefig("figures/arm_rates.png", dpi=300)


def projection() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    targets = list(range(20, 401, 10))
    rates = (("confirmation point, 0.50%", ex["yield"]["confirmation"]["distinct_yield"], 0, "-"),
             ("pooled point, 0.63%", ex["yield"]["pooled"]["distinct_yield"], 1, "-"),
             ("confirmation Wilson upper, 1.79%", ex["yield"]["confirmation"]["distinct_yield_wilson_high"], 2, "--"),
             ("plan assumption, 50%", 0.5, 3, ":"))
    for label, rate, i, ls in rates:
        ax.plot(targets, [t / rate for t in targets], color=PALETTE[i], linestyle=ls, linewidth=1.8,
                label=label)
    for pool, name, x, ha in ((948, "all ASQA evaluation questions, 948", 400, "right"),
                              (404, "confirmation pool, 404", 22, "left")):
        ax.axhline(pool, color="#595959", linewidth=0.9, linestyle=(0, (1, 2)))
        ax.text(x, pool * 1.1, name, ha=ha, va="bottom", fontsize=7.5, color="#404040")
    ax.axvline(150, color="#BDBDBD", linewidth=0.8)
    ax.text(153, 25, "planned 150\nconfirmation items", fontsize=7.5, color="#404040", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("admitted items a study needs")
    ax.set_ylabel("questions to scan (log scale)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, which="major")
    ax.set_axisbelow(True)
    ax.set_ylim(25, 2e5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/pool_projection.pdf")
    fig.savefig("figures/pool_projection.png", dpi=300)


if __name__ == "__main__":
    funnel()
    arms()
    projection()
    print("ok")
