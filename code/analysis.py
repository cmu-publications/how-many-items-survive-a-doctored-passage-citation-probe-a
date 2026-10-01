"""
Paired statistics over pooled item rows. Arms are paired by (seed, sample_id).
Plan item 41: the p-value that goes into Holm and decision_verdicts (p_inference) is the exact McNemar
test on the discordant (seed, sample_id) pairs, as the plan names it.
Items OVERLAP across seeds: data.pool_for_seed gives every plan seed the whole development pool, so the
same sample_id can be included under several seeds (main.item_overlap reports them). Greedy decoding
makes those repeats correlated, so the row-level McNemar treats correlated pairs as independent and can be
anticonservative. This is recorded, not hidden: every contrast reports n_repeated_items and a
sample_id-clustered exact sign test (p_item_sign_test) as a SENSITIVITY check, Holm is also run on the
sensitivity p, and compute_contrasts records and prints every primary contrast whose significance differs
between the two. The bootstrap resamples sample_id clusters. Fit failures are returned as
{'error': message} and counted; non-converged fits are recorded (converged=False), counted and printed.
Only numpy and the standard library are used: exact binomial tails via math.comb, exact signed-rank
enumeration, and the GLM / GLMM of glm_fit.py.
"""
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from glm_fit import (GLM_METHOD, GLMM_METHOD, MAX_ITER, TERMS_FULL, TERMS_NO_SLOT, TERMS_NO_SPAN, design_matrix,
                     drop_aliased, fit_binomial_glmm, fit_logit)
from metrics import (BASELINE, CONDITION_ORDER, NULL_ARM, POSITIVE_ARM, SELF_END, SELF_RANDOM, SPAN_ARMS, SWAP_END,
                     TOPICAL_END, TOPICAL_RANDOM, answer_flip_rate, conditional_rate, itt_rate, margin_terciles,
                     mean_or_none, natural_location_rate, natural_weights, near_tie_concentration, neighbor_excess,
                     reliance_shares, row_key, tercile_null_over_self)

Row = Dict[str, Any]
PRIMARY_CONTRASTS = ("P1a", "P1b", "P2a_self", "P2a_topical")
CELL = {SELF_END: ("self", "end"), TOPICAL_END: ("topical", "end"), SELF_RANDOM: ("self", "random"),
        TOPICAL_RANDOM: ("topical", "random")}
F_FULL = "move ~ span*position + C(slot) + redundancy + C(tercile)"
F_NO_SLOT = "move ~ span*position + redundancy + C(tercile)"
F_NO_SPAN = "move ~ position + C(slot) + redundancy + C(tercile)"
INFERENCE_TEST = "mcnemar_exact"
SENSITIVITY_TEST = "item_sign_test"
WILCOXON_EXACT_MAX_N = 16
SEED_WILCOXON_TEST = (f"two-sided Wilcoxon signed-rank on per-seed ITT, zero differences dropped, average ranks; "
                      f"exact enumeration of all sign flips for n <= {WILCOXON_EXACT_MAX_N}, tie-corrected normal "
                      f"approximation above; descriptive only (n = 5 seeds gives a minimum p of 0.0625)")
PAIRED_TEST = ("Inference (Holm and verdicts, plan item 41): exact two-sided McNemar test on the discordant "
               "pairs, arms paired by (seed, sample_id). b = pairs where x moves and y does not, c = the "
               "reverse; p_inference = p = p_row_mcnemar = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)); zero "
               "discordant pairs give p = 1.0. rd = (b - c) / n; odds_ratio = b / c (None when c = 0). The "
               "test uses only discordant pairs and is underpowered when few pairs are discordant. Caveat: a "
               "sample_id repeated across seeds gives correlated pairs that McNemar treats as independent. "
               "Sensitivity (not used for verdicts): exact two-sided sign test on sample_id clusters "
               "(p_item_sign_test). Each sample_id is collapsed to its mean move difference over seeds, ties "
               "are dropped, one df per item.")


def pair_rows(rows_x: Sequence[Row], rows_y: Sequence[Row]) -> List[Tuple[Row, Row]]:
    bx = {row_key(r): r for r in rows_x}
    by = {row_key(r): r for r in rows_y}
    keys = sorted(set(bx) & set(by), key=lambda k: (str(k[0]), k[1]))
    return [(bx[k], by[k]) for k in keys]


def odds_ratio(b: int, c: int) -> Optional[float]:
    return None if c == 0 else float(b) / float(c)


def _binom_half_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 0.5), exact integer arithmetic."""
    return sum(math.comb(n, i) for i in range(int(k) + 1)) / (2 ** int(n))


def _mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2.0 * _binom_half_cdf(min(b, c), b + c)))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks 1..n with ties given their average rank."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    srt = arr[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _signed_rank_p(d: np.ndarray) -> float:
    ranks = _average_ranks(np.abs(d))
    t = float(np.sum(ranks[d > 0]))
    n = len(d)
    if n <= WILCOXON_EXACT_MAX_N:
        masks = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
        totals = masks @ ranks
        p_lo = float(np.mean(totals <= t + 1e-9))
        p_hi = float(np.mean(totals >= t - 1e-9))
        return float(min(1.0, 2.0 * min(p_lo, p_hi)))
    mean = n * (n + 1) / 4.0
    var = float(np.sum(ranks ** 2)) / 4.0
    z = (t - mean) / math.sqrt(var)
    return float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


def item_level_differences(pairs: Sequence[Tuple[Row, Row]]) -> Dict[str, float]:
    """Mean over seeds of move_x - move_y per sample_id (one value per independent cluster)."""
    per_item: Dict[str, List[int]] = defaultdict(list)
    for x, y in pairs:
        per_item[str(x["sample_id"])].append(int(bool(x.get("move"))) - int(bool(y.get("move"))))
    return {sid: float(np.mean(d)) for sid, d in sorted(per_item.items())}


# ITEM 35
def mcnemar_paired(rows_x: Sequence[Row], rows_y: Sequence[Row]) -> Dict[str, Any]:
    """Pairs by (seed, sample_id). p_inference is the exact McNemar p (plan item 41), used for Holm and the
    verdicts. p_item_sign_test is the sample_id-clustered sign test, reported as a sensitivity check."""
    pairs = pair_rows(rows_x, rows_y)
    b = sum(1 for x, y in pairs if x.get("move") and not y.get("move"))
    c = sum(1 for x, y in pairs if not x.get("move") and y.get("move"))
    n = len(pairs)
    diffs = item_level_differences(pairs)
    reps = Counter(str(x["sample_id"]) for x, _y in pairs)
    b_items = sum(1 for d in diffs.values() if d > 0)
    c_items = sum(1 for d in diffs.values() if d < 0)
    p_row = _mcnemar_exact_p(b, c)
    p_item = _mcnemar_exact_p(b_items, c_items)
    return {"b": b, "c": c, "n": n, "p": p_row, "p_row_mcnemar": p_row,
            "rd": None if n == 0 else (b - c) / n, "odds_ratio": odds_ratio(b, c),
            "n_items": len(diffs), "n_repeated_items": sum(1 for v in reps.values() if v > 1),
            "b_items": b_items, "c_items": c_items, "n_items_tied": len(diffs) - b_items - c_items,
            "p_item_sign_test": p_item, "sensitivity_test": SENSITIVITY_TEST,
            "p_inference": p_row, "inference_test": INFERENCE_TEST, "test": PAIRED_TEST}


# ITEM 36
def holm(pvals: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: Dict[str, Optional[float]] = {k: None for k in pvals}
    running = 0.0
    for j, (k, pv) in enumerate(items):
        running = max(running, min(1.0, (m - j) * float(pv)))
        out[k] = running
    return out


# ITEM 37
def cluster_bootstrap(rows: Sequence[Row], stat_fn: Callable[[List[Row]], Optional[float]], B: int = 2000,
                      seed: int = 20260928, levels: Sequence[float] = (0.95, 0.90),
                      counters: Optional[Counter] = None) -> Dict[str, Any]:
    clusters: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        clusters[str(r["sample_id"])].append(r)
    ids = sorted(clusters)
    res: Dict[str, Any] = {"point": stat_fn(list(rows)) if rows else None, "B": int(B), "n_clusters": len(ids),
                           "n_undefined": 0, "mean_defined": None}
    for lv in levels:
        res[f"ci{int(round(lv * 100))}"] = None
    if not ids:
        return res
    groups = [clusters[i] for i in ids]
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    undefined = 0
    for _ in range(int(B)):
        idx = rng.integers(0, len(ids), size=len(ids))
        v = stat_fn([r for i in idx for r in groups[int(i)]])
        if v is None:
            undefined += 1
        else:
            vals.append(float(v))
    if counters is not None:
        counters["bootstrap_undefined_resamples"] += undefined
    res["n_undefined"] = undefined
    if undefined > B / 2 or not vals:
        return res
    arr = np.asarray(vals)
    res["mean_defined"] = float(arr.mean())
    for lv in levels:
        lo, hi = np.percentile(arr, [50.0 * (1 - lv), 50.0 * (1 + lv)])
        res[f"ci{int(round(lv * 100))}"] = (float(lo), float(hi))
    return res


# ITEM 38
def tost_equivalence(ci90: Optional[Sequence[float]], margin: float = 0.05) -> Optional[bool]:
    if ci90 is None:
        return None
    return bool(ci90[0] > -margin and ci90[1] < margin)


# ITEM 40
def rank_biserial(diffs: Sequence[Optional[float]]) -> Optional[float]:
    d = [float(x) for x in diffs if x is not None and float(x) != 0.0]
    if not d:
        return None
    ranks = _average_ranks(np.abs(d))
    rp = float(sum(r for r, x in zip(ranks, d) if x > 0))
    rm = float(sum(r for r, x in zip(ranks, d) if x < 0))
    return (rp - rm) / (rp + rm)


def seed_wilcoxon(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Descriptive only (SEED_WILCOXON_TEST): with n = 5 seeds the minimum two-sided p is 0.0625."""
    if len(x) < 2 or len(x) != len(y) or all(float(a) == float(b) for a, b in zip(x, y)):
        return None
    d = np.asarray([float(a) - float(b) for a, b in zip(x, y)])
    d = d[d != 0.0]
    if len(d) == 0:
        return None
    return _signed_rank_p(d)


def _glmm_frame(rows: Sequence[Row]) -> List[Dict[str, Any]]:
    recs = []
    for r in rows:
        if r["condition"] not in CELL:
            continue
        span, pos = CELL[r["condition"]]
        recs.append({"move": int(bool(r.get("move"))), "span": span, "position": pos, "slot": int(r.get("t", 0)),
                     "redundancy": str(r.get("redundancy")), "tercile": int(r.get("tercile", -1)),
                     "item": str(r["sample_id"]), "seed": str(r.get("seed"))})
    return recs


# ITEM 39
def fit_glmm(rows: Sequence[Row]) -> Dict[str, Any]:
    recs = _glmm_frame(rows)
    if len(recs) < 8:
        return {"error": f"too few rows for the GLMM ({len(recs)})", "n": len(recs)}
    try:
        X, names = design_matrix(recs, TERMS_FULL)
        X, names, aliased = drop_aliased(X, names)
        y = np.asarray([r["move"] for r in recs], dtype=float)
        fit = fit_binomial_glmm(X, y, {"item": [r["item"] for r in recs], "seed": [r["seed"] for r in recs]})
        fe = {nm: {"mean": float(fit["fe_mean"][i]), "sd": float(fit["fe_sd"][i])} for i, nm in enumerate(names)}
    except Exception as exc:  # recorded, counted by the caller, never re-raised into the run
        return {"error": f"{type(exc).__name__}: {exc}", "n": len(recs)}
    if not all(np.isfinite([v["mean"] for v in fe.values()] + [v["sd"] for v in fe.values()])):
        return {"error": "non-finite GLMM estimate", "n": len(recs)}
    inter = [nm for nm in names if ":" in nm]
    return {"n": len(recs), "formula": F_FULL, "method": GLMM_METHOD, "fixed_effects": fe,
            "aliased_columns_dropped": aliased, "random_effect_sd": fit["random_effect_sd"],
            "laplace_loglik": fit["laplace_loglik"], "sd_grid": fit["grid"],
            "converged": fit["converged"], "n_iter": fit["n_iter"],
            "interaction_term": inter[0] if inter else None,
            "interaction": fe[inter[0]] if inter else None}


def deviance_drop(rows: Sequence[Row]) -> Dict[str, Any]:
    """Fixed-effects Binomial GLM approximation to F4 prediction 3 (slot drop > span drop)."""
    recs = _glmm_frame(rows)
    if len(recs) < 8:
        return {"error": f"too few rows ({len(recs)})"}
    y = np.asarray([r["move"] for r in recs], dtype=float)
    fits: Dict[str, Dict[str, Any]] = {}
    aliased: Dict[str, List[str]] = {}
    try:
        for key, terms in (("full", TERMS_FULL), ("no_slot", TERMS_NO_SLOT), ("no_span", TERMS_NO_SPAN)):
            X, names = design_matrix(recs, terms)
            X, _names, dropped = drop_aliased(X, names)
            aliased[key] = dropped
            fits[key] = fit_logit(X, y)
    except Exception as exc:  # recorded and counted by the caller
        return {"error": f"{type(exc).__name__}: {exc}"}
    vals = [fits["full"]["deviance"], fits["no_slot"]["deviance"], fits["no_span"]["deviance"]]
    if not all(np.isfinite(vals)):
        return {"error": "non-finite deviance"}
    slot_drop, span_drop = vals[1] - vals[0], vals[2] - vals[0]
    return {"slot_drop": slot_drop, "span_drop": span_drop, "slot_gt_span": bool(slot_drop > span_drop),
            "deviance": {"full": vals[0], "no_slot": vals[1], "no_span": vals[2]},
            "formulas": {"full": F_FULL, "no_slot": F_NO_SLOT, "no_span": F_NO_SPAN},
            "aliased_columns_dropped": aliased, "method": GLM_METHOD,
            "converged": all(f["converged"] for f in fits.values()),
            "n_iter": {k: f["n_iter"] for k, f in fits.items()},
            "note": "fixed-effects GLM approximation"}


def _cond_rows(rows: Sequence[Row], cond: str) -> List[Row]:
    return [r for r in rows if r["condition"] == cond]


def _rd_fn(cx: str, cy: str) -> Callable[[List[Row]], Optional[float]]:
    def f(rows: List[Row]) -> Optional[float]:
        a, b = itt_rate(_cond_rows(rows, cx)), itt_rate(_cond_rows(rows, cy))
        return None if a is None or b is None else a - b
    return f


def _ratio_fn(cx: str, cy: str) -> Callable[[List[Row]], Optional[float]]:
    def f(rows: List[Row]) -> Optional[float]:
        a, b = itt_rate(_cond_rows(rows, cx)), itt_rate(_cond_rows(rows, cy))
        return None if a is None or b is None or b == 0 else a / b
    return f


def _sign_fn(rows: List[Row]) -> Optional[float]:
    s, t = _cond_rows(rows, SELF_END), _cond_rows(rows, TOPICAL_END)
    ds = mean_or_none([r.get("delta_logp_target_id") for r in s])
    dt = mean_or_none([r.get("delta_logp_target_id") for r in t])
    a, b = itt_rate(s), itt_rate(t)
    if ds is None or dt is None or a is None or b is None:
        return None
    return 1.0 if np.sign(ds - dt) == np.sign(a - b) else 0.0


def sign_agreement(rows: Sequence[Row], B: int, seed: int, counters: Optional[Counter] = None) -> Dict[str, Any]:
    sub = [r for r in rows if r["condition"] in (SELF_END, TOPICAL_END)]
    res = cluster_bootstrap(sub, _sign_fn, B=B, seed=seed, counters=counters)
    return {"agreement_fraction": res["mean_defined"], "n_undefined": res["n_undefined"], "B": B}


def holm_with_sensitivity(res: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    """Holm on the McNemar p (plan item 41) written into res[k]['p_holm']; Holm on the clustered sign test
    as a sensitivity check. Returns the caveat record naming every contrast whose significance differs."""
    adj = holm({k: res[k]["p_inference"] for k in PRIMARY_CONTRASTS})
    sens = holm({k: res[k]["p_item_sign_test"] for k in PRIMARY_CONTRASTS})
    disagree: List[str] = []
    for k in PRIMARY_CONTRASTS:
        res[k]["p_holm"] = adj[k]
        res[k]["p_holm_item_sign_test"] = sens[k]
        a, s = adj[k], sens[k]
        sig_main = a is not None and a < alpha
        sig_sens = s is not None and s < alpha
        res[k]["sig_under_item_sign_test"] = sig_sens
        if sig_main != sig_sens:
            disagree.append(k)
    res["holm"] = adj
    res["holm_input"] = "p_inference (" + INFERENCE_TEST + ", paired by (seed, sample_id); plan item 41)"
    res["holm_sensitivity_item_sign_test"] = sens
    repeated = {k: int(res[k]["n_repeated_items"]) for k in PRIMARY_CONTRASTS}
    caveat = {"n_repeated_items": repeated, "any_repeated_items": any(v > 0 for v in repeated.values()),
              "significance_differs_under_item_sign_test": disagree,
              "note": "repeated sample_ids across seeds give correlated McNemar pairs; the clustered sign "
                      "test is the sensitivity check"}
    if caveat["any_repeated_items"]:
        print(f"FLAG: McNemar pairs include sample_ids repeated across seeds {repeated}; "
              f"significance differs under the clustered sign test for {disagree}", flush=True)
    return caveat


def _report_fits(fits: Sequence[Tuple[str, Dict[str, Any]]], counters: Counter) -> None:
    for name, fit in fits:
        if "error" in fit:
            counters["glmm_failed"] += 1
            print(f"FLAG: model fit {name} failed: {fit['error']}", flush=True)
        elif fit.get("converged") is False:
            counters["glmm_nonconverged"] += 1
            print(f"FLAG: model fit {name} did not converge within {MAX_ITER} iterations; estimates recorded "
                  f"with converged=False", flush=True)


# ITEM 41
def compute_contrasts(pooled_rows: Sequence[Row], hp: Dict[str, Any], counters: Counter,
                      alias_offsets: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    B, bseed = int(hp["bootstrap_resamples"]), int(hp["bootstrap_seed"])
    thr, margin = float(hp["effect_threshold"]), float(hp["tost_margin"])
    by: Dict[str, List[Row]] = {c: _cond_rows(pooled_rows, c) for c in CONDITION_ORDER}
    terciles = margin_terciles(by[BASELINE])

    def boot(conds: Sequence[str], fn: Callable[[List[Row]], Optional[float]]) -> Dict[str, Any]:
        return cluster_bootstrap([r for c in conds for r in by[c]], fn, B=B, seed=bseed, counters=counters)

    def contrast(cx: str, cy: str) -> Dict[str, Any]:
        out = mcnemar_paired(by[cx], by[cy])
        ci = boot((cx, cy), _rd_fn(cx, cy))
        out.update(x=cx, y=cy, rd_ci95=ci["ci95"], rd_ci90=ci["ci90"])
        return out

    res: Dict[str, Any] = {}
    res["P1a"] = contrast(TOPICAL_END, NULL_ARM)
    ratio = boot((TOPICAL_END, SELF_END), _ratio_fn(TOPICAL_END, SELF_END))
    res["P1a"].update(ratio_topical_over_self=ratio["point"], ratio_ci95=ratio["ci95"])
    res["foil_share_ratio"] = {"point": ratio["point"], "ci95": ratio["ci95"]}
    res["P1b"] = contrast(SELF_END, TOPICAL_END)
    rd_b = res["P1b"]["rd"]
    res["P1b"]["mcnemar_direction"] = None if rd_b is None else ("self_gt_topical" if rd_b > 0 else
                                                                  "topical_gt_self" if rd_b < 0 else "none")
    res["P1b"]["tost_equivalent"] = tost_equivalence(res["P1b"]["rd_ci90"], margin)
    res["P1c"] = contrast(SWAP_END, TOPICAL_END)
    flips = {c: answer_flip_rate(by[c]) for c in SPAN_ARMS}
    p1d: Dict[str, Any] = {"flip_rates": flips}
    for foil in (TOPICAL_END, SWAP_END):
        fs, ff = flips[SELF_END], flips[foil]
        gap = None if fs is None or ff is None else fs - ff
        entry: Dict[str, Any] = {"flip_gap": gap}
        if gap is not None and abs(gap) >= thr:
            i_s, i_f = itt_rate(by[SELF_END]), itt_rate(by[foil])
            c_s, c_f = conditional_rate(by[SELF_END]), conditional_rate(by[foil])
            itt_gap = None if i_s is None or i_f is None else i_s - i_f
            cond_gap = None if c_s is None or c_f is None else c_s - c_f
            entry.update(itt_gap=itt_gap, conditional_gap=cond_gap,
                         gap_difference=None if itt_gap is None or cond_gap is None else cond_gap - itt_gap)
        p1d[foil] = entry
    res["P1d"] = p1d
    res["P2a_self"] = contrast(SELF_END, SELF_RANDOM)
    res["P2a_topical"] = contrast(TOPICAL_END, TOPICAL_RANDOM)
    for k in ("P2a_self", "P2a_topical"):
        res[k]["tost_equivalent"] = tost_equivalence(res[k]["rd_ci90"], margin)
    res["P2c"] = {c: {"next": neighbor_excess(by[c], by[NULL_ARM], "next"),
                      "prev": neighbor_excess(by[c], by[NULL_ARM], "prev")}
                  for c in (SELF_END, SELF_RANDOM, TOPICAL_END, TOPICAL_RANDOM)}
    weights = natural_weights(list(alias_offsets or []), int(hp["offset_bins"]))
    res["P2e"] = {"weights": weights}
    for name, end_c, rnd_c in (("self", SELF_END, SELF_RANDOM), ("topical", TOPICAL_END, TOPICAL_RANDOM)):
        res["P2e"][name] = {"natural": natural_location_rate(by[rnd_c], weights, counters),
                            "end": itt_rate(by[end_c]), "uniform_random": itt_rate(by[rnd_c])}
    # Holm across the 4 primary contrasts runs on the McNemar p (plan item 41).
    res["inference_caveat"] = holm_with_sensitivity(res, float(hp["alpha"]))
    res["paired_test"] = PAIRED_TEST
    single = [r for r in by[SELF_END] if r.get("redundancy") == "single_source"]
    redundant = [r for r in by[SELF_END] if r.get("redundancy") == "redundant"]

    def red_gap(rows: List[Row]) -> Optional[float]:
        a = itt_rate([r for r in rows if r.get("redundancy") == "single_source"])
        b = itt_rate([r for r in rows if r.get("redundancy") == "redundant"])
        return None if a is None or b is None else a - b

    rg = boot((SELF_END,), red_gap)
    res["redundancy_gap"] = {"point": rg["point"], "ci95": rg["ci95"], "n_single": len(single),
                             "n_redundant": len(redundant)}
    shares = reliance_shares(by[SELF_END])
    post_ci = boot((SELF_END,), lambda rows: reliance_shares(_cond_rows(rows, SELF_END))["post_rationalization_share"])
    post = shares["post_rationalization_share"]
    res["F3"] = {"self_shares": shares, "self_shares_single_source": reliance_shares(by[SELF_END], True),
                 "post_rationalization_ci95": post_ci["ci95"],
                 "prediction_supported": None if post is None else post <= 0.6,
                 "prediction_rejected": bool(post is not None and post >= 0.8 and post_ci["ci95"] is not None
                                             and post_ci["ci95"][0] > 0.7)}
    cells = [dict(r, tercile=terciles.get(row_key(r), -1)) for c in CELL for r in by[c]]
    glmm = fit_glmm(cells)
    dev = deviance_drop(cells)
    _report_fits((("glmm", glmm), ("deviance_drop", dev)), counters)
    res["F4"] = {"near_tie_concentration_self": near_tie_concentration(by[SELF_END], terciles),
                 "tercile0_null_over_self": tercile_null_over_self(by[NULL_ARM], by[SELF_END], terciles),
                 "deviance_drop": dev, "sign_agreement": sign_agreement(pooled_rows, B, bseed, counters),
                 "glmm": glmm}
    pairs = pair_rows(by[SELF_END], by[TOPICAL_END])
    diffs = [x["delta_logp_target_id"] - y["delta_logp_target_id"] for x, y in pairs
             if x.get("delta_logp_target_id") is not None and y.get("delta_logp_target_id") is not None]
    res["delta_logp_rank_biserial"] = rank_biserial(diffs)
    seeds = sorted({r.get("seed") for r in pooled_rows}, key=str)
    xs, ys = [], []
    for s in seeds:
        a = itt_rate([r for r in by[SELF_END] if r.get("seed") == s])
        b = itt_rate([r for r in by[TOPICAL_END] if r.get("seed") == s])
        if a is not None and b is not None:
            xs.append(a)
            ys.append(b)
    res["seed_wilcoxon"] = {"p": seed_wilcoxon(xs, ys), "n_seeds": len(xs), "test": SEED_WILCOXON_TEST,
                            "note": "descriptive only"}
    return res


def _lt(v: Optional[float], x: float) -> bool:
    return v is not None and v < x


def _ge(v: Optional[float], x: float) -> bool:
    return v is not None and v >= x


# ITEM 42
def decision_verdicts(contrasts: Optional[Dict[str, Any]], controls: Dict[str, Any],
                      hp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """p_holm comes from the exact McNemar test (p_inference, plan item 41); see PAIRED_TEST."""
    broken = []
    if controls.get("broken_null"):
        broken.append({"label": "broken_null", "evidence": {"null_itt": controls.get("null_itt")}})
    if controls.get("broken_positive"):
        broken.append({"label": "broken_positive",
                       "evidence": {"positive_target_cited": controls.get("positive_target_cited")}})
    if broken:
        return broken
    if contrasts is None:
        return [{"label": "no_valid_seeds", "evidence": {}}]
    alpha, thr = float(hp["alpha"]), float(hp["effect_threshold"])
    out: List[Dict[str, Any]] = []
    p1a, p1b, p2 = contrasts["P1a"], contrasts["P1b"], contrasts["P2a_self"]
    rci = p1a.get("ratio_ci95")
    if _lt(p1a.get("p_holm"), alpha) and _ge(p1a.get("rd"), thr) and rci is not None \
            and rci[0] > float(hp["ratio_ci_lower_min"]):
        out.append({"label": "foil_subtracted",
                    "evidence": {"p_holm": p1a["p_holm"], "p_test": p1a.get("inference_test"), "rd": p1a["rd"],
                                 "ratio_ci95": rci,
                                 "sig_under_item_sign_test": p1a.get("sig_under_item_sign_test")}})
    rd_b = p1b.get("rd")
    if p1b.get("tost_equivalent") is False and rd_b is not None and rd_b > 0 and not _lt(p1a.get("p_holm"), alpha):
        out.append({"label": "probe_valid", "evidence": {"P1b_rd": rd_b, "P1a_p_holm": p1a.get("p_holm")}})
    if _lt(p2.get("p_holm"), alpha) and p2.get("rd") is not None and abs(p2["rd"]) >= thr:
        out.append({"label": "position_artifact",
                    "evidence": {"p_holm": p2["p_holm"], "p_test": p2.get("inference_test"), "rd": p2["rd"],
                                 "sig_under_item_sign_test": p2.get("sig_under_item_sign_test")}})
    null = controls.get("null_itt")
    if null is not None and float(hp["null_expected_max"]) < null <= float(hp["null_broken_above"]):
        out.append({"label": "any_perturbation", "evidence": {"null_itt": null}})
    rg = contrasts["redundancy_gap"]
    if _ge(rg.get("point"), thr) and rg.get("ci95") is not None and rg["ci95"][0] > 0:
        out.append({"label": "redundancy_sensitivity", "evidence": rg})
    nt = contrasts["F4"]["near_tie_concentration_self"]
    if _ge(nt, float(hp["near_tie_share_threshold"])):
        out.append({"label": "near_tie_fragility", "evidence": {"near_tie_concentration_self": nt}})
    return out


# ITEM 43
def seed_validity(cond_metrics: Dict[str, Dict[str, Any]], hp: Dict[str, Any],
                  condition_names: Sequence[str] = CONDITION_ORDER) -> Dict[str, Any]:
    """Per-seed exclusion: missing arms, parse rate, determinism, undefined null ITT and the positive control
    (defined per seed by the plan). A null ITT above null_broken_above is only flagged here: the plan's
    broken-null diagnostic is run-level (main.aggregate_and_write, over every recorded seed), so seeds are
    never selected on their null-control outcome."""
    reasons = [f"missing:{c}" for c in condition_names if c not in cond_metrics]
    flags: List[str] = []
    for c in condition_names:
        pr = cond_metrics.get(c, {}).get("citation_parse_rate")
        if pr is None or pr < float(hp["parse_rate_gate"]):
            reasons.append(f"parse_rate:{c}={pr}")
    det = cond_metrics.get(BASELINE, {}).get("baseline_determinism_rate")
    if det is None or det < float(hp["determinism_gate"]):
        reasons.append(f"determinism={det}")
    null = cond_metrics.get(NULL_ARM, {}).get("itt_citation_migration_rate")
    if null is None:
        reasons.append("null_itt=None")
    elif null > float(hp["null_broken_above"]):
        flags.append(f"null_itt_above_broken={null} (run-level diagnostic; not a seed exclusion)")
    pos = cond_metrics.get(POSITIVE_ARM, {}).get("target_cited_rate")
    if pos is None or pos < float(hp["pos_broken_below"]):
        reasons.append(f"positive_target_cited={pos}")
    valid = not reasons
    success = bool(valid and null is not None and pos is not None and null <= float(hp["null_expected_max"])
                   and pos >= float(hp["pos_expected_min"]))
    return {"valid": valid, "success": success, "reasons": reasons, "flags": flags}


def success_rate(validities: Sequence[Dict[str, Any]]) -> Optional[float]:
    return None if not validities else sum(1 for v in validities if v.get("success")) / len(validities)