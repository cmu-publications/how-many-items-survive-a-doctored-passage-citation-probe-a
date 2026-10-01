"""
Pure metric functions over per-item row dicts. No model is loaded here. No function returns
NaN: an empty denominator gives None. Rows are joined by (seed, sample_id), never by position.
Condition keys are exactly the plan's replication.condition_keys.

seed_condition_metrics refuses a seed on which any condition has zero item rows: the primary
metric (itt_citation_migration_rate) must be a finite number on every recorded seed, so an
empty seed (pool exhausted / every item ineligible) raises EmptySeedConditionError instead of
being recorded with a None primary rate.
"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BASELINE = "baseline_unedited_alce_citation"
SELF_END = "self_span_end_injection"
TOPICAL_END = "topical_foil_end_injection"
SWAP_END = "entity_swap_foil_end_injection"
SELF_RANDOM = "self_span_random_injection"
TOPICAL_RANDOM = "topical_foil_random_injection"
NULL_ARM = "null_random_drop_trailing_whitespace_edit"
POSITIVE_ARM = "positive_control_planted_evidence_relocated"
NULL_ARM_PREFIX = "null_random_drop"
POSITIVE_ARM_PREFIX = "positive_control_planted"
CONDITION_ORDER = (BASELINE, SELF_END, TOPICAL_END, SWAP_END, SELF_RANDOM, TOPICAL_RANDOM, NULL_ARM, POSITIVE_ARM)
PLAN_CONDITION_KEYS = ("baseline_unedited_alce_citation", "self_span_end_injection", "topical_foil_end_injection",
                       "entity_swap_foil_end_injection", "self_span_random_injection",
                       "topical_foil_random_injection", "null_random_drop_trailing_whitespace_edit",
                       "positive_control_planted_evidence_relocated")
if set(CONDITION_ORDER) != set(PLAN_CONDITION_KEYS):
    raise RuntimeError(f"condition keys {CONDITION_ORDER} differ from the plan {PLAN_CONDITION_KEYS}")
if not NULL_ARM.startswith(NULL_ARM_PREFIX) or not POSITIVE_ARM.startswith(POSITIVE_ARM_PREFIX):
    raise RuntimeError(f"control arm keys {NULL_ARM!r}, {POSITIVE_ARM!r} lack the mandatory prefixes "
                       f"{NULL_ARM_PREFIX!r} / {POSITIVE_ARM_PREFIX!r}")
EDITED_CONDITIONS = (SELF_END, TOPICAL_END, SWAP_END, SELF_RANDOM, TOPICAL_RANDOM, NULL_ARM, POSITIVE_ARM)
SPAN_ARMS = (SELF_END, TOPICAL_END, SWAP_END, SELF_RANDOM, TOPICAL_RANDOM)
RANDOM_ARMS = (SELF_RANDOM, TOPICAL_RANDOM)
RELIANCE_ARMS = (SELF_END, TOPICAL_END, SWAP_END, SELF_RANDOM, TOPICAL_RANDOM, NULL_ARM)
RELIANCE_CLASSES = ("re_sourcing", "post_rationalization", "mixed", "undefined")
FALLBACK_COUNTER_KEYS = (
    "slot_mismatch", "digit_merged_slot_skipped", "loo_base_nonpositive", "natural_bin_empty",
    "nli_truncated", "bootstrap_undefined_resamples", "swap_nli_rejected", "ppl_too_short",
    "item_rows_missing", "pool_exhausted", "item_exceptions", "glmm_failed", "item_overlap_across_seeds")

Row = Dict[str, Any]
Key = Tuple[Any, str]


class EmptySeedConditionError(ValueError):
    """A seed has zero item rows in at least one condition, so its primary metric is undefined.

    Raised instead of returning a record whose itt_citation_migration_rate is None; the caller
    treats it as a failed seed (recorded with this message), never as a recorded seed.
    """

    def __init__(self, empty_conditions: Sequence[str]):
        self.empty_conditions = list(empty_conditions)
        super().__init__(
            f"seed has zero item rows in condition(s) {self.empty_conditions}; the primary metric "
            f"itt_citation_migration_rate would be undefined, so the seed is not recorded "
            f"(pool exhausted or every item ineligible)")


def new_counters() -> Counter:
    c: Counter = Counter()
    for k in FALLBACK_COUNTER_KEYS:
        c[k] = 0
    return c


def row_key(row: Row) -> Key:
    return (row.get("seed"), str(row["sample_id"]))


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else float(num) / float(den)


def _count(rows: Sequence[Row], key: str) -> int:
    return sum(1 for r in rows if r.get(key))


# ITEM 27
def itt_rate(rows: Sequence[Row]) -> Optional[float]:
    return _rate(_count(rows, "move"), len(rows))


def target_cited_rate(rows: Sequence[Row]) -> Optional[float]:
    return _rate(_count(rows, "target_cited"), len(rows))


def answer_flip_rate(rows: Sequence[Row]) -> Optional[float]:
    return _rate(_count(rows, "answer_flip"), len(rows))


def conditional_rate(rows: Sequence[Row]) -> Optional[float]:
    return _rate(_count(rows, "move"), _count(rows, "answer_kept"))


def parse_rate(rows: Sequence[Row]) -> Optional[float]:
    return _rate(_count(rows, "parse_ok"), len(rows))


def determinism_rate(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    return _rate(sum(1 for a, b in pairs if a == b), len(pairs))


# ITEM 28
def conditional_rate_intersection(rows_by_cond: Dict[str, Sequence[Row]],
                                  edited_conds: Sequence[str] = EDITED_CONDITIONS) -> Dict[str, Optional[float]]:
    kept_sets = [{row_key(r) for r in rows_by_cond.get(c, []) if r.get("answer_kept")} for c in edited_conds]
    joint = set.intersection(*kept_sets) if kept_sets else set()
    out: Dict[str, Optional[float]] = {}
    for c, rows in rows_by_cond.items():
        by = {row_key(r): r for r in rows}
        out[c] = _rate(sum(1 for k in joint if k in by and by[k].get("move")), len(joint))
    return out


# ITEM 29
def neighbor_excess(rows_c: Sequence[Row], rows_null: Sequence[Row], side: str) -> Optional[float]:
    flag, exists = f"neighbor_{side}_new", f"{side}_exists"
    null_by = {row_key(r): r for r in rows_null}
    c_by = {row_key(r): r for r in rows_c}
    keys = sorted((k for k, r in c_by.items() if r.get(exists) and k in null_by), key=str)
    rc = _rate(sum(1 for k in keys if c_by[k].get(flag)), len(keys))
    rn = _rate(sum(1 for k in keys if null_by[k].get(flag)), len(keys))
    return None if rc is None or rn is None else rc - rn


# ITEM 30
def natural_weights(offsets: Sequence[float], bins: int = 5) -> Optional[List[float]]:
    if not offsets:
        return None
    counts = [0] * bins
    for x in offsets:
        counts[min(bins - 1, max(0, int(bins * float(x))))] += 1
    total = sum(counts)
    return [c / total for c in counts]


def natural_location_rate(rows_random: Sequence[Row], weights: Optional[Sequence[float]],
                          counters: Optional[Counter] = None) -> Optional[float]:
    if weights is None:
        return None
    num, wsum = 0.0, 0.0
    for b, w in enumerate(weights):
        if w <= 0:
            continue
        rb = itt_rate([r for r in rows_random if r.get("offset_bin") == b])
        if rb is None:
            if counters is not None:
                counters["natural_bin_empty"] += 1
            continue
        num += w * rb
        wsum += w
    return None if wsum == 0 else num / wsum


# ITEM 31
def margin_terciles(rows: Sequence[Row]) -> Dict[Key, int]:
    valid = [r for r in rows if r.get("margin") is not None]
    valid.sort(key=lambda r: (float(r["margin"]), str(r["sample_id"]), str(r.get("seed"))))
    n = len(valid)
    return {row_key(r): (3 * i) // n for i, r in enumerate(valid)}


def near_tie_concentration(rows: Sequence[Row], terciles: Dict[Key, int]) -> Optional[float]:
    moved = [r for r in rows if r.get("move")]
    return _rate(sum(1 for r in moved if terciles.get(row_key(r)) == 0), len(moved))


def tercile_null_over_self(rows_null: Sequence[Row], rows_self: Sequence[Row],
                           terciles: Dict[Key, int]) -> Optional[float]:
    a = itt_rate([r for r in rows_null if terciles.get(row_key(r)) == 0])
    b = itt_rate([r for r in rows_self if terciles.get(row_key(r)) == 0])
    if a is None or b is None or b == 0:
        return None
    return a / b


# ITEM 32
def reliance_shares(rows: Sequence[Row], single_source_only: bool = False) -> Dict[str, Optional[float]]:
    moved = [r for r in rows if r.get("move") and (not single_source_only or r.get("redundancy") == "single_source")]
    return {f"{c}_share": _rate(sum(1 for r in moved if r.get("rel_class") == c), len(moved))
            for c in RELIANCE_CLASSES}


# ITEM 33
def stratified_itt(rows: Sequence[Row], terciles: Optional[Dict[Key, int]] = None) -> Dict[str, Optional[float]]:
    terciles = terciles or {}
    out = {"itt_single_source": itt_rate([r for r in rows if r.get("redundancy") == "single_source"]),
           "itt_redundant": itt_rate([r for r in rows if r.get("redundancy") == "redundant"]),
           "itt_early": itt_rate([r for r in rows if r.get("stratum") == "early"]),
           "itt_late": itt_rate([r for r in rows if r.get("stratum") == "late"])}
    for k in range(3):
        out[f"itt_tercile_{k}"] = itt_rate([r for r in rows if terciles.get(row_key(r)) == k])
    return out


def mean_or_none(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None]
    return None if not vals else float(np.mean(vals))


def _check_finite(name: str, v: Any) -> None:
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError(f"non-finite metric {name}={v}")


def _require_rows_in_every_condition(rows_by_cond: Dict[str, Sequence[Row]], counters: Counter) -> None:
    """The primary metric must be finite on every recorded seed: refuse a seed with an empty condition."""
    empty = [c for c in CONDITION_ORDER if len(rows_by_cond.get(c, [])) == 0]
    if empty:
        counters["item_rows_missing"] += 1
        raise EmptySeedConditionError(empty)


# ITEM 34
def seed_condition_metrics(rows_by_cond: Dict[str, Sequence[Row]], baseline_scan_stats: Dict[str, int],
                           determinism: Sequence[Tuple[str, str]], alias_offsets: Optional[Sequence[float]] = None,
                           counters: Optional[Counter] = None, bins: int = 5) -> Dict[str, Dict[str, Any]]:
    counters = counters if counters is not None else new_counters()
    _require_rows_in_every_condition(rows_by_cond, counters)
    terciles = margin_terciles(list(rows_by_cond.get(BASELINE, [])))
    inter = conditional_rate_intersection(rows_by_cond)
    null_rows = list(rows_by_cond.get(NULL_ARM, []))
    weights = natural_weights(list(alias_offsets or []), bins)
    scan_parse = _rate(int(baseline_scan_stats.get("n_parse_ok", 0)), int(baseline_scan_stats.get("n_generated", 0)))
    out: Dict[str, Dict[str, Any]] = {}
    for cond in CONDITION_ORDER:
        rows = list(rows_by_cond.get(cond, []))
        deltas = [r.get("delta_logp_target_id") for r in rows]
        dvals = [float(d) for d in deltas if d is not None]
        itt = itt_rate(rows)
        if itt is None:
            # unreachable after _require_rows_in_every_condition; kept as an explicit contract guard
            counters["item_rows_missing"] += 1
            raise EmptySeedConditionError([cond])
        m: Dict[str, Any] = {
            "itt_citation_migration_rate": itt, "primary_metric": itt,
            "target_cited_rate": target_cited_rate(rows), "answer_flip_rate": answer_flip_rate(rows),
            "conditional_migration_rate": conditional_rate(rows),
            "conditional_migration_rate_intersection": inter.get(cond),
            "citation_parse_rate": scan_parse if cond == BASELINE else parse_rate(rows),
            "delta_logp_target_id_mean": mean_or_none(dvals), "n_delta_none": len(deltas) - len(dvals),
            "neighbor_next_excess": neighbor_excess(rows, null_rows, "next") if cond in SPAN_ARMS else None,
            "neighbor_prev_excess": neighbor_excess(rows, null_rows, "prev") if cond in SPAN_ARMS else None,
            "alce_recall": mean_or_none([r.get("alce_recall") for r in rows]),
            "alce_precision": mean_or_none([r.get("alce_precision") for r in rows]),
            "natural_location_rate": natural_location_rate(rows, weights, counters) if cond in RANDOM_ARMS else None,
            "near_tie_concentration": near_tie_concentration(rows, terciles),
            "baseline_determinism_rate": determinism_rate(determinism) if cond == BASELINE else None,
            "n_items": len(rows),
        }
        empty = {f"{c}_share": None for c in RELIANCE_CLASSES}
        rel_all = reliance_shares(rows) if cond in RELIANCE_ARMS else dict(empty)
        rel_ss = reliance_shares(rows, True) if cond in RELIANCE_ARMS else dict(empty)
        m.update(rel_all)
        m.update({f"{k}_single_source": v for k, v in rel_ss.items()})
        m.update(stratified_itt(rows, terciles))
        for k, v in m.items():
            _check_finite(f"{cond}/{k}", v)
        out[cond] = m
    return out


def aggregate_over_seeds(per_seed: Dict[Any, Dict[str, Dict[str, Any]]], key: str) -> Dict[str, Dict[str, Any]]:
    """Mean and sample std (ddof=1) over seeds; never the last value."""
    out: Dict[str, Dict[str, Any]] = {}
    conds = sorted({c for conds in per_seed.values() for c in conds})
    for c in conds:
        vals = [per_seed[s][c].get(key) for s in sorted(per_seed, key=str) if c in per_seed[s]]
        nums = [float(v) for v in vals if v is not None and not isinstance(v, bool)]
        n = len(nums)
        out[c] = {"mean": float(np.mean(nums)) if n else None,
                  "std": float(np.std(nums, ddof=1)) if n >= 2 else None, "n": n}
    return out