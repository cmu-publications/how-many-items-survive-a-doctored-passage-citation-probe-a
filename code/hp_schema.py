"""
Per-key validation of HYPERPARAMETERS (plan item 44).

build_config reads every key exactly once through HP_VALIDATORS and rejects a value outside its
allowed range, so every tunable is consumed by the config builder. Cross-key rules (n_min <= n_max,
expected <= broken thresholds, ...) run after the per-key checks.
"""
import math
from typing import Any, Callable, Dict, List, Mapping, Tuple


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def pos_int(v: Any) -> bool:
    return _is_int(v) and v > 0


def nonneg_int(v: Any) -> bool:
    return _is_int(v) and v >= 0


def pos_num(v: Any) -> bool:
    return _is_num(v) and float(v) > 0


def nonneg_num(v: Any) -> bool:
    return _is_num(v) and float(v) >= 0


def prob(v: Any) -> bool:
    return _is_num(v) and 0.0 <= float(v) <= 1.0


def pos_prob(v: Any) -> bool:
    return prob(v) and float(v) > 0


def text(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def any_str(v: Any) -> bool:
    return isinstance(v, str)


def str_list(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(text(x) for x in v)


def seed_list(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) > 0 and all(nonneg_int(x) for x in v)
            and len(set(v)) == len(v))


def rank_pair(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(pos_int(x) for x in v) and v[0] <= v[1]


def slot_strata(v: Any) -> bool:
    return (isinstance(v, dict) and len(v) > 0
            and all(isinstance(s, (list, tuple)) and len(s) > 0 and all(pos_int(x) for x in s) for s in v.values()))


def one_of(*allowed: Any) -> Callable[[Any], bool]:
    return lambda v: v in allowed


HP_VALIDATORS: Dict[str, Callable[[Any], bool]] = {
    "reader_model_id": text, "embed_model_id": text, "nli_model_id": text, "spacy_model": text,
    "decoding": one_of("greedy"), "batch_size": pos_int,
    "seeds": seed_list, "n_passages": pos_int, "n_demos": nonneg_int, "max_new_tokens": pos_int,
    "separator": any_str, "slot_strata": slot_strata, "min_target_sentences": pos_int,
    "foil_mining_ranks": rank_pair, "foil_length_tol": nonneg_num, "foil_cosine_tol": nonneg_num,
    "foil_ppl_tol_nats": nonneg_num, "topical_nli_contradiction_max": prob,
    "entity_types": str_list, "swap_max_tries": pos_int, "swap_nli_contradiction_min": prob,
    "offset_bins": pos_int,
    "reliance_delta_plant_nats_per_token": nonneg_num, "reliance_orig_drop_resourcing": prob,
    "reliance_orig_drop_postrat": prob,
    "alce_entail_threshold": prob, "nli_max_length": pos_int,
    "bootstrap_resamples": pos_int, "bootstrap_seed": nonneg_int, "effect_threshold": prob, "tost_margin": prob,
    "alpha": pos_prob, "ratio_ci_lower_min": nonneg_num, "near_tie_share_threshold": prob,
    "null_expected_max": prob, "null_broken_above": prob, "pos_expected_min": prob, "pos_broken_below": prob,
    "parse_rate_gate": prob, "determinism_gate": prob,
    "expected_eval_items": pos_int, "manifest_glob": text, "nli_sanity_min_correct": nonneg_int,
    "n_max_items": pos_int, "n_min_items": pos_int, "det_items": nonneg_int, "eligible_rate_prior": pos_prob,
    "time_budget_sec": pos_num, "overhead_sec": nonneg_num, "seed_start_safety_factor": pos_num,
    "pilot_scan_items": pos_int, "max_item_exceptions": nonneg_int,
    "smoke_n_items": pos_int, "smoke_min_items": pos_int, "smoke_scan_cap": pos_int,
}

CROSS_CHECKS: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
    ("n_min_items <= n_max_items", lambda h: h["n_min_items"] <= h["n_max_items"]),
    ("smoke_min_items <= smoke_n_items <= n_max_items",
     lambda h: h["smoke_min_items"] <= h["smoke_n_items"] <= h["n_max_items"]),
    ("null_expected_max <= null_broken_above", lambda h: h["null_expected_max"] <= h["null_broken_above"]),
    ("pos_broken_below <= pos_expected_min", lambda h: h["pos_broken_below"] <= h["pos_expected_min"]),
    ("pilot_scan_items <= smoke_scan_cap", lambda h: h["pilot_scan_items"] <= h["smoke_scan_cap"]),
]


def validate_hyperparameters(hp: Mapping[str, Any]) -> Dict[str, Any]:
    """Reads every key of HP_VALIDATORS from hp once, checks it, then checks the cross-key rules."""
    values = {k: hp[k] for k in HP_VALIDATORS}
    bad = [f"{k}={values[k]!r}" for k, ok in HP_VALIDATORS.items() if not ok(values[k])]
    if bad:
        raise ValueError(f"invalid hyperparameters: {bad}")
    broken = [name for name, rule in CROSS_CHECKS if not rule(values)]
    if broken:
        raise ValueError(f"hyperparameter rules violated: {broken}")
    return values