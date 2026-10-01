"""
Pilot timing for the measured-estimate contract (plan item 46).

Times baseline generation plus eligibility on up to pilot_scan_items items of the pilot seed's
pool. Then it times one eligible item through every arm with run_item, with the reliance audit forced
(all eight arms, teacher-forced pass, LOO reliance audit on every arm whether or not the row moved,
ALCE NLI proxy), so measured_item_cost always contains the LOO cost. Finally it times the aggregation
phase (per-seed metrics plus paired contrasts) on pilot rows replicated to the design size.

The pilot also returns what the seed can reuse: the entity-swap donor pool (built once per seed) and,
for each scanned item, its baseline and (first eligible item only) its arm rows with the counter
increments they caused. In smoke mode the seed reuses all of it, so the pilot is the smoke seed's own
scan; in a full run only the donor pool is reused and the pilot rows are discarded.
Also holds the seed plan (seeds(SEEDS) -> run seeds and confirmation seeds) and the list of
reduced components (items_per_seed and determinism_check carry the fraction of items kept).
"""
import copy
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis import compute_contrasts
from data import pool_for_seed, scan_order
from metrics import CONDITION_ORDER, new_counters, seed_condition_metrics
from probes import BaselineUneditedArm, Deps, EntitySwapFoilProbe, build_arms, check_eligibility, run_item

ITEM_ERRORS = (RuntimeError, ValueError, FloatingPointError, KeyError, IndexError)
PILOT_STRATA = ("single_source", "redundant")


def plan_run_seeds(all_seeds: Sequence[int], plan_seeds: Sequence[int], smoke: bool,
                   done: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Run seeds = seeds(SEEDS) (first only in smoke) minus seeds already recorded. Confirmation seeds come
    from the full seeds(SEEDS) list, so they do not depend on which seeds this process runs."""
    ordered = [int(s) for s in all_seeds]
    plan_set = {int(s) for s in plan_seeds}
    confirmation = [s for s in ordered if s not in plan_set]
    done_set = {int(s) for s in done}
    selected = ordered[:1] if smoke else ordered
    return [s for s in selected if s not in done_set], confirmation


def donor_items(pool: Sequence[Dict[str, Any]], seed: int, scan_cap: Optional[int]) -> List[Dict[str, Any]]:
    """Items the donor pool is built from: the whole seed pool, or with a scan cap (smoke) only the first
    scan_cap items of the seed's scan order, i.e. the items the capped scan can reach."""
    if scan_cap is None:
        return list(pool)
    by_id = {str(it["sample_id"]): it for it in pool}
    return [by_id[str(sid)] for sid in list(scan_order(list(pool), seed))[: int(scan_cap)]]


def build_seed_donors(pool: Sequence[Dict[str, Any]], seed: int, ner: Callable, entity_types: Sequence[str],
                      scan_cap: Optional[int]) -> Dict[str, List[Tuple[str, str]]]:
    """The one place a seed's entity-swap donor pool (NER over the donor items) is built."""
    return EntitySwapFoilProbe.build_donor_pool(donor_items(pool, seed, scan_cap), ner, entity_types)


def _counter_delta(before: Counter, after: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in dict(after).items():
        d = int(v) - int(before.get(k, 0))
        if d > 0:
            out[str(k)] = d
    return out


def measure_pilot(deps: Deps, datasets: Dict[str, Any], hp: Dict[str, Any], pilot_seed: int,
                  plan_seeds: Sequence[int], pilot_items: int, scan_limit: Optional[int]) -> Dict[str, Any]:
    """Scan-stage times on >= pilot_items items (more only while no eligible item was found, up to scan_limit)
    and the full per-item arm time on the first eligible item, with the LOO reliance audit forced on every
    arm (force_audit=True) so the LOO cost is always measured. sec_arms is None if no item was eligible."""
    deps.reset_counters()
    pool = pool_for_seed(pilot_seed, plan_seeds, datasets)
    by_id = {str(it["sample_id"]): it for it in pool}
    donors = build_seed_donors(pool, pilot_seed, deps.ner, hp["entity_types"], scan_limit)
    n_donor_items = len(donor_items(pool, pilot_seed, scan_limit))
    deps.set_seed_pool(pool, donors)
    arms = build_arms(hp)
    quota_state = {k: [0, len(pool) + 1] for k in PILOT_STRATA}
    scan_secs: List[float] = []
    gen_secs: List[float] = []
    errors: List[Dict[str, str]] = []
    sec_arms: Optional[float] = None
    eligible_id: Optional[str] = None
    item_rows: List[Tuple[str, Dict[str, Any]]] = []
    reuse: Dict[str, Dict[str, Any]] = {"baselines": {}, "rows": {}, "counters": {}}
    n_eligible = 0
    for sid in scan_order(pool, pilot_seed):
        if len(scan_secs) >= int(pilot_items) and sec_arms is not None:
            break
        if scan_limit is not None and len(scan_secs) >= int(scan_limit):
            break
        item = by_id[sid]
        t0 = time.perf_counter()
        arms_t = 0.0
        stage = "pre_eligibility"
        try:
            _ok, pre, _p = check_eligibility(item, None, pilot_seed, deps, quota_state, arms)
            if pre not in ("c1_uncovered", "c2_quota_full"):
                stage = "baseline_generation"
                before = Counter(deps.counters)
                tg = time.perf_counter()
                baseline = BaselineUneditedArm.generate_baseline(item, deps.generate, deps.tf_slots,
                                                                 int(hp["n_passages"]))
                gen_secs.append(time.perf_counter() - tg)
                reuse["baselines"][sid] = copy.deepcopy(baseline)
                reuse["counters"][sid] = {"baseline": _counter_delta(before, deps.counters), "arms": {}}
                stage = "eligibility"
                ok, _label, payload = check_eligibility(item, baseline, pilot_seed, deps, quota_state, arms)
                if ok:
                    n_eligible += 1
                    if sec_arms is None:
                        stage = "arms"
                        before = Counter(deps.counters)
                        ta = time.perf_counter()
                        rows = run_item(item, baseline, payload, deps.generate, deps.tf_slots, deps.answer_lp,
                                        deps.nli, hp=hp, splitter=deps.splitter, counters=deps.counters, arms=arms,
                                        force_audit=True)
                        arms_t = time.perf_counter() - ta
                        sec_arms = arms_t
                        eligible_id = sid
                        item_rows = [(cond, dict(r)) for (cond, _s), r in rows.items()]
                        reuse["rows"][sid] = [(cond, copy.deepcopy(r)) for (cond, _s), r in rows.items()]
                        reuse["counters"][sid]["arms"] = _counter_delta(before, deps.counters)
        except ITEM_ERRORS as exc:
            errors.append({"sample_id": sid, "stage": stage, "error_class": type(exc).__name__, "message": str(exc)})
            print(f"FLAG: pilot item {sid} failed at {stage}: {type(exc).__name__}: {exc}", flush=True)
            if len(errors) > int(hp["max_item_exceptions"]):
                raise RuntimeError(f"pilot: {len(errors)} item exceptions > max_item_exceptions="
                                   f"{hp['max_item_exceptions']}; first: {errors[0]}") from exc
        scan_secs.append(time.perf_counter() - t0 - arms_t)
    deps.reset_counters()
    alias_offsets = [float(o) for o in by_id[eligible_id].get("alias_sentence_offsets", [])] if eligible_id else []
    return {"pilot_seed": int(pilot_seed), "pilot_items": int(pilot_items), "n_scanned": len(scan_secs),
            "n_eligible": n_eligible,
            "eligible_rate_measured": (n_eligible / len(scan_secs)) if scan_secs else None,
            "sec_scan_mean": float(np.mean(scan_secs)) if scan_secs else None,
            "sec_gen_mean": float(np.mean(gen_secs)) if gen_secs else None,
            "sec_arms": sec_arms, "eligible_item": eligible_id, "item_rows": item_rows,
            "alias_offsets": alias_offsets, "errors": errors, "n_donor_items": n_donor_items,
            "donors": donors, "reuse": reuse, "loo_forced": True,
            "stages": ("pre_eligibility + baseline_generation + eligibility (scan); run_item with force_audit=True "
                       "(all arms + teacher-forced pass + LOO reliance audit on every arm + ALCE NLI proxy)")}


def time_aggregation(item_rows: Sequence[Tuple[str, Dict[str, Any]]], n_items: int, n_seeds: int,
                     hp: Dict[str, Any], alias_offsets: Sequence[float]) -> Dict[str, Any]:
    """Times seed_condition_metrics per seed and compute_contrasts on the pilot rows replicated to
    n_items x n_seeds. Uses its own counters; any failure is recorded and printed."""
    counters = new_counters()
    errors: List[Dict[str, str]] = []
    pooled: List[Dict[str, Any]] = []
    n_items, n_seeds = max(1, int(n_items)), max(1, int(n_seeds))
    t0 = time.perf_counter()
    for s in range(n_seeds):
        by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONDITION_ORDER}
        for j in range(n_items):
            for cond, r in item_rows:
                rr = dict(r, sample_id=f"{r.get('sample_id')}#rep{j}", seed=s, pool="dev")
                by_cond.setdefault(cond, []).append(rr)
                pooled.append(rr)
        det = [("a", "a")] * min(int(hp["det_items"]), n_items)
        try:
            seed_condition_metrics(by_cond, {"n_generated": n_items, "n_parse_ok": n_items}, det,
                                   list(alias_offsets), counters, int(hp["offset_bins"]))
        except Exception as exc:  # timing only; recorded in design.fixed_phase.errors and printed
            errors.append({"stage": "seed_metrics", "error_class": type(exc).__name__, "message": str(exc)})
    try:
        compute_contrasts(pooled, hp, counters, list(alias_offsets))
    except Exception as exc:  # timing only; recorded in design.fixed_phase.errors and printed
        errors.append({"stage": "contrasts", "error_class": type(exc).__name__, "message": str(exc)})
    for e in errors:
        print(f"FLAG: pilot aggregation timing {e['stage']} raised {e['error_class']}: {e['message']}", flush=True)
    return {"seconds": time.perf_counter() - t0, "n_rows": len(pooled), "errors": errors}


def _fraction_kept(used: int, planned: int) -> float:
    """Fraction of the planned items kept; planned is validated positive by hp_schema."""
    if int(planned) <= 0:
        raise ValueError(f"planned item count must be positive, got {planned}")
    return float(int(used)) / float(int(planned))


def reduced_components(hp: Dict[str, Any], design: Dict[str, Any], run_seeds: Sequence[int],
                       smoke: bool) -> List[Dict[str, Any]]:
    """Reduced components; items_per_seed and determinism_check record fraction_kept = used / planned items."""
    reason = "smoke" if smoke else "budget"
    n = int(design["n_items"])
    det = int(design.get("det_items", hp["det_items"]))
    n_plan = int(hp["n_max_items"])
    det_plan = int(hp["det_items"])
    out: List[Dict[str, Any]] = []
    if n < n_plan:
        out.append({"component": "items_per_seed", "fraction_kept": _fraction_kept(n, n_plan),
                    "planned": n_plan, "used": n, "reason": reason,
                    "fraction_basis": "eligible items per seed kept / n_max_items"})
    if det < det_plan:
        out.append({"component": "determinism_check", "fraction_kept": _fraction_kept(det, det_plan),
                    "planned": det_plan, "used": det, "reason": reason,
                    "fraction_basis": "determinism regeneration items kept / det_items"})
    if smoke:
        out += [{"component": "seeds", "planned": list(hp["seeds"]), "used": [int(s) for s in run_seeds],
                 "reason": reason},
                {"component": "scan", "planned": "whole pool", "used": design.get("scan_cap"), "reason": reason},
                {"component": "donor_pool", "planned": "every item of the seed pool",
                 "used": f"first {design.get('scan_cap')} items of the seed scan order", "reason": reason},
                {"component": "calibration_pilot", "planned": int(hp["pilot_scan_items"]),
                 "used": (design.get("pilot") or {}).get("pilot_items"), "reason": reason,
                 "note": "the pilot scan is the smoke seed's own scan; the seed reuses its baselines, arm rows "
                         "and donor pool"}]
    return out