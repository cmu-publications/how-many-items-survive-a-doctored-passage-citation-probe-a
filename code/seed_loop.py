"""
One seed (run_seed) and the seed loop (run_all_seeds).

Every seed scans its whole pool in its own seeded order. Eligibility, baseline generation and the
arms of one item share one counted exception handler. The determinism regenerations run in their own
counted handler: a failed regeneration counts as a non-identical pair (so the determinism gate sees
it), is counted under determinism_regen_failed and is listed in extra.determinism_errors. If the
per-seed metric computation fails, the seed is logged as a failure that carries its item rows
(seed_failures[].partial), so the finished work reaches results.json.
Seed-start rule: every reserved seed is started; no seed is ever dropped or made conditional on the
remaining time, because the confirmation run needs all of them. Before each seed the loop prints the time
arithmetic (remaining vs seed_start_safety_factor x the measured cost of the previous seed); when the
remaining time is below that, it prints a FLAG and appends an entry to run_info['seed_start_warnings'],
then starts the seed anyway. Only the harness hard budget stop (should_stop, checked before every item)
can end the loop early; that emergency is recorded as a reduced `seeds` component. The partial seed is
never passed to record_seed, but its finished item rows are kept as descriptive data under
unrun_seeds[].partial (the same payload shape as seed_failures[].partial) and counted in
run_info['budget_stop_item_rows_preserved'].
Exceptions raised by the harness itself (should_stop, record_seed) become HarnessAborted, which ends
the loop; the caller then writes no results.
"""
import copy
import math
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from data import pool_for_seed, scan_order
from design import CONDITION_NAMES, PILOT_CACHE, _hp, should_start_seed
from metrics import SELF_END, TOPICAL_END, seed_condition_metrics
from pilot import ITEM_ERRORS, build_seed_donors
from probes import BaselineUneditedArm, Deps, build_arms, check_eligibility, run_item

# Stands in for the answer of a failed determinism regeneration. It contains NUL bytes, which the reader
# never emits, so the pair counts as non-identical and determinism_rate falls below the gate.
DET_REGEN_FAILED = "\x00determinism regeneration failed\x00"
METRIC_ERRORS = (ValueError, FloatingPointError, KeyError, IndexError, ZeroDivisionError)
SEED_START_WARNING = "remaining_below_safety_factor_seed_started_anyway"
BUDGET_STOP_MID_SEED = "budget_stop_mid_seed"
BUDGET_STOP_ROWS_KEY = "budget_stop_item_rows_preserved"


class SeedStopped(Exception):
    """The harness hard budget stop fired inside a seed. The partial seed is never recorded through
    record_seed; partial keeps the item rows that were finished, so they are written under
    unrun_seeds[].partial as descriptive data."""

    def __init__(self, message: str, partial: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.partial: Optional[Dict[str, Any]] = partial


class SeedFailed(Exception):
    """A seed could not be recorded (too many item exceptions, or its metrics failed). partial keeps the
    item rows that were finished, so they are written under seed_failures[].partial."""

    def __init__(self, message: str, item_errors: Optional[List[Dict[str, str]]] = None,
                 partial: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.item_errors: List[Dict[str, str]] = list(item_errors or [])
        self.partial: Optional[Dict[str, Any]] = partial


class HarnessAborted(Exception):
    """The harness raised inside the seed loop; the run stops and results.json is not written."""


def _should_stop(harness: Any) -> bool:
    try:
        return bool(harness.should_stop())
    except Exception as exc:
        raise HarnessAborted(f"harness.should_stop raised {type(exc).__name__}: {exc}") from exc


def record_seed_payload(harness: Any, seed: int, conds: Dict[str, Dict[str, Any]], extra: Dict[str, Any]) -> None:
    missing = [c for c in CONDITION_NAMES if c not in conds]
    if missing:
        raise ValueError(f"seed {seed}: missing conditions {missing}")
    for c, m in conds.items():
        for k, v in m.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"seed {seed}: non-finite metric {c}/{k}={v}")
    try:
        harness.record_seed(seed, conds, expected_conditions=list(CONDITION_NAMES), extra=extra)
    except Exception as exc:
        raise HarnessAborted(f"harness.record_seed failed for seed {seed}: {type(exc).__name__}: {exc}") from exc


def determinism_pairs(sids: Sequence[str], by_id: Dict[str, Dict[str, Any]], baselines: Dict[str, Dict[str, Any]],
                      generate: Callable[[str, Any], str], counters: Counter,
                      seed: int) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """Regenerates each baseline once. A failure is counted, recorded and scored as a non-identical pair."""
    pairs: List[Tuple[str, str]] = []
    errors: List[Dict[str, str]] = []
    for sid in sids:
        item = by_id[sid]
        try:
            regen = generate(item["question"], item["context_docs"])
        except ITEM_ERRORS as exc:
            counters["determinism_regen_failed"] += 1
            errors.append({"sample_id": sid, "error_class": type(exc).__name__, "message": str(exc)})
            print(f"FLAG: seed {seed} determinism regeneration of {sid} failed: {type(exc).__name__}: {exc}; "
                  f"counted as a non-identical pair", flush=True)
            regen = DET_REGEN_FAILED
        pairs.append((str(baselines[sid]["answer"]), regen))
    return pairs, errors


def _pilot_cache_for(design: Dict[str, Any], seed: int) -> Optional[Dict[str, Any]]:
    cache = design.get(PILOT_CACHE)
    if isinstance(cache, dict) and int(cache.get("seed", -1)) == int(seed):
        return cache
    return None


def run_seed(seed: int, design: Dict[str, Any], deps: Deps, datasets: Dict[str, Any], cfg: Any,
             harness: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Raises SeedStopped if the harness hard stop fires mid-seed and SeedFailed past max_item_exceptions
    or when the seed's metrics cannot be computed. In every case the finished item rows ride along in the
    exception's partial payload."""
    hp = _hp(cfg)
    deps.reset_counters()
    counters = deps.counters
    pool = pool_for_seed(seed, hp["seeds"], datasets)
    pool_role = "dev" if int(seed) in {int(s) for s in hp["seeds"]} else "confirmation"
    by_id = {str(it["sample_id"]): it for it in pool}
    scan_cap = design.get("scan_cap")
    cache = _pilot_cache_for(design, seed)
    if cache is not None and cache.get("donors") is not None:
        donors = cache["donors"]
        donor_source = f"pilot of seed {seed} (built once, reused)"
    else:
        donors = build_seed_donors(pool, seed, deps.ner, hp["entity_types"], scan_cap)
        donor_source = "built for this seed"
    deps.set_seed_pool(pool, donors)
    cached_baselines: Dict[str, Any] = dict((cache or {}).get("baselines") or {})
    cached_rows: Dict[str, Any] = dict((cache or {}).get("rows") or {})
    cached_counters: Dict[str, Any] = dict((cache or {}).get("counters") or {})
    reused: Dict[str, List[str]] = {"baselines": [], "rows": []}
    record_design = {k: v for k, v in design.items() if k != PILOT_CACHE}
    arms = build_arms(hp)
    quota_state = {k: [0, int(v)] for k, v in design["quota"].items()}
    n_target = int(design["n_items"])
    exclusions: Counter = Counter()
    rows_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONDITION_NAMES}
    item_rows: List[Dict[str, Any]] = []
    item_errors: List[Dict[str, str]] = []
    included: List[str] = []
    baselines: Dict[str, Dict[str, Any]] = {}
    scan_stats = {"n_generated": 0, "n_parse_ok": 0}
    n_scanned, capped = 0, False
    for sid in scan_order(pool, seed):
        if len(included) >= n_target:
            break
        if scan_cap is not None and n_scanned >= int(scan_cap):
            capped = True
            break
        if _should_stop(harness):
            print(f"FLAG: seed {seed} harness hard budget stop; {len(item_rows)} item rows of "
                  f"{len(included)} included items kept under unrun_seeds[].partial", flush=True)
            raise SeedStopped(f"seed {seed}: harness hard budget stop after {n_scanned} scanned items "
                              f"({len(included)}/{n_target} included); the partial seed is not recorded",
                              partial={"item_rows": item_rows, "included_ids": included,
                                       "n_scanned": n_scanned, "item_errors": item_errors,
                                       "exclusions": dict(exclusions), "counters": dict(counters),
                                       "pool_role": pool_role})
        item = by_id[sid]
        n_scanned += 1
        status = "included"
        stage = "pre_eligibility"
        try:
            # c1/c2 need no generation; a c3 answer here only means "pre-checks passed".
            _ok, pre_label, _p = check_eligibility(item, None, seed, deps, quota_state, arms)
            if pre_label in ("c1_uncovered", "c2_quota_full"):
                status = str(pre_label)
            else:
                stage = "baseline_generation"
                if sid in cached_baselines:
                    baseline = copy.deepcopy(cached_baselines[sid])
                    counters.update((cached_counters.get(sid) or {}).get("baseline", {}))
                    reused["baselines"].append(sid)
                else:
                    baseline = BaselineUneditedArm.generate_baseline(item, deps.generate, deps.tf_slots,
                                                                     int(hp["n_passages"]))
                scan_stats["n_generated"] += 1
                scan_stats["n_parse_ok"] += int(bool(baseline["parse_ok"]))
                stage = "eligibility"
                ok, label, payload = check_eligibility(item, baseline, seed, deps, quota_state, arms)
                if not ok:
                    status = str(label)
                else:
                    stage = "arms"
                    if sid in cached_rows:
                        pairs = [(cond, copy.deepcopy(r)) for cond, r in cached_rows[sid]]
                        counters.update((cached_counters.get(sid) or {}).get("arms", {}))
                        reused["rows"].append(sid)
                    else:
                        rows = run_item(item, baseline, payload, deps.generate, deps.tf_slots, deps.answer_lp,
                                        deps.nli, hp=hp, splitter=deps.splitter, counters=counters, arms=arms)
                        pairs = [(cond, r) for (cond, _s), r in rows.items()]
                    for cond, r in pairs:
                        r["seed"] = seed
                        r["pool"] = pool_role
                        rows_by_cond[cond].append(r)
                        item_rows.append(r)
                    quota_state[item["redundancy"]][0] += 1
                    included.append(sid)
                    baselines[sid] = baseline
        except ITEM_ERRORS as exc:
            status = "error"
            counters["item_exceptions"] += 1
            item_errors.append({"sample_id": sid, "stage": stage, "error_class": type(exc).__name__,
                                "message": str(exc)})
            print(f"FLAG: seed {seed} item {sid} failed at {stage}: {type(exc).__name__}: {exc}", flush=True)
            if counters["item_exceptions"] > int(hp["max_item_exceptions"]):
                raise SeedFailed(f"seed {seed}: {counters['item_exceptions']} item exceptions > "
                                 f"max_item_exceptions={hp['max_item_exceptions']}", item_errors,
                                 partial={"item_rows": item_rows, "included_ids": included,
                                          "n_scanned": n_scanned}) from exc
        if status not in ("included", "error"):
            exclusions[status] += 1
        print(f"seed {seed} item {n_scanned} {sid}: {status} (included {len(included)}/{n_target})", flush=True)
    shortfall = n_target - len(included)
    if shortfall > 0 and not capped:
        counters["pool_exhausted"] += 1
        print(f"FLAG: seed {seed} pool exhausted; {len(included)}/{n_target} items included "
              f"(pool size {len(pool)})", flush=True)
    det_sids = included[: int(design.get("det_items", 0))]
    det_pairs, det_errors = determinism_pairs(det_sids, by_id, baselines, deps.generate, counters, seed)
    alias_offsets = [float(o) for sid in included for o in by_id[sid].get("alias_sentence_offsets", [])]
    extra = {"design": record_design, "hyperparameters": copy.deepcopy(hp), "item_rows": item_rows,
             "counters": dict(counters), "smoke": bool(design.get("smoke", False)),
             "reduced_components": list(design.get("reduced_components", [])),
             "skipped_components": list(design.get("skipped_components", [])),
             "protocol_deviation": bool(shortfall > 0), "shortfall": int(shortfall),
             "n_included": len(included), "n_scanned": n_scanned, "scan_capped": capped,
             "exclusions": dict(exclusions), "item_errors": item_errors, "scan_stats": scan_stats,
             "alias_offsets": alias_offsets, "included_ids": included, "pool_role": pool_role,
             "pool_size": len(pool), "donor_source": donor_source, "pilot_reused": reused,
             "determinism_pairs_planned": len(det_sids), "determinism_errors": det_errors}
    try:
        conds = seed_condition_metrics(rows_by_cond, scan_stats, det_pairs, alias_offsets, counters,
                                       int(hp["offset_bins"]))
    except METRIC_ERRORS as exc:
        print(f"FLAG: seed {seed} metric computation failed: {type(exc).__name__}: {exc}; "
              f"{len(item_rows)} item rows kept under seed_failures[].partial", flush=True)
        raise SeedFailed(f"seed {seed}: metric computation failed: {type(exc).__name__}: {exc}", item_errors,
                         partial=extra) from exc
    extra["counters"] = dict(counters)
    return conds, extra


def _record_seed_reduction(run_info: Dict[str, Any], run_seeds: Sequence[int], i: int, reason: str) -> float:
    """Records the seeds component as reduced to run_seeds[:i]; returns the fraction of seeds used."""
    fraction = i / float(len(run_seeds)) if run_seeds else 0.0
    run_info.setdefault("reduced_components", []).append(
        {"component": "seeds", "planned": [int(s) for s in run_seeds],
         "used": [int(s) for s in run_seeds[:i]], "planned_n": len(run_seeds), "used_n": i,
         "fraction_used": fraction, "reason": reason})
    return fraction


def _seed_start_check(run_info: Dict[str, Any], seed: int, remaining: float, last: Optional[float],
                      factor: float) -> None:
    """Prints the time arithmetic before a seed. Below the safety factor it FLAGs and records a warning in
    run_info['seed_start_warnings']; the seed is started either way (no reserved seed is ever dropped)."""
    required = factor * float(last) if last is not None else None
    if should_start_seed(remaining, last, factor):
        detail = "no measured seed yet" if required is None else f"required {required:.0f}s"
        print(f"seed {seed}: starting; remaining {remaining:.0f}s ({detail}, safety factor {factor})", flush=True)
        return
    run_info.setdefault("seed_start_warnings", []).append(
        {"seed": int(seed), "reason": SEED_START_WARNING, "remaining_sec": remaining,
         "measured_seed_sec": last, "required_sec": required, "safety_factor": factor, "started": True})
    print(f"FLAG: seed {seed}: remaining {remaining:.0f}s < {factor} x measured seed cost "
          f"{float(last or 0.0):.0f}s = {float(required or 0.0):.0f}s; starting it anyway because every "
          f"reserved seed must run; only the harness hard stop can end the loop; recorded in "
          f"seed_start_warnings", flush=True)


def _budget_stop_entry(seed: int, exc: SeedStopped, run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the unrun_seeds entry of a seed stopped mid-run. Its finished item rows are kept under
    partial (never passed to record_seed) and counted in run_info[BUDGET_STOP_ROWS_KEY]."""
    partial = exc.partial
    n_rows = len((partial or {}).get("item_rows") or [])
    run_info[BUDGET_STOP_ROWS_KEY] = int(run_info.get(BUDGET_STOP_ROWS_KEY, 0)) + n_rows
    return {"seed": int(seed), "reason": BUDGET_STOP_MID_SEED, "detail": str(exc), "partial": partial,
            "n_item_rows_preserved": n_rows, "recorded": False}


def run_all_seeds(run_seeds: Sequence[int], design: Dict[str, Any], deps: Deps, datasets: Dict[str, Any],
                  cfg: Any, harness: Any, run_info: Dict[str, Any], budget: float) -> None:
    """Seed loop. Every reserved seed is started: the pre-seed budget check only prints the time arithmetic
    and, below seed_start_safety_factor x the previous seed's cost, records a warning. harness.should_stop()
    is checked before every item (inside run_seed); that hard stop is the only thing that ends the loop
    early. It is recorded as a reduction; the partial seed is not recorded through record_seed, but its
    finished item rows are kept under unrun_seeds[].partial. A harness exception (HarnessAborted)
    propagates so the caller writes nothing."""
    hp = _hp(cfg)
    t_start = float(run_info["t_start"])
    factor = float(hp["seed_start_safety_factor"])
    last: Optional[float] = None
    for i, seed in enumerate(run_seeds):
        remaining = budget - (time.time() - t_start)
        _seed_start_check(run_info, seed, remaining, last, factor)
        t0 = time.time()
        try:
            conds, extra = run_seed(seed, design, deps, datasets, cfg, harness)
            extra["seconds"] = time.time() - t0
            record_seed_payload(harness, seed, conds, extra)
        except SeedStopped as exc:
            entry = _budget_stop_entry(seed, exc, run_info)
            run_info["unrun_seeds"].append(entry)
            run_info["unrun_seeds"].extend({"seed": int(s), "reason": "harness_hard_stop"} for s in run_seeds[i + 1:])
            fraction = _record_seed_reduction(run_info, run_seeds, i,
                                              "harness hard budget stop (emergency); partial seed not recorded, "
                                              "its item rows kept under unrun_seeds[].partial")
            print(f"FLAG: {exc}; {entry['n_item_rows_preserved']} item rows kept under unrun_seeds[].partial "
                  f"({BUDGET_STOP_ROWS_KEY}={run_info[BUDGET_STOP_ROWS_KEY]}); seeds {list(run_seeds[i + 1:])} "
                  f"not run; reduced_components seeds fraction_used={fraction:.2f} ({i}/{len(run_seeds)})",
                  flush=True)
            return
        except HarnessAborted:
            raise
        except Exception as exc:  # logged with traceback; the remaining seeds still run
            run_info["seed_failures"].append({"seed": int(seed), "error_class": type(exc).__name__,
                                              "message": str(exc), "traceback": traceback.format_exc(),
                                              "item_errors": list(getattr(exc, "item_errors", [])),
                                              "partial": getattr(exc, "partial", None)})
            print(f"FLAG: seed {seed} failed ({type(exc).__name__}: {exc}); not recorded, run continues", flush=True)
            last = time.time() - t0
            continue
        last = float(extra["seconds"])
        print(f"seed {seed}: n={extra['n_included']} {TOPICAL_END} itt="
              f"{conds[TOPICAL_END].get('itt_citation_migration_rate')} {SELF_END} itt="
              f"{conds[SELF_END].get('itt_citation_migration_rate')} ({last:.0f}s)", flush=True)
        if "recalibrated_to_sec" not in run_info:
            new_est = (time.time() - t_start) + (len(run_seeds) - i - 1) * last + float(hp["overhead_sec"])
            run_info["recalibrated_to_sec"] = new_est
            print(f"recalibrated_from={float(design.get('est_sec', 0.0)):.0f}s to={new_est:.0f}s", flush=True)