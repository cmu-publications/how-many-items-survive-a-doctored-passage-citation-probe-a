"""
Design sizing: smoke plan, pool sufficiency, the measured pilot and the fixed per-seed design.

Plan item 46. Unless harness.resumed_design is set, a measured pilot runs before the seed loop. It runs on
the first seed this process executes, and that seed always comes from experiment_harness.seeds(SEEDS) (the
run_seeds list); a plan seed id from HYPERPARAMETERS is never used as a random seed. The pilot times
baseline generation + eligibility on up to pilot_scan_items items and one eligible item through every arm,
the teacher-forced pass, the LOO pass and ALCE NLI, then times aggregation on replicated pilot rows. It
prints measured_item_cost, fixed_phase_cost and TIME_ESTIMATE, then chooses n by the Analysis formula:
the largest n in [n_min_items, n_max_items] (full run: [20, 50]) whose estimate fits the remaining budget
for the plan's development seed count (len(HYPERPARAMETERS["seeds"])). The seed count used for sizing never
depends on which seeds this process runs, so a confirmation run (the reserved seeds) sizes on the same basis
as the development run. The chosen n is then checked against the seeds this run actually executes; if that
does not fit (a run with more seeds than the plan), it prints BUDGET_INSUFFICIENT and exits 1 rather than
changing n. If even n_min does not fit, it prints BUDGET_INSUFFICIENT and exits 1 before any seed runs. A
chosen n below n_max_items is recorded in reduced_components (items_per_seed) and
scale_factor = n / n_max_items.
Smoke mode fits n in [smoke_min_items, smoke_n_items] for its single seed. Its pilot scans that seed's own
scan order up to the smoke scan cap, and the pilot's baselines, its eligible item's arm rows and its donor
pool go to the seed through design["pilot_cache"] and are reused, not recomputed. The cache is never
recorded with the seed.
A pilot that finds no eligible item prints every recorded counter (nested counter blocks flattened to
dotted keys, non-numeric values reported verbatim, nothing dropped) and the pilot's scalar fields on stdout
and stderr, then exits 1 before any seed runs.
"""
import math
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data import pool_for_seed
from metrics import CONDITION_ORDER, SELF_END, TOPICAL_END
from pilot import measure_pilot, reduced_components, time_aggregation

CONDITION_NAMES = list(CONDITION_ORDER)
ANALYSIS_COMPONENTS = ["reliance_audit", "alce_citation_proxy", "determinism_check", "paired_contrasts"]
REDUNDANCY_STRATA = ("single_source", "redundant")
PILOT_CACHE = "pilot_cache"
PILOT_SEED_SOURCE = "experiment_harness.seeds(SEEDS): first seed this run executes"
SIZING_SEED_BASIS = "plan development seed count len(HYPERPARAMETERS['seeds']); independent of the run's seed set"


def _hp(cfg: Any) -> Dict[str, Any]:
    return dict(cfg) if isinstance(cfg, dict) else dict(vars(cfg))


def smoke_config(cfg: Any, smoke: bool) -> Dict[str, Any]:
    hp = _hp(cfg)
    return {"components": list(CONDITION_NAMES) + list(ANALYSIS_COMPONENTS),
            "n_max": int(hp["smoke_n_items"]) if smoke else int(hp["n_max_items"]),
            "n_min": int(hp["smoke_min_items"]) if smoke else int(hp["n_min_items"]),
            "scan_cap": int(hp["smoke_scan_cap"]) if smoke else None,
            "seeds": list(hp["seeds"])[:1] if smoke else list(hp["seeds"]),
            "smoke": bool(smoke)}


def parse_rc_seeds(value: Optional[str]) -> Optional[List[int]]:
    """Diagnostic only: compared against experiment_harness.seeds(SEEDS), which is always the list used."""
    if value is None or not value.strip():
        return None
    return [int(x) for x in value.replace(",", " ").split()]


def quota_for(n_items: int) -> Dict[str, int]:
    return {"single_source": (int(n_items) + 1) // 2, "redundant": int(n_items) // 2}


def pool_sufficiency(pool_sizes: Dict[int, int], strata_counts: Dict[int, Dict[str, int]], n_items: int,
                     quota: Dict[str, int], eligible_rate: float) -> Tuple[bool, List[str]]:
    """Pre-run check that every seed's pool can plausibly yield n_items eligible items."""
    if not eligible_rate > 0:
        return False, [f"POOL_INSUFFICIENT: eligible_rate_prior={eligible_rate} must be > 0"]
    required = int(math.ceil(int(n_items) / float(eligible_rate)))
    ok = True
    lines: List[str] = []
    for seed in sorted(pool_sizes):
        size = int(pool_sizes[seed])
        counts = strata_counts.get(seed, {})
        size_ok = size >= required
        lines.append(f"pool seed {seed}: size={size} required=ceil({n_items}/{eligible_rate})={required} "
                     f"expected_eligible={size * eligible_rate:.1f} -> {'ok' if size_ok else 'INSUFFICIENT'}")
        if not size_ok:
            ok = False
            lines.append(f"POOL_INSUFFICIENT: seed {seed} pool of {size} < {required} scans needed for "
                         f"{n_items} eligible items at rate {eligible_rate}")
        for stratum, q in sorted(quota.items()):
            c = int(counts.get(stratum, 0))
            if c < int(q):
                ok = False
                lines.append(f"POOL_INSUFFICIENT: seed {seed} stratum {stratum} has {c} items < quota {q}")
            elif c * float(eligible_rate) < int(q):
                lines.append(f"FLAG: seed {seed} stratum {stratum}: expected eligible {c * eligible_rate:.1f} "
                             f"< quota {q}; a shortfall is likely")
    return ok, lines


def choose_n_items(sec_arms_per_item: float, sec_scan_per_item: float, eligible_rate: float, n_seeds: int,
                   sec_det_regen: float, overhead_sec: float, budget_sec: float, n_max: int, n_min: int,
                   det_items: int) -> Tuple[Optional[int], float, Dict[str, Any]]:
    """Analysis formula: est(n) = overhead + n_seeds * (det_items * sec_det_regen + n * (arms + scan));
    the largest n <= n_max that fits the budget, or None when even n_min does not fit."""
    per_seed_fixed = float(det_items) * float(sec_det_regen)
    per_item = float(sec_arms_per_item) + float(sec_scan_per_item)

    def est(n: int) -> float:
        return float(overhead_sec) + float(n_seeds) * (per_seed_fixed + n * per_item)

    avail = (float(budget_sec) - float(overhead_sec)) / max(1, int(n_seeds)) - per_seed_fixed
    n_fit = int(math.floor(avail / per_item)) if per_item > 0 else int(n_max)
    n = min(int(n_max), n_fit)
    info = {"per_item_sec": per_item, "per_seed_fixed_sec": per_seed_fixed, "n_fit": n_fit,
            "expected_scanned_per_seed": (int(math.ceil(max(n, 0) / eligible_rate)) if eligible_rate > 0 else None)}
    if n < int(n_min) or est(n) > float(budget_sec):
        return None, est(max(int(n_min), n)), info
    return n, est(n), info


def should_start_seed(remaining_sec: float, last_seed_sec: Optional[float], safety_factor: float) -> bool:
    """Diagnostic only: False means the seed is expected to overrun. It never drops or shrinks a seed."""
    if last_seed_sec is None:
        return True
    return float(last_seed_sec) * float(safety_factor) <= float(remaining_sec)


def sizing_basis(hp: Dict[str, Any], run_seeds: Sequence[int], smoke: bool) -> Tuple[int, int]:
    """(pilot_seed, n_seeds the item count is sized for). The pilot seed is always run_seeds[0], a seed from
    experiment_harness.seeds(SEEDS); it is never a plan seed id used as a random seed. The seed count is the
    plan's development seed count len(hp["seeds"]) whatever run_seeds holds, so the chosen n never changes
    with the seed set (a confirmation run sizes exactly like the development run). Smoke sizes for 1 seed."""
    if not run_seeds:
        raise ValueError(f"sizing_basis needs at least one run seed (plan seeds {list(hp['seeds'])})")
    plan_seeds = list(hp["seeds"])
    if not plan_seeds:
        raise ValueError("sizing_basis needs a non-empty plan seed list hp['seeds']")
    return int(run_seeds[0]), (1 if smoke else len(plan_seeds))


def _item_bounds(hp: Dict[str, Any], plan: Dict[str, Any], smoke: bool) -> Tuple[int, int, str]:
    """(n_max, n_min, rule). Full: largest fitting n in [n_min_items, n_max_items]. Smoke: in [2, 5]."""
    if smoke:
        return (int(plan["n_max"]), int(plan["n_min"]),
                f"smoke: largest fitting n in [{plan['n_min']}, {plan['n_max']}]")
    n_max, n_min = int(hp["n_max_items"]), int(hp["n_min_items"])
    return n_max, n_min, (f"full: largest fitting n in [{n_min}, {n_max}] by the Analysis formula for the "
                          f"plan's {len(list(hp['seeds']))} development seeds")


def _is_count(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _flatten_counters(counters: Mapping[Any, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """Every counter entry as (dotted key, value), sorted by key. Nested mappings are flattened; a value that
    is not a finite number is kept verbatim (repr) so nothing the pilot recorded is dropped from the report."""
    out: List[Tuple[str, Any]] = []
    for k in sorted(counters, key=str):
        v = counters[k]
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.extend(_flatten_counters(v, prefix=f"{key}."))
        elif isinstance(v, bool):
            out.append((key, int(v)))
        elif _is_count(v):
            out.append((key, int(v) if float(v).is_integer() else float(v)))
        else:
            out.append((key, f"<non-count {type(v).__name__}: {v!r}>"))
    return out


def _pilot_failure_summary(reuse: Any) -> str:
    counters = reuse.get("counters") if isinstance(reuse, Mapping) else None
    if not counters:
        return "no fallback counters recorded by the pilot"
    if not isinstance(counters, Mapping):
        return f"pilot counters of unexpected type {type(counters).__name__}: {counters!r}"
    flat = _flatten_counters(counters)
    nonzero = {k: v for k, v in flat if not (_is_count(v) and float(v) == 0.0)}
    return f"pilot fallback/error counters: {nonzero}" if nonzero else "all pilot fallback counters are zero"


def _pilot_scalar_summary(pilot: Mapping[str, Any]) -> str:
    """The pilot's scalar fields (counts, rates, ids), for the refusal message."""
    parts = [f"{k}={v!r}" for k, v in sorted(pilot.items(), key=lambda kv: str(kv[0]))
             if v is None or isinstance(v, (bool, int, float, str))]
    return "pilot fields: " + (", ".join(parts) if parts else "none")


def _refuse(message: str) -> None:
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)
    sys.exit(1)


def pilot_and_design(harness: Any, deps: Any, datasets: Dict[str, Any], cfg: Any, run_seeds: Sequence[int],
                     confirmation_seeds: Sequence[int], smoke: bool, budget: float, t_start: float,
                     dtype_name: str) -> Dict[str, Any]:
    """Plan item 46. Returns harness.resumed_design when set. Otherwise it runs the measured pilot on the
    first harness seed of this run, prints measured_item_cost / fixed_phase_cost / TIME_ESTIMATE and
    chooses n by the Analysis formula for the plan's development seed count. The chosen n is then checked
    against the seeds this run executes. If either check fails it prints BUDGET_INSUFFICIENT and exits 1
    before any seed runs or any result is written; n is never refitted to the run's seed set."""
    resumed = getattr(harness, "resumed_design", None)
    if resumed:
        design = dict(resumed)
        print(f"resumed design: n_items/seed={design.get('n_items')} est={design.get('est_sec')}s; pilot skipped, "
              f"sizing taken from harness.resumed_design", flush=True)
        return design
    hp = _hp(cfg)
    plan = smoke_config(cfg, smoke)
    pilot_seed, n_seeds = sizing_basis(hp, run_seeds, smoke)
    n_run_seeds = 1 if smoke else len(run_seeds)
    n_max, n_min, n_rule = _item_bounds(hp, plan, smoke)
    # Smoke: the pilot is the smoke seed's own scan, so it may use the whole capped scan to find its item.
    pilot_items = int(plan["scan_cap"]) if smoke else int(hp["pilot_scan_items"])
    pilot = measure_pilot(deps, datasets, hp, pilot_seed, hp["seeds"], pilot_items, plan["scan_cap"])
    donors = pilot.pop("donors")
    reuse = pilot.pop("reuse")
    if pilot["sec_arms"] is None:
        _refuse(f"BUDGET_INSUFFICIENT: the pilot scanned {pilot['n_scanned']} items of seed {pilot_seed} (limit "
                f"{pilot_items}) without an eligible item, so the per-item cost cannot be measured; "
                f"{_pilot_failure_summary(reuse)}; {_pilot_scalar_summary(pilot)}; "
                f"refusing to start, no results written")
    agg = time_aggregation(pilot.pop("item_rows"), n_max, n_seeds, hp, pilot["alias_offsets"])
    elig = float(hp["eligible_rate_prior"])
    fixed = float(agg["seconds"])
    overhead = float(hp["overhead_sec"]) + fixed
    det_upper = min(int(hp["det_items"]), n_max)
    remaining = float(budget) - (time.time() - float(t_start))
    chosen, est, info = choose_n_items(pilot["sec_arms"], pilot["sec_scan_mean"] / elig, elig, n_seeds,
                                       pilot["sec_gen_mean"], overhead, remaining, n_max, n_min, det_upper)
    n_shown = n_min if chosen is None else int(chosen)
    est_run = overhead + float(n_run_seeds) * (float(info["per_seed_fixed_sec"])
                                               + n_shown * float(info["per_item_sec"]))
    print(f"measured_item_cost={info['per_item_sec']:.2f}s from arms={pilot['sec_arms']:.2f}s (all "
          f"{len(CONDITION_NAMES)} arms + teacher-forced + LOO + ALCE NLI on item {pilot['eligible_item']}) + "
          f"scan={pilot['sec_scan_mean']:.2f}s/eligible_rate_prior {elig} (baseline generation + eligibility, mean "
          f"over {pilot['n_scanned']} scanned; measured eligible rate {pilot['eligible_rate_measured']})", flush=True)
    print(f"fixed_phase_cost={overhead:.2f}s from aggregation timing {fixed:.2f}s (seed metrics + contrasts on "
          f"{agg['n_rows']} replicated pilot rows) + overhead {hp['overhead_sec']}s", flush=True)
    print(f"TIME_ESTIMATE: {est:.0f}s = {overhead:.0f}s + {n_seeds} sizing seeds x ({det_upper} det x "
          f"{pilot['sec_gen_mean']:.2f}s + n={n_shown} x {info['per_item_sec']:.2f}s); item rule: {n_rule}; "
          f"sizing seed basis: {SIZING_SEED_BASIS if not smoke else 'smoke: 1 seed'}; "
          f"this run executes {n_run_seeds} seed(s): estimate {est_run:.0f}s; "
          f"n_fit={info['n_fit']}; pilot seed {pilot_seed} ({PILOT_SEED_SOURCE}); "
          f"remaining budget {remaining:.0f}s", flush=True)
    if chosen is None:
        _refuse(f"BUDGET_INSUFFICIENT: n_min={n_min} items/seed for {n_seeds} sizing seed(s) needs {est:.0f}s > "
                f"remaining {remaining:.0f}s (n_fit={info['n_fit']}); refusing to start, no results written")
    n = int(n_shown)
    if not n_min <= n <= n_max:
        raise ValueError(f"design n_items={n} outside [{n_min}, {n_max}]")
    if est_run > remaining:
        _refuse(f"BUDGET_INSUFFICIENT: n={n} items/seed (sized for the plan's {n_seeds} development seeds) over "
                f"the {n_run_seeds} seeds this run executes needs {est_run:.0f}s > remaining {remaining:.0f}s; n is "
                f"not refitted to the run's seed set; refusing to start, no results written")
    if n < n_max:
        print(f"FLAG: items_per_seed reduced to n={n} < planned {n_max} by the budget (recorded in "
              f"reduced_components; scale_factor={n / float(hp['n_max_items']):.2f})", flush=True)
    cache: Dict[str, Any] = {"seed": int(pilot_seed), "donors": donors}
    if smoke:
        cache.update(baselines=reuse["baselines"], rows=reuse["rows"], counters=reuse["counters"])
    plan_set = {int(s) for s in hp["seeds"]}
    return {"n_items": n, "quota": quota_for(n), "n_items_rule": n_rule, "dtype": dtype_name,
            "reader_dtype": dtype_name, "aux_device": "cpu", "primary_condition": TOPICAL_END,
            "baseline_condition": SELF_END,
            "scale_factor": n / float(hp["n_max_items"]), "scan_cap": plan["scan_cap"],
            "det_items": min(int(hp["det_items"]), n), "est_sec": est_run, "est_sec_sizing_basis": est,
            "budget_info": info, "pilot": pilot,
            "fixed_phase": {"seconds": fixed, "n_rows": agg["n_rows"], "errors": agg["errors"]},
            "smoke": bool(smoke), "components": plan["components"], "skipped_components": [],
            "confirmation_seeds": [int(s) for s in confirmation_seeds], "run_seeds": [int(s) for s in run_seeds],
            "sizing": {"pilot_seed": int(pilot_seed), "pilot_seed_source": PILOT_SEED_SOURCE,
                       "pilot_pool": "dev" if int(pilot_seed) in plan_set else "confirmation",
                       "n_seeds_for_sizing": int(n_seeds),
                       "n_seeds_sizing_basis": "smoke: 1 seed" if smoke else SIZING_SEED_BASIS,
                       "n_seeds_this_run": int(n_run_seeds),
                       "est_sec_this_run": est_run,
                       "pilot_scan_limit": int(pilot_items),
                       "n_bounds": [int(n_min), int(n_max)],
                       "rule": ("smoke: pilot = the smoke seed's own capped scan, 1 seed, n fitted" if smoke else
                                "full: n = largest fitting value in [n_min_items, n_max_items] for the plan's "
                                "development seed count; pilot on the first run seed; the run's own seed count "
                                "is only checked against the budget and never changes n")},
            "pilot_reuse": ("baselines, arm rows and donor pool reused by the smoke seed" if smoke
                            else "donor pool reused by the pilot seed; pilot rows discarded"),
            "item_pool": "whole pool per seed (overlapping across seeds); paired by (seed, sample_id)",
            PILOT_CACHE: cache}


def finalize_design(design: Dict[str, Any], hp: Dict[str, Any], run_seeds: Sequence[int],
                    smoke: bool) -> Dict[str, Any]:
    """The design block recorded with every seed. It always names primary_condition=TOPICAL_END and
    baseline_condition=SELF_END (a resumed design naming other arms is an error), skipped_components and
    reduced_components. A resumed design keeps the reduced_components it was recorded with."""
    out = dict(design)
    for key, want in (("primary_condition", TOPICAL_END), ("baseline_condition", SELF_END)):
        have = out.get(key)
        if have is not None and have != want:
            raise ValueError(f"design {key}={have!r} contradicts the plan ({want!r})")
        out[key] = want
    out.setdefault("skipped_components", [])
    out.setdefault("smoke", bool(smoke))
    if "reduced_components" not in out:
        out["reduced_components"] = reduced_components(hp, out, run_seeds, smoke)
    return out


def check_pools(run_seeds: Sequence[int], plan_seeds: Sequence[int], datasets: Dict[str, Any], n_items: int,
                eligible_rate: float) -> Tuple[bool, List[str]]:
    sizes: Dict[int, int] = {}
    strata: Dict[int, Dict[str, int]] = {}
    for seed in run_seeds:
        pool = pool_for_seed(int(seed), plan_seeds, datasets)
        sizes[int(seed)] = len(pool)
        strata[int(seed)] = {r: sum(1 for it in pool if it.get("redundancy") == r) for r in REDUNDANCY_STRATA}
    return pool_sufficiency(sizes, strata, n_items, quota_for(n_items), eligible_rate)