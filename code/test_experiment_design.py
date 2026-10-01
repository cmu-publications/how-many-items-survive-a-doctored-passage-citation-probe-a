import time
from typing import Any, Dict

import pytest

import design as design_module
from data import pool_for_seed, scan_order
from main import (CONDITION_NAMES, HYPERPARAMETERS, PLAN_ENTITY_TYPES, aggregate_and_write, build_config,
                  choose_n_items, finalize_design, pilot_and_design, quota_for, run_all_seeds, run_seed,
                  sizing_basis, smoke_config)
from metrics import SELF_END, TOPICAL_END
from pilot import build_seed_donors, donor_items, plan_run_seeds, reduced_components
from probes import EntitySwapFoilProbe
from testing_fakes import FakeHarness, answer_table, fake_ner, make_fake_deps, toy_datasets, toy_pool


def _run_info(t_start: float) -> Dict[str, Any]:
    return {"t_start": t_start, "unrun_seeds": [], "seed_failures": []}


# ---- plan items 46, 53: smoke mode, seed plan, measured estimate ---------------------------------
def test_smoke_config_keeps_components_and_floor_of_two():
    cfg = build_config(HYPERPARAMETERS)
    s = smoke_config(cfg, True)
    assert s["smoke"] and s["seeds"] == [0] and s["n_max"] == 5 and s["n_min"] == 2 and s["scan_cap"] == 40
    assert all(c in s["components"] for c in CONDITION_NAMES) and "alce_citation_proxy" in s["components"]
    p = smoke_config(cfg, False)
    assert not p["smoke"] and p["n_max"] == 50 and p["n_min"] == 20 and p["scan_cap"] is None


def test_seed_plan_confirmation_seeds_survive_restart():
    ds = toy_datasets()
    plan = HYPERPARAMETERS["seeds"]
    allseeds = [0, 1, 2, 3, 4, 7, 9]
    run, conf = plan_run_seeds(allseeds, plan, False, [])
    assert run == allseeds and conf == [7, 9]
    run2, conf2 = plan_run_seeds(allseeds, plan, False, [0, 1, 2, 3, 4, 7])
    assert run2 == [9] and conf2 == conf
    assert pool_for_seed(9, plan, ds) == pool_for_seed(9, plan, ds)
    run3, conf3 = plan_run_seeds(allseeds, plan, True, [])
    assert run3 == [0] and conf3 == conf


def test_sizing_basis_pilots_on_first_harness_run_seed():
    """The pilot seed is the first seed this run executes (from experiment_harness.seeds), never the plan's
    hp["seeds"][0]. The seed count the item count is sized for is the plan's development seed count
    len(HYPERPARAMETERS["seeds"]) and never depends on the run's seed set (Seed Sets contract); smoke sizes
    for its single seed."""
    n_plan = len(HYPERPARAMETERS["seeds"])
    assert n_plan == 5
    assert sizing_basis(HYPERPARAMETERS, [0, 1, 2, 3, 4], False) == (0, 5)
    assert sizing_basis(HYPERPARAMETERS, [7, 8, 9], False) == (7, 5)
    assert sizing_basis(HYPERPARAMETERS, [3, 4, 7], False) == (3, 5)
    assert sizing_basis(HYPERPARAMETERS, [7], True) == (7, 1)
    with pytest.raises(ValueError):
        sizing_basis(HYPERPARAMETERS, [], False)


def test_sizing_seed_count_never_follows_the_run_seed_set():
    """Fails under the rejected rule where n_seeds = len(run_seeds): run seed sets of different lengths
    must all size on the plan's development seed count."""
    n_plan = len(HYPERPARAMETERS["seeds"])
    for run_seeds in ([9], [7, 8, 9], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7]):
        pilot_seed, n_seeds = sizing_basis(HYPERPARAMETERS, run_seeds, False)
        assert pilot_seed == run_seeds[0]
        assert n_seeds == n_plan


def test_choose_n_items_zero_cost_still_checks_budget():
    n, est, _ = choose_n_items(0.0, 0.0, 0.3, 1, 0.0, 600.0, 0.0, 5, 2, 5)
    assert n is None and est >= 600.0
    n, _e, _i = choose_n_items(0.0, 0.0, 0.3, 1, 0.0, 600.0, 1e6, 5, 2, 5)
    assert n == 5


def _pilot_cfg():
    return build_config(dict(HYPERPARAMETERS, bootstrap_resamples=20))


def _fake_pilot(monkeypatch, seen):
    """Measured pilot with known costs: arms 7s, scan 0.9s (/0.3 -> 3s), regeneration 1s, aggregation 0s.
    Per item 10s, per seed fixed 20 det x 1s = 20s, overhead 600s."""
    def fake_measure(deps, datasets, hp, pilot_seed, plan_seeds, pilot_items, scan_cap):
        seen.append(int(pilot_seed))
        return {"sec_arms": 7.0, "sec_scan_mean": 0.9, "sec_gen_mean": 1.0, "n_scanned": 4,
                "eligible_item": "q1", "eligible_rate_measured": 0.25, "alias_offsets": [],
                "donors": {}, "reuse": {"counters": {}}, "item_rows": []}

    def fake_agg(item_rows, n_items, n_seeds, hp, alias_offsets):
        return {"seconds": 0.0, "n_rows": 0, "errors": 0}

    monkeypatch.setattr(design_module, "measure_pilot", fake_measure)
    monkeypatch.setattr(design_module, "time_aggregation", fake_agg)


def test_full_run_chooses_n_by_the_analysis_formula(monkeypatch, capsys):
    """Plan item 46: a full run picks the largest n in [20, 50] whose estimate fits, not a fixed 50.
    5 seeds, 10s/item, 20s/seed fixed, 600s overhead: budget 2225s -> n=30; 1e9s -> n=50."""
    seen = []
    _fake_pilot(monkeypatch, seen)
    cfg = _pilot_cfg()
    seeds = [0, 1, 2, 3, 4]
    d = pilot_and_design(FakeHarness(), None, {}, cfg, seeds, [], False, 2225.0, time.time(), "float32")
    assert d["n_items"] == 30 and d["quota"] == {"single_source": 15, "redundant": 15}
    assert d["scale_factor"] == pytest.approx(0.6) and d["sizing"]["n_bounds"] == [20, 50]
    out = capsys.readouterr().out
    assert "TIME_ESTIMATE:" in out and "items_per_seed reduced to n=30" in out
    red = reduced_components(HYPERPARAMETERS, d, seeds, False)
    assert any(r["component"] == "items_per_seed" for r in red)
    big = pilot_and_design(FakeHarness(), None, {}, cfg, seeds, [], False, 1e9, time.time(), "float32")
    assert big["n_items"] == 50 and big["scale_factor"] == pytest.approx(1.0)


def test_full_run_budget_insufficient_only_below_n_min(monkeypatch, capsys):
    """Budget 1675s fits only n=19 < n_min=20: BUDGET_INSUFFICIENT and exit 1 before any seed runs."""
    _fake_pilot(monkeypatch, [])
    h = FakeHarness()
    with pytest.raises(SystemExit) as ei:
        pilot_and_design(h, None, {}, _pilot_cfg(), [0, 1, 2, 3, 4], [], False, 1675.0, time.time(), "float32")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "TIME_ESTIMATE:" in out and "BUDGET_INSUFFICIENT: n_min=20" in out
    assert getattr(h, "written", None) is None and h.seed_records() == []


def test_pilot_seed_comes_from_harness_run_seeds(monkeypatch):
    """A confirmation-only run [7, 8, 9] pilots on seed 7 (a harness seed), never on hp["seeds"][0]=0."""
    seen = []
    _fake_pilot(monkeypatch, seen)
    d = pilot_and_design(FakeHarness(), None, {}, _pilot_cfg(), [7, 8, 9], [7, 8, 9], False, 1e9, time.time(),
                         "float32")
    assert seen == [7]
    assert d["sizing"]["pilot_seed"] == 7 and d["sizing"]["pilot_pool"] == "confirmation"
    assert d["pilot_cache"]["seed"] == 7 and "experiment_harness" in d["sizing"]["pilot_seed_source"]
    assert d["sizing"]["n_seeds_for_sizing"] == len(HYPERPARAMETERS["seeds"])
    assert d["sizing"]["n_seeds_this_run"] == 3


def test_pilot_and_design_prints_estimate_and_fixes_design(capsys):
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    design = pilot_and_design(FakeHarness(), deps, toy_datasets(), _pilot_cfg(), [0], [], True, 1e9,
                              time.time(), "float32")
    out = capsys.readouterr().out
    assert "measured_item_cost=" in out and "fixed_phase_cost=" in out and "TIME_ESTIMATE:" in out
    n = design["n_items"]
    assert 2 <= n <= 5 and sum(design["quota"].values()) == n
    assert design["primary_condition"] == TOPICAL_END and design["baseline_condition"] == SELF_END
    assert design["aux_device"] == "cpu" and design["dtype"] == "float32"
    assert design["scale_factor"] == pytest.approx(n / 50) and design["det_items"] == min(20, n)
    assert design["skipped_components"] == [] and "item_rows" not in design["pilot"]
    assert "donors" not in design["pilot"] and "reuse" not in design["pilot"]
    red = reduced_components(HYPERPARAMETERS, design, [0], True)
    assert {r["component"] for r in red} >= {"items_per_seed", "seeds", "scan", "calibration_pilot", "donor_pool"}


def test_smoke_pilot_is_the_seed_scan(monkeypatch):
    """Smoke: the donor pool is built once (pilot) over the capped scan, and the seed reuses the pilot's
    baselines and its eligible item's arm rows instead of recomputing them."""
    calls = []
    original = EntitySwapFoilProbe.build_donor_pool

    def counting(pool_items, ner, entity_types):
        calls.append(len(pool_items))
        return original(pool_items, ner, entity_types)

    monkeypatch.setattr(EntitySwapFoilProbe, "build_donor_pool", staticmethod(counting))
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    cfg = _pilot_cfg()
    design = finalize_design(pilot_and_design(FakeHarness(), deps, toy_datasets(), cfg, [0], [], True, 1e9,
                                              time.time(), "float32"), HYPERPARAMETERS, [0], True)
    eligible = design["pilot"]["eligible_item"]
    _conds, extra = run_seed(0, design, deps, toy_datasets(), cfg, FakeHarness())
    pool = pool_for_seed(0, HYPERPARAMETERS["seeds"], toy_datasets())
    assert calls == [len(donor_items(pool, 0, HYPERPARAMETERS["smoke_scan_cap"]))]
    assert extra["donor_source"].startswith("pilot")
    assert extra["pilot_reused"]["rows"] == [eligible] and eligible in extra["included_ids"]
    assert extra["pilot_reused"]["baselines"]
    assert "pilot_cache" not in extra["design"]
    assert {r["sample_id"] for r in extra["item_rows"]} >= {eligible}


def test_donor_pool_is_capped_to_the_scan(monkeypatch):
    calls = []
    original = EntitySwapFoilProbe.build_donor_pool

    def counting(pool_items, ner, entity_types):
        calls.append([str(it["sample_id"]) for it in pool_items])
        return original(pool_items, ner, entity_types)

    monkeypatch.setattr(EntitySwapFoilProbe, "build_donor_pool", staticmethod(counting))
    pool = toy_pool()
    build_seed_donors(pool, 0, fake_ner, PLAN_ENTITY_TYPES, 3)
    assert calls == [[str(s) for s in list(scan_order(pool, 0))[:3]]]
    build_seed_donors(pool, 0, fake_ner, PLAN_ENTITY_TYPES, None)
    assert len(calls[1]) == len(pool)


def test_pilot_and_design_budget_insufficient_exits_1(capsys):
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    h = FakeHarness()
    with pytest.raises(SystemExit) as ei:
        pilot_and_design(h, deps, toy_datasets(), _pilot_cfg(), [0], [], True, 0.0, time.time(), "float32")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "TIME_ESTIMATE:" in out and "BUDGET_INSUFFICIENT" in out
    assert getattr(h, "written", None) is None and h.seed_records() == []


def test_resumed_design_skips_pilot(capsys):
    """harness.resumed_design is set on the instance (FakeHarness.__init__ resets it to None, so a class
    attribute would be shadowed); the pilot generator raises if called, so any pilot run fails the test."""
    def boom(question, docs):
        raise RuntimeError("pilot must not run")

    harness = FakeHarness()
    harness.resumed_design = {"n_items": 30, "quota": quota_for(30), "est_sec": 123.0}
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()), generate=boom)
    design = pilot_and_design(harness, deps, toy_datasets(), _pilot_cfg(), [0], [], False, 1e9, time.time(),
                              "float32")
    assert design["n_items"] == 30 and design["est_sec"] == 123.0
    out = capsys.readouterr().out
    assert "resumed design" in out and "pilot must not run" not in out


# ---- plan item 50: the design block recorded with every seed -------------------------------------
def test_recorded_design_names_primary_baseline_and_reductions():
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    cfg = _pilot_cfg()
    piloted = pilot_and_design(FakeHarness(), deps, toy_datasets(), cfg, [0], [], True, 1e9, time.time(),
                               "float32")
    legacy = {k: v for k, v in piloted.items() if k not in ("primary_condition", "baseline_condition")}
    design = finalize_design(legacy, HYPERPARAMETERS, [0], True)
    assert design["primary_condition"] == TOPICAL_END and design["baseline_condition"] == SELF_END
    assert design["reduced_components"] == reduced_components(HYPERPARAMETERS, legacy, [0], True)
    assert design["skipped_components"] == []
    h = FakeHarness()
    run_info = _run_info(time.time())
    run_all_seeds([0], design, deps, toy_datasets(), cfg, h, run_info, 1e12)
    recs = h.seed_records()
    assert [int(r["seed"]) for r in recs] == [0]
    extra = recs[0]["extra"]
    assert extra["design"]["primary_condition"] == TOPICAL_END
    assert extra["design"]["baseline_condition"] == SELF_END
    assert extra["reduced_components"] == design["reduced_components"]
    res = aggregate_and_write(h, cfg, {"t_start": 0.0})
    for r in design["reduced_components"]:
        assert r in res["reduced_components"]


def test_finalize_design_keeps_resumed_reductions_and_rejects_other_arms():
    kept = [{"component": "items_per_seed", "planned": 50, "used": 30, "reason": "budget"}]
    d = finalize_design({"n_items": 30, "reduced_components": kept}, HYPERPARAMETERS, [0], False)
    assert d["reduced_components"] == kept and d["primary_condition"] == TOPICAL_END
    with pytest.raises(ValueError):
        finalize_design({"primary_condition": SELF_END, "reduced_components": []}, HYPERPARAMETERS, [0], False)