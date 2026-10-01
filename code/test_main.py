import pytest

from main import (CONDITION_NAMES, HYPERPARAMETERS, aggregate_and_write, build_config, choose_n_items,
                  control_checks, format_precondition, item_overlap, parse_rc_seeds, protocol_deviations,
                  record_seed_payload, run_seed, should_start_seed, smoke_config)
from metrics import NULL_ARM, POSITIVE_ARM, SELF_END
from probes import EntitySwapFoilProbe
from testing_fakes import FakeHarness, answer_table, make_fake_deps, toy_datasets, toy_pool


def test_hyperparameters_used_by_config_builder():
    cfg = build_config(HYPERPARAMETERS)
    assert set(vars(cfg)) == set(HYPERPARAMETERS)
    with pytest.raises(KeyError):
        build_config({k: v for k, v in HYPERPARAMETERS.items() if k != "alpha"})
    with pytest.raises(KeyError):
        build_config(dict(HYPERPARAMETERS, unused_key=1))


def test_hyperparameters_match_preregistration():
    assert protocol_deviations(HYPERPARAMETERS) == []
    devs = protocol_deviations(dict(HYPERPARAMETERS, effect_threshold=0.05,
                                    entity_types=HYPERPARAMETERS["entity_types"] + ["FAC"]))
    assert {d["key"] for d in devs} == {"effect_threshold", "entity_types"}


def test_precondition_line_format():
    assert format_precondition("x", True) == "PRECONDITION x: pass"
    assert format_precondition("x", False, "why") == "PRECONDITION x: FAIL why"


def test_choose_n_items_budget_arithmetic():
    n, est, _ = choose_n_items(10, 5, 0.5, 2, 2, 100, 2000, 50, 20, 20)
    assert n == 50 and est == pytest.approx(1680)
    n, est, _ = choose_n_items(10, 5, 0.5, 2, 2, 100, 1000, 50, 20, 20)
    assert n == 27 and est == pytest.approx(990)
    assert choose_n_items(10, 5, 0.5, 2, 2, 100, 500, 50, 20, 20)[0] is None


def test_should_start_seed_rule():
    assert should_start_seed(100, 90, 1.1) and not should_start_seed(98, 90, 1.1)
    assert should_start_seed(1, None, 1.1)


def test_smoke_config_shrinks_but_keeps_components():
    cfg = build_config(HYPERPARAMETERS)
    s, f = smoke_config(cfg, True), smoke_config(cfg, False)
    assert s["components"] == f["components"] and s["n_max"] < f["n_max"] and s["scan_cap"] == 40
    assert f["n_max"] == 50


def test_rc_seeds_and_overlap_helpers():
    assert parse_rc_seeds("10, 11 12") == [10, 11, 12] and parse_rc_seeds(None) is None
    assert item_overlap({"0": ["a", "b"], "1": ["c"]}) == []
    assert item_overlap({"0": ["a", "b"], "1": ["b"]}) == ["b"]


def test_record_payload_with_fake_harness():
    h = FakeHarness()
    conds = {c: {"itt_citation_migration_rate": 0.1, "x": None} for c in CONDITION_NAMES}
    record_seed_payload(h, 0, conds, {"design": {}})
    assert set(h.records[0]["conditions"]) == set(CONDITION_NAMES)
    with pytest.raises(ValueError):
        record_seed_payload(h, 1, {c: {"v": float("nan")} for c in CONDITION_NAMES}, {})


def test_control_checks_flags():
    per = {0: {NULL_ARM: {"itt_citation_migration_rate": 0.06}, POSITIVE_ARM: {"target_cited_rate": 0.8}}}
    out = control_checks(per, HYPERPARAMETERS)
    assert out["broken_null"] and not out["broken_positive"] and out["per_seed"]["0"]["null_broken"]


def test_aggregate_excludes_seeds_without_item_rows_from_headline_only():
    """Records without item rows leave the headline set only. The aggregates stay means over every harness
    record (the harness PRIMARY-line basis), so SELF_END is in `aggregates` with mean 1/6 and absent from
    `aggregates_headline_seeds`. The missing rows are counted and flagged."""
    def conds(v):
        return {c: {"itt_citation_migration_rate": (v if c == SELF_END else 0.0)} for c in CONDITION_NAMES}
    h = FakeHarness([{"seed": 0, "conditions": conds(1 / 3), "extra": {"design": {}}},
                     {"seed": 1, "conditions": conds(0.0), "extra": {"design": {}}}])
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert h.written is res
    assert SELF_END in res["aggregates"]
    assert res["aggregates"][SELF_END]["itt_citation_migration_rate"]["mean"] == pytest.approx(1 / 6)
    assert res["baseline_metric"] == pytest.approx(1 / 6)
    assert SELF_END not in res["aggregates_headline_seeds"]
    assert res["baseline_metric_headline_seeds"] is None
    assert res["headline_seeds"] == [] and res["recorded_seeds"] == ["0", "1"]
    assert res["fallback_counters"]["item_rows_missing"] == 2 and "item_rows_missing" in res["flags"]
    assert all("item_rows missing" in e["reasons"] for e in res["excluded_seeds"])


def test_run_seed_with_fakes_pairs_same_ids():
    pool = toy_pool()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(pool))
    design = {"n_items": 2, "quota": {"single_source": 1, "redundant": 1}, "scan_cap": None, "det_items": 20}
    cfg = build_config(dict(HYPERPARAMETERS, seeds=[0]))  # one dev seed reads the whole toy pool
    conds, extra = run_seed(0, design, deps, toy_datasets(), cfg, FakeHarness())
    ids = {c: sorted(r["sample_id"] for r in extra["item_rows"] if r["condition"] == c) for c in CONDITION_NAMES}
    assert len({tuple(v) for v in ids.values()}) == 1 and len(ids[SELF_END]) == 2
    assert conds[SELF_END]["n_items"] == 2 and not extra["protocol_deviation"]
    assert extra["pool_role"] == "dev" and all(r["pool"] == "dev" for r in extra["item_rows"])
    assert EntitySwapFoilProbe.build_donor_pool(pool, deps.ner, ["GPE"])["GPE"]