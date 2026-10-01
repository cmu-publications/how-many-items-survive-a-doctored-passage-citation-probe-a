import time
from collections import Counter
from typing import Any, Dict

import pytest

import main as main_module
import seed_loop
from analysis import cluster_bootstrap
from baselines import citation_positions, count_truncation, mean_nll_or_none
from data import AsqaAlceDataset, pool_for_seed
from main import (HYPERPARAMETERS, PLAN_ENTITY_TYPES, SeedFailed, SeedStopped, aggregate_and_write, build_config,
                  pool_sufficiency, protocol_deviations, quota_for, run_all_seeds, run_preconditions, run_seed)
from metrics import natural_location_rate
from preconditions import NLI_SANITY_PAIRS, nli_sanity_correct, precondition_checks
from probes import EntitySwapFoilProbe, delta_logp_target, reliance_audit
from testing_fakes import (FakeHarness, answer_table, fake_ner, make_fake_deps, make_item, toy_datasets,
                           toy_pool)


def _char_offsets(text):
    """One token per character: every digit is its own token."""
    return [(i, i + 1) for i in range(len(text))]


def _run_info(t_start: float) -> Dict[str, Any]:
    return {"t_start": t_start, "unrun_seeds": [], "seed_failures": []}


def _design():
    return {"n_items": 10, "quota": {"single_source": 5, "redundant": 5}, "scan_cap": None, "det_items": 20,
            "est_sec": 100.0}


# ---- run_and_write never catches an exception escaping the seed loop -----------------------------
def test_run_and_write_reports_any_escaping_error_and_writes_nothing(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("loop fault")

    monkeypatch.setattr(main_module, "run_all_seeds", boom)
    h = FakeHarness()
    with pytest.raises(RuntimeError, match="loop fault"):
        main_module.run_and_write(h, {}, [0], None, {}, build_config(HYPERPARAMETERS), _run_info(0.0), 1e9)
    assert "ABORTED" in capsys.readouterr().out
    assert getattr(h, "written", None) is None


def test_run_and_write_lets_system_exit_through_unchanged(monkeypatch, capsys):
    raised = SystemExit(3)

    def leave(*args, **kwargs):
        raise raised

    monkeypatch.setattr(main_module, "run_all_seeds", leave)
    h = FakeHarness()
    with pytest.raises(SystemExit) as ei:
        main_module.run_and_write(h, {}, [0], None, {}, build_config(HYPERPARAMETERS), _run_info(0.0), 1e9)
    assert ei.value is raised and ei.value.code == 3
    assert "ABORTED" in capsys.readouterr().out
    assert getattr(h, "written", None) is None


def test_pool_for_seed_by_id():
    dev = [make_item(f"d{i}", "Paris") for i in range(6)]
    conf = [make_item(f"c{i}", "Berlin") for i in range(4)]
    ds = {"val": AsqaAlceDataset(dev, role="development_pool"),
          "test": AsqaAlceDataset(conf, role="confirmation_pool")}
    plan = [0, 1, 2]
    dev_ids = sorted(str(it["sample_id"]) for it in dev)
    conf_ids = sorted(str(it["sample_id"]) for it in conf)
    for s in plan:
        assert sorted(str(it["sample_id"]) for it in pool_for_seed(s, plan, ds)) == dev_ids
        assert sorted(str(it["sample_id"]) for it in pool_for_seed(s, [s], ds)) == dev_ids
    for s in (7, 99):
        assert sorted(str(it["sample_id"]) for it in pool_for_seed(s, plan, ds)) == conf_ids
    assert pool_for_seed(0, [2, 1, 0], ds) == pool_for_seed(0, [0], ds)
    assert not set(dev_ids) & set(conf_ids)


def test_pool_sufficiency_arithmetic():
    quota = quota_for(50)
    assert quota == {"single_source": 25, "redundant": 25}
    ok, lines = pool_sufficiency({0: 569}, {0: {"single_source": 200, "redundant": 200}}, 50, quota, 0.3)
    assert ok and any("required=ceil(50/0.3)=167" in ln for ln in lines)
    ok, lines = pool_sufficiency({0: 114}, {0: {"single_source": 50, "redundant": 40}}, 50, quota, 0.3)
    assert not ok and any(ln.startswith("POOL_INSUFFICIENT") for ln in lines)
    ok, lines = pool_sufficiency({0: 400}, {0: {"single_source": 300, "redundant": 10}}, 50, quota, 0.3)
    assert not ok and any("stratum redundant" in ln for ln in lines)
    ok, _ = pool_sufficiency({0: 400}, {0: {"single_source": 200, "redundant": 200}}, 50, quota, 0.0)
    assert not ok


def test_fallback_counters_increment():
    c: Counter = Counter()
    delta_logp_target([], [], 2, c)
    citation_positions("A [2].", 0, [(0, 1), (1, 3), (3, 5), (5, 6)], c)
    citation_positions("A [2, 9].", 0, _char_offsets("A [2, 9]."), c)
    reliance_audit({"move": True, "edited_answer": "x"}, lambda q, d, a: -1.0, [{"title": "", "text": "a"}],
                   [{"title": "", "text": "a"}], 1, question="q", hp=HYPERPARAMETERS, counters=c)
    natural_location_rate([], [1.0, 0, 0, 0, 0], c)
    count_truncation(513, 512, c)
    cluster_bootstrap([{"sample_id": "a", "condition": "x"}], lambda r: None, B=3, counters=c)
    EntitySwapFoilProbe(HYPERPARAMETERS).swap("Paris is it.", ["paris"], {"GPE": [("q2", "Berlin")]}, 0, "q1",
                                              lambda p, h: {"contradiction": 0.0}, fake_ner, c)
    mean_nll_or_none([], c)
    for key in ("slot_mismatch", "digit_merged_slot_skipped", "citation_out_of_range_slot_skipped",
                "loo_base_nonpositive", "natural_bin_empty", "nli_truncated", "bootstrap_undefined_resamples",
                "swap_nli_rejected", "ppl_too_short"):
        assert c[key] >= 1, key
    h = FakeHarness([{"seed": 0, "conditions": {}, "extra": {"design": {}}}])
    aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert h.written["fallback_counters"]["item_rows_missing"] == 1
    pool = toy_pool()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(pool))
    design = {"n_items": 10, "quota": {"single_source": 5, "redundant": 5}, "scan_cap": None, "det_items": 20}
    cfg = build_config(HYPERPARAMETERS)
    _c, extra = run_seed(0, design, deps, toy_datasets(), cfg, FakeHarness())
    assert extra["counters"]["pool_exhausted"] == 1 and extra["protocol_deviation"] and extra["shortfall"] == 6


# ---- plan items 17, 44: hyperparameter contract --------------------------------------------------
def test_hyperparameters_match_plan_values():
    hp = HYPERPARAMETERS
    assert hp["swap_max_tries"] == 10 and hp["n_min_items"] == 20 and hp["n_max_items"] == 50
    assert hp["smoke_n_items"] == 5 and hp["smoke_scan_cap"] == 40 and hp["pilot_scan_items"] == 10
    assert hp["smoke_min_items"] == 2
    assert hp["det_items"] == 20 and hp["max_new_tokens"] == 300
    assert hp["decoding"] == "greedy" and hp["batch_size"] == 1
    assert hp["reader_model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert hp["nli_model_id"] == "cross-encoder/nli-deberta-v3-base"
    assert sorted(hp["entity_types"]) == sorted(PLAN_ENTITY_TYPES) and len(hp["entity_types"]) == 9
    assert protocol_deviations(hp) == []
    assert protocol_deviations(dict(hp, swap_max_tries=5))[0]["key"] == "swap_max_tries"


class _RecordingHP(dict):
    """A hyperparameter dict that records every key read through []."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def test_build_config_consumes_every_hyperparameter():
    rec = _RecordingHP(HYPERPARAMETERS)
    cfg = build_config(rec)
    assert rec.read == set(HYPERPARAMETERS)
    assert vars(cfg) == HYPERPARAMETERS
    assert "tf_forwards_per_item" not in HYPERPARAMETERS


def test_build_config_rejects_invalid_or_unknown_values():
    with pytest.raises(ValueError):
        build_config(dict(HYPERPARAMETERS, n_min_items=0))
    with pytest.raises(ValueError):
        build_config(dict(HYPERPARAMETERS, n_min_items=60))
    with pytest.raises(ValueError):
        build_config(dict(HYPERPARAMETERS, decoding="sampling"))
    with pytest.raises(KeyError):
        build_config(dict(HYPERPARAMETERS, tf_forwards_per_item=32))


def test_hyperparameters_are_passed_into_seed_extra():
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    _c, extra = run_seed(0, _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), FakeHarness())
    assert extra["hyperparameters"] == HYPERPARAMETERS


# ---- plan item 45: preconditions ----------------------------------------------------------------
def test_precondition_checks_pass_and_fail():
    data = {"n_items": 948, "n_demos": 2, "n_dev": 500, "n_conf": 448, "manifests": ["m.json"]}
    digits = {d: 1 for d in range(1, 6)}
    checks = precondition_checks(data, (True, "ok"), digits, 8, PLAN_ENTITY_TYPES, HYPERPARAMETERS, {})
    assert [c[0] for c in checks] == ["data_and_models", "reader_numerics", "digit_tokens", "nli_sanity",
                                      "spacy_ner"]
    assert all(ok for _n, ok, _d in checks)
    assert precondition_checks(data, (True, "ok"), digits, 7, PLAN_ENTITY_TYPES, HYPERPARAMETERS, {})[3][1]
    bad = precondition_checks(dict(data, manifests=[], n_items=947), (False, "nan"), {**digits, 3: 2}, 6,
                              ["PERSON"], HYPERPARAMETERS, {})
    assert not any(ok for _n, ok, _d in bad)
    err = precondition_checks(data, (True, "ok"), digits, 8, PLAN_ENTITY_TYPES, HYPERPARAMETERS,
                              {"nli_sanity": "RuntimeError: boom"})
    assert err[3] == ("nli_sanity", False, "RuntimeError: boom")


def test_nli_sanity_pairs_oracle():
    assert len(NLI_SANITY_PAIRS) == 8
    table = {(p, h): lab for p, h, lab in NLI_SANITY_PAIRS}

    def oracle(p, h):
        return {lab: (0.9 if lab == table[(p, h)] else 0.05) for lab in ("entailment", "neutral", "contradiction")}
    assert nli_sanity_correct(oracle) == 8


def test_run_preconditions_exits_1_on_fail(tmp_path, capsys):
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    with pytest.raises(SystemExit) as ei:
        run_preconditions(toy_datasets(), str(tmp_path), HYPERPARAMETERS, deps, lambda: [], "bfloat16")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "PRECONDITION data_and_models: FAIL" in out and "PRECONDITION spacy_ner: FAIL" in out
    assert "PRECONDITION_FAILED" in out


# ---- plan items 47, 49 and the partial-seed contract ---------------------------------------------
class _StopHarness(FakeHarness):
    def should_stop(self):
        return True


def test_budget_stop_mid_seed_is_not_recorded():
    h = _StopHarness()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    with pytest.raises(SeedStopped):
        run_seed(0, _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), h)
    run_info = _run_info(0.0)
    run_all_seeds([0, 1], _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), h, run_info, 1e12)
    assert h.seed_records() == []
    unrun = list(run_info["unrun_seeds"])
    assert [u["seed"] for u in unrun] == [0, 1]
    assert unrun[0]["reason"] == "budget_stop_mid_seed"
    red = list(run_info["reduced_components"])
    assert red[0]["component"] == "seeds" and red[0]["used"] == [] and "emergency" in red[0]["reason"]


def test_seeds_are_never_dropped_for_time(capsys):
    """Budget 0: the remaining time is below seed_start_safety_factor x the measured seed cost before seed 1.
    The seed loop only FLAGs that and records it under seed_start_warnings; both seeds still run and are
    recorded, no seed is listed unrun and no seeds reduction is recorded (only the hard stop may end the loop).
    This fails if the loop ever drops a seed for time again."""
    h = FakeHarness()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    run_info = _run_info(time.time())
    run_all_seeds([0, 1], _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), h, run_info, 0.0)
    assert sorted(int(r["seed"]) for r in h.seed_records()) == [0, 1]
    assert run_info["unrun_seeds"] == []
    assert not [r for r in run_info.get("reduced_components", []) if r["component"] == "seeds"]
    assert "time_warnings" not in run_info
    warnings = run_info["seed_start_warnings"]
    assert 1 in [w["seed"] for w in warnings]
    assert all(w["started"] is True and w["reason"] == seed_loop.SEED_START_WARNING for w in warnings)
    w1 = next(w for w in warnings if w["seed"] == 1)
    assert w1["measured_seed_sec"] is not None and w1["remaining_sec"] < w1["required_sec"]
    out = capsys.readouterr().out
    assert "starting it anyway" in out and "seed_start_warnings" in out


def test_run_seed_prints_progress_per_item(capsys):
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    run_seed(0, _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), FakeHarness())
    assert "seed 0 item 1 " in capsys.readouterr().out


def test_failing_seed_does_not_abort_run():
    def boom(question, docs):
        raise RuntimeError("gpu fault")
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()), generate=boom)
    cfg = build_config(dict(HYPERPARAMETERS, max_item_exceptions=1))
    with pytest.raises(SeedFailed):
        run_seed(0, _design(), deps, toy_datasets(), cfg, FakeHarness())
    h = FakeHarness()
    run_info = _run_info(0.0)
    run_all_seeds([0, 1], _design(), deps, toy_datasets(), cfg, h, run_info, 1e12)
    failures = list(run_info["seed_failures"])
    assert [f["seed"] for f in failures] == [0, 1]
    assert failures[0]["error_class"] == "SeedFailed"
    res = aggregate_and_write(h, cfg, run_info)
    assert h.written is res and len(res["seed_failures"]) == 2 and "seed_failed:0:SeedFailed" in res["flags"]


# ---- plan item 46: smoke mode and budget from the environment -------------------------------------
def test_smoke_and_budget_read_from_environment(monkeypatch):
    monkeypatch.setenv("RC_SMOKE_TEST", "1")
    monkeypatch.setenv("RC_TIME_BUDGET_SEC", "1500")
    assert main_module._env_flag("RC_SMOKE_TEST") is True
    assert main_module._budget_sec(HYPERPARAMETERS) == pytest.approx(1500.0)
    monkeypatch.delenv("RC_SMOKE_TEST")
    monkeypatch.delenv("RC_TIME_BUDGET_SEC")
    monkeypatch.delenv("TIME_BUDGET_SEC", raising=False)
    assert main_module._env_flag("RC_SMOKE_TEST") is False
    assert main_module._budget_sec(HYPERPARAMETERS) == pytest.approx(float(HYPERPARAMETERS["time_budget_sec"]))