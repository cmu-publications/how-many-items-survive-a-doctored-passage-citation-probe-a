import pytest

from analysis import (INFERENCE_TEST, PRIMARY_CONTRASTS, SENSITIVITY_TEST, holm, holm_with_sensitivity,
                      mcnemar_paired, seed_validity)
from main import CONDITION_NAMES, HYPERPARAMETERS, aggregate_and_write, build_config, control_checks
from metrics import BASELINE, NULL_ARM, POSITIVE_ARM, SELF_END, TOPICAL_END
from testing_fakes import FakeHarness


# ---- plan items 35, 41: exact McNemar p is the inference p; clustered sign test is sensitivity ----
def _mrow(cond, seed, sid, move):
    return {"condition": cond, "seed": seed, "sample_id": sid, "move": move}


def test_mcnemar_p_is_inference_one_item_three_seeds():
    """One item under three seeds moves only in arm x: McNemar sees b=3, c=0 and p = 2 * 0.5**3 = 0.25, which
    is p_inference (plan item 41). The sample_id-clustered sign test (one item) gives 1.0 and is reported only
    as the sensitivity check."""
    xs = [_mrow(SELF_END, s, "a", True) for s in (0, 1, 2)]
    ys = [_mrow(TOPICAL_END, s, "a", False) for s in (2, 0, 1)]
    mc = mcnemar_paired(xs, ys)
    assert (mc["b"], mc["c"], mc["n"]) == (3, 0, 3)
    assert mc["p"] == pytest.approx(0.25) and mc["p_row_mcnemar"] == pytest.approx(0.25)
    assert mc["p_inference"] == mc["p_row_mcnemar"]
    assert mc["inference_test"] == INFERENCE_TEST == "mcnemar_exact"
    assert mc["p_item_sign_test"] == pytest.approx(1.0)
    assert mc["sensitivity_test"] == SENSITIVITY_TEST == "item_sign_test"
    assert mc["n_items"] == 1 and mc["n_repeated_items"] == 1
    assert mc["rd"] == pytest.approx(1.0) and mc["odds_ratio"] is None
    assert "McNemar" in mc["test"] and "clusters" in mc["test"]


def test_holm_input_is_the_mcnemar_p_and_sign_test_is_sensitivity(capsys):
    """Holm runs on p_inference = the McNemar p (plan item 41). For eight repeats of one item that p is
    2 * 0.5**8; the clustered sign test (1.0) only feeds p_holm_item_sign_test, and the disagreement is
    recorded in the caveat."""
    xs = [_mrow(SELF_END, s, "a", True) for s in range(8)]
    ys = [_mrow(TOPICAL_END, s, "a", False) for s in range(8)]
    mc = mcnemar_paired(xs, ys)
    assert mc["p_inference"] == mc["p_row_mcnemar"] == pytest.approx(2 * 0.5 ** 8)
    assert holm({"P1a": mc["p_inference"]})["P1a"] == pytest.approx(2 * 0.5 ** 8)
    assert mc["p_item_sign_test"] == pytest.approx(1.0)

    res = {k: dict(mc) for k in PRIMARY_CONTRASTS}
    caveat = holm_with_sensitivity(res, 0.05)
    for k in PRIMARY_CONTRASTS:
        assert res[k]["p_holm"] == pytest.approx(4 * 2 * 0.5 ** 8)
        assert res[k]["p_holm_item_sign_test"] == pytest.approx(1.0)
        assert res[k]["sig_under_item_sign_test"] is False
    assert "mcnemar" in res["holm_input"]
    assert caveat["any_repeated_items"] is True
    assert caveat["significance_differs_under_item_sign_test"] == list(PRIMARY_CONTRASTS)
    assert "FLAG" in capsys.readouterr().out

    distinct_x = [_mrow(SELF_END, 0, f"i{k}", True) for k in range(8)]
    distinct_y = [_mrow(TOPICAL_END, 0, f"i{k}", False) for k in range(8)]
    md = mcnemar_paired(distinct_x, distinct_y)
    assert md["p_inference"] == pytest.approx(md["p_row_mcnemar"]) and md["p_inference"] < 0.05
    assert md["p_item_sign_test"] == pytest.approx(md["p_row_mcnemar"]) and md["n_repeated_items"] == 0


def test_mcnemar_pairs_by_seed_and_id_not_position():
    xs = [_mrow(SELF_END, 0, "a", True), _mrow(SELF_END, 0, "b", False), _mrow(SELF_END, 1, "a", True),
          _mrow(SELF_END, 0, "c", True), _mrow(SELF_END, 5, "only_x", True)]
    ys = [_mrow(TOPICAL_END, 1, "a", True), _mrow(TOPICAL_END, 0, "c", False), _mrow(TOPICAL_END, 0, "b", True),
          _mrow(TOPICAL_END, 0, "a", False)]
    mc = mcnemar_paired(xs, ys)
    # (0,a): x moves, y not -> b; (0,b): y only -> c; (0,c): x only -> b; (1,a): both -> concordant
    assert (mc["b"], mc["c"], mc["n"]) == (2, 1, 4)
    assert mc["p"] == pytest.approx(1.0)
    assert (mc["b_items"], mc["c_items"]) == (2, 1) and mc["p_inference"] == pytest.approx(1.0)
    assert mc["rd"] == pytest.approx(0.25) and mc["odds_ratio"] == pytest.approx(2.0)
    tied = mcnemar_paired([_mrow(SELF_END, 0, "a", False)], [_mrow(TOPICAL_END, 0, "a", False)])
    assert tied["b"] + tied["c"] == 0 and tied["p"] == 1.0 and tied["odds_ratio"] is None
    assert tied["p_inference"] == 1.0


# ---- plan item 42: run-level controls; seeds are never selected on their control outcome ----------
def _valid_conds(null, pos=0.8):
    out = {c: {"itt_citation_migration_rate": 0.1, "target_cited_rate": 0.1, "citation_parse_rate": 1.0}
           for c in CONDITION_NAMES}
    out[BASELINE]["baseline_determinism_rate"] = 1.0
    out[NULL_ARM]["itt_citation_migration_rate"] = null
    out[POSITIVE_ARM]["target_cited_rate"] = pos
    return out


def _control_records(nulls, poss):
    return [{"seed": s, "conditions": _valid_conds(n, pos=p), "extra": {"design": {}, "item_rows": []}}
            for s, (n, p) in enumerate(zip(nulls, poss))]


def test_seed_validity_flags_but_keeps_seed_with_high_null():
    v = seed_validity(_valid_conds(0.12), HYPERPARAMETERS)
    assert v["valid"] is True and v["success"] is False
    assert v["reasons"] == [] and any("null_itt_above_broken" in f for f in v["flags"])
    assert not seed_validity(_valid_conds(0.12, pos=0.1), HYPERPARAMETERS)["valid"]


def test_broken_null_decided_over_every_recorded_seed():
    h = FakeHarness(_control_records((0.12, 0.12, 0.04, 0.04, 0.04), (0.8,) * 5))
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert res["headline_seeds"] == ["0", "1", "2", "3", "4"]
    assert res["control_checks_all_seeds"]["null_itt"] == pytest.approx(0.072)
    assert res["controls_for_verdicts"]["broken_null"] is True
    assert res["verdicts"][0]["label"] == "broken_null"
    sv = res["scientific_validity"]
    assert sv["controls_ok"] is False and sv["is_scientific_result"] is False
    assert sv["null_control_broken_run_level"] is True
    assert "null_control_broken_run_level" in res["flags"]


def test_broken_positive_decided_over_every_recorded_seed():
    """Seeds with a broken positive control fail seed validity and leave the headline set, yet the run-level
    positive control (mean over every recorded seed: 0.8/5 = 0.16 < 0.30) is broken and the verdicts say so.
    control_checks holds that run-level block; the headline-only block (seed 0 alone, 0.8) is descriptive
    and shows no broken control."""
    h = FakeHarness(_control_records((0.0,) * 5, (0.8, 0.0, 0.0, 0.0, 0.0)))
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert res["headline_seeds"] == ["0"]
    assert res["control_checks"]["broken_positive"] is True
    assert res["control_checks"] == res["controls_for_verdicts"]
    headline = res["control_checks_headline_descriptive"]
    assert headline["positive_broken_headline_descriptive"] is False
    assert headline["positive_target_cited"] == pytest.approx(0.8)
    assert "broken_positive" not in headline and headline["decides_verdicts"] is False
    assert res["control_checks_all_seeds"]["positive_target_cited"] == pytest.approx(0.16)
    assert res["controls_for_verdicts"]["broken_positive"] is True
    assert [v["label"] for v in res["verdicts"]] == ["broken_positive"]
    assert res["verdicts"][0]["evidence"]["positive_target_cited"] == pytest.approx(0.16)
    sv = res["scientific_validity"]
    assert sv["positive_control_broken_run_level"] is True and sv["controls_ok"] is False
    assert "positive_control_broken_run_level" in res["flags"]


def test_broken_controls_reported_when_every_seed_is_excluded():
    """All seeds broken on both controls: the verdicts are the broken labels, not no_valid_seeds."""
    h = FakeHarness(_control_records((0.2,) * 5, (0.0,) * 5))
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert res["headline_seeds"] == [] and res["contrasts"] is None
    assert [v["label"] for v in res["verdicts"]] == ["broken_null", "broken_positive"]
    assert res["scientific_validity"]["controls_ok"] is False


# ---- plan items 51, 52: aggregation and control checks -------------------------------------------
def _conds(itt, null, pos):
    out = {c: {"itt_citation_migration_rate": itt, "target_cited_rate": 0.1} for c in CONDITION_NAMES}
    out[NULL_ARM] = {"itt_citation_migration_rate": null, "target_cited_rate": 0.0}
    out[POSITIVE_ARM] = {"itt_citation_migration_rate": 0.0, "target_cited_rate": pos}
    return out


def _printed_mean(out, cond, key):
    prefix = f"condition={cond} {key}_mean: "
    for line in out.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):].split()[0])
    return None


def test_control_checks_expected_and_broken_thresholds():
    cc = control_checks({0: _conds(0.1, 0.04, 0.4)}, HYPERPARAMETERS)
    s = cc["per_seed"]["0"]
    assert s["null_expected_met"] is False and s["null_broken"] is False
    assert s["positive_expected_met"] is False and s["positive_broken"] is False
    cc = control_checks({0: _conds(0.1, 0.06, 0.2), 1: _conds(0.1, 0.0, 0.2)}, HYPERPARAMETERS)
    assert cc["per_seed"]["0"]["null_broken"] and not cc["per_seed"]["1"]["null_broken"]
    assert cc["null_itt"] == pytest.approx(0.03) and cc["null_expected_met"] is True
    assert cc["positive_broken"] and cc["broken_positive"]
    empty = control_checks({}, HYPERPARAMETERS)
    assert empty["null_expected_met"] is None and empty["null_broken"] is False


def test_aggregate_prints_per_seed_and_mean_lines(capsys):
    """Records without item rows are not screened by seed validity, so the per-seed lines and the mean over
    both harness records are printed and the primary metric is that mean; no PRIMARY line is printed. With no
    item rows anywhere, the contrasts, GLMM and deviance drop are listed in skipped_components and flagged."""
    recs = [{"seed": s, "conditions": _conds(v, 0.0, 0.8), "extra": {"design": {}}}
            for s, v in ((0, 0.2), (1, 0.4))]
    h = FakeHarness(recs)
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS),
                              {"t_start": 0.0, "unrun_seeds": [{"seed": 2, "reason": "budget exhausted"}],
                               "smoke": True})
    out = capsys.readouterr().out
    assert f"condition={TOPICAL_END} seed=0 itt_citation_migration_rate: 0.2" in out
    assert f"condition={TOPICAL_END} seed=1 itt_citation_migration_rate: 0.4" in out
    assert _printed_mean(out, TOPICAL_END, "itt_citation_migration_rate") == pytest.approx(0.3)
    assert "PRIMARY" not in out and "primary_metric:" not in out
    assert res["primary_metric"] == pytest.approx(0.3) and res["smoke"] is True
    for key in ("control_checks", "flags", "backend", "scientific_validity", "reduced_components",
                "control_arm_mapping", "novelty_precondition", "deviations", "fallback_counters",
                "excluded_selections", "hyperparameters", "unrun_seeds", "condition_names"):
        assert key in res, key
    assert set(res["skipped_components"]) == {"contrasts", "glmm", "deviance_drop"}
    assert "contrasts_glmm_deviance_skipped_no_item_rows" in res["flags"]
    assert res["unrun_seeds"][0]["seed"] == 2
    assert res["control_arm_mapping"]["null"].startswith("null_random_drop")
    assert res["control_arm_mapping"]["positive_control"].startswith("positive_control_planted")


def test_aggregate_empty_item_rows_excludes_seeds_from_headline(capsys):
    """Seeds that fail seed validity leave the headline set, but they stay in the record basis: the per-seed
    lines are printed, primary_metric is the mean over every harness record (0.3, the same basis as the
    harness PRIMARY line), and only the *_headline_seeds keys are None; the run flags that the primary
    metric includes seeds excluded from the headline set."""
    recs = [{"seed": s, "conditions": _conds(v, 0.0, 0.8), "extra": {"design": {}, "item_rows": []}}
            for s, v in ((0, 0.2), (1, 0.4))]
    h = FakeHarness(recs)
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    out = capsys.readouterr().out
    assert f"condition={TOPICAL_END} seed=0 itt_citation_migration_rate: 0.2" in out
    assert f"condition={TOPICAL_END} seed=1 itt_citation_migration_rate: 0.4" in out
    assert res["primary_metric"] == pytest.approx(0.3)
    assert res["primary_metric_headline_seeds"] is None
    assert res["headline_seeds"] == [] and res["recorded_seeds"] == ["0", "1"]
    assert {e["seed"] for e in res["excluded_seeds"]} == {"0", "1"}
    assert "primary_metric_includes_seeds_excluded_from_headline" in res["flags"]
    assert "PRIMARY" not in out
    assert h.written is res


def test_aggregate_mean_not_last_value(capsys):
    """Plan item 51: printed means and aggregates are means over every harness record (two seeds, no
    item rows, so no seed reaches the headline set), not the last seed's value; no PRIMARY line."""
    recs = [{"seed": s, "conditions": _conds(v, 0.0, 0.8), "extra": {"design": {}}}
            for s, v in ((0, 0.1), (1, 0.5))]
    h = FakeHarness(recs)
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    out = capsys.readouterr().out
    assert _printed_mean(out, TOPICAL_END, "itt_citation_migration_rate") == pytest.approx(0.3)
    assert res["aggregates"][TOPICAL_END]["itt_citation_migration_rate"]["mean"] == pytest.approx(0.3)
    assert res["primary_metric"] == pytest.approx(0.3)
    assert res["primary_metric_headline_seeds"] is None and res["headline_seeds"] == []
    assert res["fallback_counters"]["item_rows_missing"] == 2 and "item_rows_missing" in res["flags"]
    assert f"condition={TOPICAL_END} seed=1 itt_citation_migration_rate: 0.5" in out
    assert "PRIMARY" not in out and "primary_metric:" not in out
    assert h.written is res


def test_aggregate_writes_only_through_harness(tmp_path, monkeypatch):
    """No results file is written by the code itself; the harness's write_results is the only writer."""
    monkeypatch.chdir(tmp_path)
    recs = [{"seed": 0, "conditions": _conds(0.2, 0.0, 0.8), "extra": {"design": {}, "item_rows": []}}]
    h = FakeHarness(recs)
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert h.written is res
    assert not (tmp_path / "results.json").exists() and not (tmp_path / "seed_records.json").exists()


def test_aggregate_zero_seeds_runs_cleanly(capsys):
    h = FakeHarness()
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    assert res["primary_metric"] is None and "no_seed_recorded" in res["flags"]
    assert res["scientific_validity"]["n_seeds_recorded"] == 0
    assert "PRIMARY" not in capsys.readouterr().out