"""Plan item 53 / No Optional Component contract: the contrasts, GLMM and deviance drop are never skipped
silently, and the positive-control donor deviation reaches the run-level deviations block."""
import pytest

from main import (CONDITION_NAMES, CONTRAST_COMPONENTS, CONTRASTS_BASIS_ALL, CONTRASTS_BASIS_HEADLINE,
                  CONTRASTS_BASIS_NONE, HYPERPARAMETERS, aggregate_and_write, build_config)
from metrics import BASELINE, NULL_ARM, POSITIVE_ARM
from probes import POSCONTROL_DONOR_POPULATION, probe_deviations
from testing_fakes import FakeHarness, self_check_rows


def _conds(pos):
    out = {c: {"itt_citation_migration_rate": 0.1, "target_cited_rate": 0.1, "citation_parse_rate": 1.0}
           for c in CONDITION_NAMES}
    out[BASELINE]["baseline_determinism_rate"] = 1.0
    out[NULL_ARM]["itt_citation_migration_rate"] = 0.0
    out[POSITIVE_ARM]["target_cited_rate"] = pos
    return out


def _rows():
    return [dict(r) for rows in self_check_rows().values() for r in rows]


def _cfg():
    return build_config(dict(HYPERPARAMETERS, bootstrap_resamples=20))


def test_smoke_seed_failing_validity_still_runs_contrasts_glmm_and_deviance():
    """A smoke seed that fails seed_validity (broken positive control) on its few items still gets the
    contrasts, GLMM and deviance drop, over its item rows; nothing is skipped, and the reduced basis is
    recorded and flagged while the verdicts do not use it."""
    rec = {"seed": 0, "conditions": _conds(0.0), "extra": {"design": {}, "item_rows": _rows()}}
    res = aggregate_and_write(FakeHarness([rec]), _cfg(), {"t_start": 0.0, "smoke": True})
    assert res["headline_seeds"] == [] and res["contrasts"] is None
    assert res["contrasts_basis"] == CONTRASTS_BASIS_ALL
    assert isinstance(res["contrasts_descriptive_all_recorded_seeds"], dict)
    assert res["contrasts_descriptive_all_recorded_seeds"]
    assert res["skipped_components"] == []
    reduced = {r["component"] for r in res["reduced_components"] if isinstance(r, dict)}
    assert set(CONTRAST_COMPONENTS) <= reduced
    assert "contrasts_computed_on_seeds_excluded_from_headline" in res["flags"]
    assert res["scientific_validity"]["contrasts_basis"] == CONTRASTS_BASIS_ALL


def test_valid_seed_uses_headline_contrasts_only():
    rec = {"seed": 0, "conditions": _conds(0.8), "extra": {"design": {}, "item_rows": _rows()}}
    res = aggregate_and_write(FakeHarness([rec]), _cfg(), {"t_start": 0.0})
    assert res["headline_seeds"] == ["0"] and isinstance(res["contrasts"], dict)
    assert res["contrasts_basis"] == CONTRASTS_BASIS_HEADLINE
    assert res["contrasts_descriptive_all_recorded_seeds"] is None
    assert res["skipped_components"] == []


def test_no_item_rows_lists_the_analysis_as_skipped_and_flags_it(capsys):
    rec = {"seed": 0, "conditions": _conds(0.8), "extra": {"design": {}, "item_rows": []}}
    res = aggregate_and_write(FakeHarness([rec]), _cfg(), {"t_start": 0.0})
    assert res["contrasts"] is None and res["contrasts_basis"] == CONTRASTS_BASIS_NONE
    assert set(CONTRAST_COMPONENTS) <= set(res["skipped_components"])
    assert "contrasts_glmm_deviance_skipped_no_item_rows" in res["flags"]
    assert "FLAG: contrasts_glmm_deviance_skipped_no_item_rows" in capsys.readouterr().out


def test_positive_control_donor_deviation_is_in_run_level_deviations():
    rec = {"seed": 0, "conditions": _conds(0.8), "extra": {"design": {}, "item_rows": []}}
    res = aggregate_and_write(FakeHarness([rec]), _cfg(), {"t_start": 0.0})
    assert res["deviations"]["probes"] == probe_deviations()
    assert res["protocol_deviations"]["probes"][0]["used"] == POSCONTROL_DONOR_POPULATION
    assert "protocol_deviation:positive_control_donor_population" in res["flags"]


def test_empty_run_still_reports_probe_deviation():
    res = aggregate_and_write(FakeHarness(), _cfg(), {"t_start": 0.0})
    assert res["deviations"]["probes"] and "protocol_deviation:positive_control_donor_population" in res["flags"]
    assert res["primary_metric"] is None
    assert pytest.approx(0.0) == 0.0