import math

import pytest

import main
from main import CONDITION_NAMES, HYPERPARAMETERS, aggregate_and_write, build_config
from metrics import TOPICAL_END
from testing_fakes import FakeHarness


def _conds(v):
    return {c: {"itt_citation_migration_rate": v, "target_cited_rate": 0.1} for c in CONDITION_NAMES}


def _fake_validity(conds, hp, *args, **kwargs):
    return {"valid": conds[TOPICAL_END]["itt_citation_migration_rate"] < 0.5}


def _printed_mean(out, cond, key):
    prefix = f"condition={cond} {key}_mean: "
    for line in out.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):].split()[0])
    return None


def test_headline_uses_only_valid_seeds(monkeypatch, capsys):
    monkeypatch.setattr(main, "seed_validity", _fake_validity)
    recs = [{"seed": s, "conditions": _conds(v), "extra": {"design": {}, "item_rows": []}}
            for s, v in ((0, 0.1), (1, 0.9), (2, 0.2))]
    h = FakeHarness(recs)
    res = aggregate_and_write(h, build_config(HYPERPARAMETERS), {"t_start": 0.0})
    out = capsys.readouterr().out
    # primary_metric and aggregates: every record (same basis as the harness PRIMARY line)
    assert res["primary_metric"] == pytest.approx(0.4)
    assert res["aggregates"][TOPICAL_END]["itt_citation_migration_rate"]["mean"] == pytest.approx(0.4)
    assert _printed_mean(out, TOPICAL_END, "itt_citation_migration_rate") == pytest.approx(0.4)
    # headline: valid seeds only, under distinct keys
    assert res["headline_seeds"] == ["0", "2"] and res["recorded_seeds"] == ["0", "1", "2"]
    assert res["primary_metric_headline_seeds"] == pytest.approx(0.15)
    assert res["aggregates_headline_seeds"][TOPICAL_END]["itt_citation_migration_rate"]["mean"] == \
        pytest.approx(0.15)
    assert "primary_metric_all_seeds" not in res
    assert "seed_excluded_from_headline:1:seed_validity failed" in res["flags"]
    assert "primary_metric_includes_seeds_excluded_from_headline" in res["flags"]
    assert "PRIMARY" not in out
    assert all(math.isfinite(float(v)) for v in (res["primary_metric"], res["primary_metric_headline_seeds"]))