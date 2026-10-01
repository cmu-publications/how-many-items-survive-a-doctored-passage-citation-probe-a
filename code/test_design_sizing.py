"""The item count per seed never depends on which seeds a run executes (Seed Sets contract)."""
import time
from typing import Any, Dict, List

import pytest

import design as design_module
from design import sizing_basis
from main import HYPERPARAMETERS, build_config
from testing_fakes import FakeHarness

DEV_SEEDS = [0, 1, 2, 3, 4]
CONFIRMATION_SEEDS = [5, 6, 7]
# overhead 600 + 5 seeds x (20 det x 1s + n x 20s) fits n=30 at 3750s; 3 seeds would fit n=50.
BUDGET = 3750.0


def _cfg():
    return build_config(dict(HYPERPARAMETERS))


def _patch_pilot(monkeypatch, seen: List[int]) -> None:
    def fake_measure(deps, datasets, hp, pilot_seed, plan_seeds, pilot_items, scan_cap):
        seen.append(int(pilot_seed))
        return {"donors": {}, "reuse": {"counters": {}}, "sec_arms": 10.0, "sec_scan_mean": 3.0,
                "sec_gen_mean": 1.0, "n_scanned": 3, "eligible_item": "q1", "eligible_rate_measured": 0.3,
                "item_rows": [], "alias_offsets": []}

    def fake_agg(item_rows, n_items, n_seeds, hp, alias_offsets) -> Dict[str, Any]:
        return {"seconds": 0.0, "n_rows": 0, "errors": []}

    monkeypatch.setattr(design_module, "measure_pilot", fake_measure)
    monkeypatch.setattr(design_module, "time_aggregation", fake_agg)


def _design(run_seeds, confirmation, budget=BUDGET):
    return design_module.pilot_and_design(FakeHarness(), None, {}, _cfg(), run_seeds, confirmation, False,
                                          budget, time.time(), "float32")


def test_sizing_basis_uses_plan_dev_seed_count_for_any_run_seed_set():
    hp = dict(HYPERPARAMETERS)
    assert sizing_basis(hp, DEV_SEEDS, False) == (0, len(hp["seeds"]))
    assert sizing_basis(hp, CONFIRMATION_SEEDS, False) == (5, len(hp["seeds"]))
    assert sizing_basis(hp, [9], True) == (9, 1)
    with pytest.raises(ValueError):
        sizing_basis(hp, [], False)


def test_confirmation_run_chooses_the_same_n_as_the_dev_run(monkeypatch, capsys):
    seen: List[int] = []
    _patch_pilot(monkeypatch, seen)
    dev = _design(DEV_SEEDS, CONFIRMATION_SEEDS)
    conf = _design(CONFIRMATION_SEEDS, CONFIRMATION_SEEDS)
    assert seen == [0, 5]
    assert dev["n_items"] == 30
    assert conf["n_items"] == dev["n_items"]
    assert conf["quota"] == dev["quota"]
    assert conf["det_items"] == dev["det_items"]
    assert conf["sizing"]["n_seeds_for_sizing"] == dev["sizing"]["n_seeds_for_sizing"] == 5
    assert conf["sizing"]["n_seeds_this_run"] == 3
    assert conf["est_sec"] < dev["est_sec"]
    assert "TIME_ESTIMATE" in capsys.readouterr().out


def test_run_with_more_seeds_than_the_plan_refuses_instead_of_shrinking_n(monkeypatch, capsys):
    _patch_pilot(monkeypatch, [])
    with pytest.raises(SystemExit) as exc:
        _design(DEV_SEEDS + CONFIRMATION_SEEDS, CONFIRMATION_SEEDS)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "BUDGET_INSUFFICIENT" in out
    assert "never" not in out or "refitted" in out