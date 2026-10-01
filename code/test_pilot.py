import pytest

import main  # noqa: F401  sets the CUDA / HF-offline environment before torch is imported
import pilot
from main import HYPERPARAMETERS
from pilot import _fraction_kept, measure_pilot, reduced_components
from testing_fakes import answer_table, make_fake_deps, toy_datasets, toy_pool


# ---- plan item 46: the pilot times the LOO passes on the eligible item ----------------------------
def test_pilot_times_loo_audit_on_the_eligible_item(monkeypatch):
    calls = []
    original = pilot.run_item

    def recording(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pilot, "run_item", recording)
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    res = measure_pilot(deps, toy_datasets(), HYPERPARAMETERS, 0, HYPERPARAMETERS["seeds"],
                        int(HYPERPARAMETERS["pilot_scan_items"]), int(HYPERPARAMETERS["smoke_scan_cap"]))
    assert res["sec_arms"] is not None and res["eligible_item"] is not None
    assert len(calls) == 1 and calls[0].get("force_audit") is True
    assert res["loo_forced"] is True and "LOO" in res["stages"]


# ---- reduced components carry the fraction of items kept ------------------------------------------
def test_reduced_components_record_fraction_of_items_kept():
    red = reduced_components(HYPERPARAMETERS, {"n_items": 5, "det_items": 5, "scan_cap": 40}, [0], True)
    by = {r["component"]: r for r in red}
    assert by["items_per_seed"]["fraction_kept"] == pytest.approx(5 / 50)
    assert by["determinism_check"]["fraction_kept"] == pytest.approx(5 / 20)
    budget = reduced_components(HYPERPARAMETERS, {"n_items": 30, "det_items": 20}, [0, 1, 2], False)
    assert [r["component"] for r in budget] == ["items_per_seed"]
    assert budget[0]["fraction_kept"] == pytest.approx(0.6) and budget[0]["reason"] == "budget"
    assert reduced_components(HYPERPARAMETERS, {"n_items": 50, "det_items": 20}, [0], False) == []


def test_fraction_kept_rejects_nonpositive_plan():
    with pytest.raises(ValueError):
        _fraction_kept(3, 0)