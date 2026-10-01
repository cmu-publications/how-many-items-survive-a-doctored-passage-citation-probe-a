"""Results contracts for aggregate_and_write: one broken-control value per decision, and per-pool
contrasts that never add to the run-level fallback counters."""
import time
from typing import Any, Dict, List

import main
from main import CONDITION_NAMES, HYPERPARAMETERS, aggregate_and_write, build_config
from metrics import NULL_ARM, POSITIVE_ARM
from testing_fakes import FakeHarness


def _conds(pos: float) -> Dict[str, Dict[str, Any]]:
    out = {c: {"itt_citation_migration_rate": 0.01, "target_cited_rate": 0.8} for c in CONDITION_NAMES}
    out[NULL_ARM] = {"itt_citation_migration_rate": 0.0, "target_cited_rate": 0.0}
    out[POSITIVE_ARM] = {"itt_citation_migration_rate": 0.9, "target_cited_rate": pos}
    return out


def _record(seed: int, pos: float, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"seed": seed, "conditions": _conds(pos),
            "extra": {"item_rows": rows, "included_ids": [r["sample_id"] for r in rows]}}


def _validity_by_positive(conds, hp, *args, **kwargs):
    pos = conds[POSITIVE_ARM]["target_cited_rate"]
    return {"valid": bool(pos >= float(hp["pos_broken_below"]))}


def _always_valid(conds, hp, *args, **kwargs):
    return {"valid": True}


def _run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return aggregate_and_write(FakeHarness(records), build_config(HYPERPARAMETERS), {"t_start": time.time()})


def test_controls_and_control_checks_hold_the_run_level_verdict_values(monkeypatch):
    monkeypatch.setattr(main, "seed_validity", _validity_by_positive)
    monkeypatch.setattr(main, "compute_contrasts", lambda rows, hp, counters, offsets: None)
    records = [_record(s, pos, [{"sample_id": f"s{s}", "pool": "dev"}])
               for s, pos in enumerate((0.8, 0.0, 0.0, 0.0, 0.0))]
    res = _run(records)
    assert res["headline_seeds"] == ["0"]
    decided = res["controls_for_verdicts"]["broken_positive"]
    assert decided is True
    assert res["controls"]["broken_positive"] is decided
    assert res["control_checks"]["broken_positive"] is decided
    assert res["control_checks_all_seeds"]["broken_positive"] is decided
    assert res["scientific_validity"]["positive_control_broken_run_level"] is decided
    headline = res["control_checks_headline_descriptive"]
    for key in ("broken_null", "broken_positive", "null_broken", "positive_broken"):
        assert key not in headline
    assert headline["positive_broken_headline_descriptive"] is False
    assert headline["decides_verdicts"] is False


def test_pool_contrasts_do_not_add_to_run_level_counters(monkeypatch):
    monkeypatch.setattr(main, "seed_validity", _always_valid)

    def counting(rows, hp, counters, offsets):
        counters["bootstrap_undefined_resamples"] += len(rows)
        counters["natural_bin_empty"] += 1
        return None

    monkeypatch.setattr(main, "compute_contrasts", counting)
    dev_rows = [{"sample_id": f"d{i}", "pool": "dev"} for i in range(3)]
    conf_rows = [{"sample_id": f"c{i}", "pool": "confirmation"} for i in range(2)]
    res = _run([_record(0, 0.8, dev_rows), _record(5, 0.8, conf_rows)])
    # One pass over the 5 headline rows; the per-pool passes must not be added again.
    assert res["fallback_counters"]["bootstrap_undefined_resamples"] == 5
    assert res["fallback_counters"]["natural_bin_empty"] == 1
    by_pool = res["fallback_counters_by_pool"]
    assert by_pool["dev"]["bootstrap_undefined_resamples"] == 3
    assert by_pool["confirmation"]["bootstrap_undefined_resamples"] == 2
    assert by_pool["dev"]["natural_bin_empty"] == 1
    assert by_pool["confirmation"]["natural_bin_empty"] == 1
    assert set(res["contrasts_by_pool"]) == {"dev", "confirmation"}