import math

import pytest

from metrics import (BASELINE, CONDITION_ORDER, SELF_END, EmptySeedConditionError, new_counters,
                     seed_condition_metrics)
from testing_fakes import self_check_rows

SCAN = {"n_parse_ok": 1, "n_generated": 1}
DET = [("a", "a")]


def test_zero_items_seed_is_refused_not_recorded_with_none_primary():
    counters = new_counters()
    with pytest.raises(EmptySeedConditionError) as exc:
        seed_condition_metrics({c: [] for c in CONDITION_ORDER}, SCAN, DET, counters=counters)
    assert set(exc.value.empty_conditions) == set(CONDITION_ORDER)
    assert counters["item_rows_missing"] == 1


def test_missing_condition_keys_are_refused():
    with pytest.raises(EmptySeedConditionError):
        seed_condition_metrics({}, SCAN, DET)


def test_one_empty_condition_is_refused():
    rows = {c: list(v) for c, v in self_check_rows().items()}
    rows[SELF_END] = []
    with pytest.raises(EmptySeedConditionError) as exc:
        seed_condition_metrics(rows, SCAN, DET)
    assert exc.value.empty_conditions == [SELF_END]


def test_empty_seed_error_is_a_value_error():
    assert issubclass(EmptySeedConditionError, ValueError)


def test_nonempty_seed_primary_metric_finite_in_every_condition():
    rows = self_check_rows()
    assert all(len(rows.get(c, [])) > 0 for c in CONDITION_ORDER)
    out = seed_condition_metrics(rows, SCAN, DET)
    for c in CONDITION_ORDER:
        v = out[c]["itt_citation_migration_rate"]
        assert isinstance(v, float) and math.isfinite(v)
        assert out[c]["primary_metric"] == v
        assert out[c]["n_items"] > 0
    assert out[BASELINE]["baseline_determinism_rate"] == 1.0