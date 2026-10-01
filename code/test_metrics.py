import math
from collections import Counter

import pytest

from metrics import (CONDITION_ORDER, EDITED_CONDITIONS, NULL_ARM, NULL_ARM_PREFIX, PLAN_CONDITION_KEYS,
                     POSITIVE_ARM, POSITIVE_ARM_PREFIX, SELF_END, SWAP_END, answer_flip_rate, conditional_rate,
                     conditional_rate_intersection, determinism_rate, itt_rate, margin_terciles,
                     natural_location_rate, natural_weights, near_tie_concentration, neighbor_excess,
                     reliance_shares, seed_condition_metrics, stratified_itt, target_cited_rate)
from testing_fakes import row, self_check_rows

LEGACY_NULL_ARM = "null_trailing_whitespace_edit"
LEGACY_POSITIVE_ARM = "positive_control_evidence_relocated"


def test_condition_keys_match_plan():
    assert set(CONDITION_ORDER) == set(PLAN_CONDITION_KEYS)
    assert NULL_ARM == "null_random_drop_trailing_whitespace_edit"
    assert POSITIVE_ARM == "positive_control_planted_evidence_relocated"


def test_control_arm_keys_carry_mandatory_prefixes_and_old_names_are_gone():
    """Fails while the unprefixed legacy control-arm names are used anywhere in the condition contract."""
    assert NULL_ARM_PREFIX == "null_random_drop" and POSITIVE_ARM_PREFIX == "positive_control_planted"
    assert NULL_ARM.startswith(NULL_ARM_PREFIX) and NULL_ARM.startswith("null_")
    assert POSITIVE_ARM.startswith(POSITIVE_ARM_PREFIX) and POSITIVE_ARM.startswith("positive_control")
    for legacy in (LEGACY_NULL_ARM, LEGACY_POSITIVE_ARM):
        assert legacy not in (NULL_ARM, POSITIVE_ARM)
        assert legacy not in CONDITION_ORDER and legacy not in PLAN_CONDITION_KEYS
    assert NULL_ARM in PLAN_CONDITION_KEYS and POSITIVE_ARM in PLAN_CONDITION_KEYS


def test_itt_rate_toy_and_empty():
    assert itt_rate(self_check_rows()[SELF_END]) == pytest.approx(1 / 3)
    assert itt_rate([]) is None


def test_secondary_rates_toy_and_empty():
    rows = self_check_rows()
    assert target_cited_rate(rows[SELF_END]) == pytest.approx(2 / 3)
    assert answer_flip_rate(rows[SELF_END]) == pytest.approx(1 / 3)
    assert conditional_rate(rows[SELF_END]) == pytest.approx(0.5)
    assert conditional_rate(rows[SWAP_END]) is None
    assert target_cited_rate([]) is None and answer_flip_rate([]) is None


def test_intersection_by_id():
    by = {c: [row(c, s, kept=not (c == SWAP_END and s == "q2"), move=(c == SELF_END and s == "q1"))
              for s in ("q3", "q2", "q1")] for c in EDITED_CONDITIONS}
    assert conditional_rate_intersection(by)[SELF_END] == pytest.approx(0.5)


def test_neighbor_excess_toy():
    c = [row("x", "q2"), dict(row("x", "q1"), neighbor_next_new=True)]
    n = [row("n", "q1"), row("n", "q2")]
    assert neighbor_excess(c, n, "next") == pytest.approx(0.5)
    assert neighbor_excess(c, [], "next") is None


def test_natural_location_rate_toy():
    w = natural_weights([0.1, 0.5])
    assert w == [0.5, 0.0, 0.5, 0.0, 0.0]
    rows = [row("r", "a", move=True, obin=0), row("r", "b", obin=2)]
    assert natural_location_rate(rows, w) == pytest.approx(0.5)
    c = Counter()
    assert natural_location_rate([], w, c) is None and c["natural_bin_empty"] == 2
    assert natural_weights([]) is None


def test_margin_terciles_and_near_tie():
    rows = self_check_rows()
    ter = margin_terciles(rows["baseline_unedited_alce_citation"])
    assert ter == {(0, "q1"): 0, (0, "q2"): 1, (0, "q3"): 2}
    assert near_tie_concentration(rows[SELF_END], ter) == pytest.approx(1.0)
    assert near_tie_concentration([], ter) is None


def test_reliance_shares_empty_none():
    assert all(v is None for v in reliance_shares([]).values())
    assert reliance_shares(self_check_rows()[SELF_END])["re_sourcing_share"] == pytest.approx(1.0)


def test_stratified_itt_empty_stratum():
    out = stratified_itt(self_check_rows()[SELF_END])
    assert out["itt_single_source"] == pytest.approx(1 / 3)
    assert out["itt_redundant"] is None and out["itt_late"] is None and out["itt_tercile_0"] is None


def test_determinism_rate_toy():
    assert determinism_rate([("a", "a"), ("b", "b"), ("c", "d")]) == pytest.approx(2 / 3)
    assert determinism_rate([]) is None


def test_seed_condition_metrics_keys_and_finite():
    out = seed_condition_metrics(self_check_rows(), {"n_generated": 4, "n_parse_ok": 4}, [("a", "a")], [0.1])
    keysets = {frozenset(m) for m in out.values()}
    assert set(out) == set(CONDITION_ORDER) and len(keysets) == 1
    for m in out.values():
        for v in m.values():
            assert v is None or (isinstance(v, (int, float)) and math.isfinite(v))