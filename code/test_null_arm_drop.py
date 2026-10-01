import pytest

from data import PURPOSE_CODE, item_rng
from metrics import NULL_ARM, NULL_ARM_PREFIX, POSITIVE_ARM, SELF_END
from probes import NULL_DROP_PURPOSE, NULL_DROP_RULE, NULL_DROP_TREATMENT, null_random_drop

SEED = 11


def _rows(n, kept_fn):
    rows = {}
    for i in range(n):
        sid = f"s{i:03d}"
        rows[(SELF_END, sid)] = {"condition": SELF_END, "sample_id": sid, "kept": kept_fn(i)}
        rows[(NULL_ARM, sid)] = {"condition": NULL_ARM, "sample_id": sid, "kept": True}
        rows[(POSITIVE_ARM, sid)] = {"condition": POSITIVE_ARM, "sample_id": sid, "kept": True}
    return rows


def _null_ids(rows):
    return sorted(sid for (cond, sid) in rows if cond == NULL_ARM)


def test_null_drop_purpose_is_a_registered_item_rng_purpose():
    assert NULL_DROP_PURPOSE in PURPOSE_CODE
    u = float(item_rng(SEED, "s000", NULL_DROP_PURPOSE).random())
    assert 0.0 <= u < 1.0
    assert f"'{NULL_DROP_PURPOSE}'" in NULL_DROP_RULE
    codes = list(PURPOSE_CODE.values())
    assert len(codes) == len(set(codes))


def test_null_arm_named_random_drop_actually_drops_at_treatment_rate():
    assert NULL_ARM.startswith(NULL_ARM_PREFIX)
    assert NULL_DROP_TREATMENT == SELF_END
    rows = _rows(400, lambda i: i % 2 == 0)
    out, rec = null_random_drop(rows, SEED)
    assert rec["treatment_kept_rate"] == pytest.approx(0.5)
    kept_share = len(_null_ids(out)) / 400
    assert 0.40 <= kept_share <= 0.60
    assert rec["n_null_before"] == 400
    assert rec["n_null_after"] == len(_null_ids(out))
    assert len(rec["dropped_sample_ids"]) == 400 - rec["n_null_after"] > 0


def test_drop_decision_follows_the_seeded_item_draw():
    rows = _rows(50, lambda i: i % 4 != 0)
    out, rec = null_random_drop(rows, SEED)
    rate = rec["treatment_kept_rate"]
    for i in range(50):
        sid = f"s{i:03d}"
        u = float(item_rng(SEED, sid, NULL_DROP_PURPOSE).random())
        assert ((NULL_ARM, sid) in out) == (u < rate)


def test_full_treatment_retention_keeps_every_null_row_and_zero_drops_all():
    out_all, rec_all = null_random_drop(_rows(30, lambda i: True), SEED)
    assert len(_null_ids(out_all)) == 30 and rec_all["dropped_sample_ids"] == []
    out_none, rec_none = null_random_drop(_rows(30, lambda i: False), SEED)
    assert _null_ids(out_none) == [] and rec_none["n_null_after"] == 0


def test_other_arms_untouched_and_input_not_mutated():
    rows = _rows(40, lambda i: i % 3 == 0)
    before = dict(rows)
    out, _rec = null_random_drop(rows, SEED)
    assert rows == before
    for key, row in rows.items():
        if key[0] != NULL_ARM:
            assert out[key] is row


def test_deterministic_and_order_independent():
    rows = _rows(60, lambda i: i % 2 == 1)
    reversed_rows = dict(reversed(list(rows.items())))
    out1, rec1 = null_random_drop(rows, SEED)
    out2, rec2 = null_random_drop(reversed_rows, SEED)
    assert _null_ids(out1) == _null_ids(out2)
    assert rec1 == rec2


def test_missing_treatment_rows_or_kept_field_raises():
    only_null = {(NULL_ARM, "a"): {"condition": NULL_ARM, "sample_id": "a", "kept": True}}
    with pytest.raises(ValueError):
        null_random_drop(only_null, SEED)
    no_kept = {(SELF_END, "a"): {"condition": SELF_END, "sample_id": "a"},
               (NULL_ARM, "a"): {"condition": NULL_ARM, "sample_id": "a", "kept": True}}
    with pytest.raises(ValueError):
        null_random_drop(no_kept, SEED)