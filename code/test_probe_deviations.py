from main import HYPERPARAMETERS
from metrics import CONDITION_ORDER, NULL_ARM, POSITIVE_ARM
from probes import NULL_EDIT_PLAN, NullTrailingWhitespaceProbe, null_edit, probe_deviations

FORBIDDEN_NULL_ROW_FRAGMENTS = ("retention", "deviation", "dropped", "drop_rate", "drop_idx")


def _docs():
    return [{"title": f"T{k}", "text": f"Passage {k} one. Passage {k} two. Passage {k} three."}
            for k in range(1, 6)]


def _null_row(t=2):
    arm = NullTrailingWhitespaceProbe(HYPERPARAMETERS)
    item = {"sample_id": "q1", "redundancy": "single_source", "context_docs": _docs()}
    return arm._row_meta(item, {"margin": 0.1}, {"t": t, "o": 1}, {"plant": "\n"}, "x")


def test_null_arm_is_not_a_recorded_deviation():
    """Plan item 20 is implemented as written, so the null arm has no deviation entry."""
    assert not [d for d in probe_deviations() if d["component"] == NULL_ARM]


def test_every_deviation_names_a_known_condition_and_all_fields():
    devs = probe_deviations()
    assert devs
    for d in devs:
        assert d["component"] in CONDITION_ORDER
        assert set(d) == {"component", "key", "planned", "used", "reason"}
        assert all(isinstance(d[k], str) and d[k] for k in d)
    assert any(d["component"] == POSITIVE_ARM for d in devs)


def test_null_arm_edit_is_the_plan_helper():
    arm = NullTrailingWhitespaceProbe(HYPERPARAMETERS)
    item = {"context_docs": _docs()}
    for t in range(1, 6):
        edited, meta = arm.edit_context(item, {"t": t})
        assert edited == null_edit(_docs(), t)
        assert meta["changed_slots"] == [t]


def test_null_row_records_the_plan_edit():
    row = _null_row()
    assert row["null_edit"] == NULL_EDIT_PLAN
    assert row["condition"] == NULL_ARM


def test_null_row_has_no_retention_rate():
    """The plan's null arm drops nothing, so a retention rate would describe an edit that never ran."""
    row = _null_row()
    assert "null_retention_rate" not in row
    assert "null_edit_deviation" not in row


def test_null_row_carries_no_sentence_drop_field_for_any_target():
    for t in range(1, 6):
        row = _null_row(t)
        leaked = sorted(k for k in row if any(f in str(k) for f in FORBIDDEN_NULL_ROW_FRAGMENTS))
        assert leaked == [], f"null row for t={t} carries sentence-drop fields {leaked}"