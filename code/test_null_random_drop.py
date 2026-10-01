"""The null arm is the plan's trailing-whitespace edit (plan item 20): slot t text + "\\n", every other slot
and slot t's title unchanged. It drops no sentence, draws nothing at random and adds no eligibility criterion.
These tests pin that single null-arm specification through its behaviour."""
import copy

import pytest

from baselines import split_sentences
from main import HYPERPARAMETERS
from metrics import CONDITION_ORDER, NULL_ARM, NULL_ARM_PREFIX
from probes import (NULL_EDIT_PLAN, NullTrailingWhitespaceProbe, assert_only_slots_changed, build_arms,
                    null_edit)
from testing_fakes import make_fake_deps

HP = HYPERPARAMETERS
SEED = 7
T = 3
TARGET_TEXT = ("Alpha one is here. Beta two is there. Gamma three is near. "
               "Delta four sits far away from here. Epsilon five waits by the river bank.")
SPAN = "Paris is the capital and largest city of France."
FORBIDDEN_NULL_FIELD_FRAGMENTS = ("retention", "dropped", "drop_rate", "drop_idx")


def _item():
    docs = [{"title": f"Title {k}", "text": f"Passage {k} opens. It has a middle. It ends here."}
            for k in range(1, 6)]
    docs[T - 1]["text"] = TARGET_TEXT
    return {"sample_id": "null1", "question": "What is the capital of France?", "aliases": ["paris"],
            "context_docs": docs, "redundancy": "single_source"}


def _payload():
    return {"t": T, "o": 1, "span": SPAN, "stratum": "late"}


def _deps():
    return make_fake_deps(HP, {})


def test_exactly_one_null_arm_is_registered_and_it_is_the_whitespace_edit():
    arms = build_arms(HP)
    null_conds = [c for c in CONDITION_ORDER if c.startswith(NULL_ARM_PREFIX)]
    assert null_conds == [NULL_ARM]
    assert type(arms[NULL_ARM]) is NullTrailingWhitespaceProbe


def test_null_arm_is_the_registered_null_condition():
    arms = build_arms(HP)
    assert isinstance(arms[NULL_ARM], NullTrailingWhitespaceProbe)
    assert NULL_ARM.startswith(NULL_ARM_PREFIX)
    assert NULL_EDIT_PLAN.startswith("slot t text + ")


def test_null_edit_appends_only_a_newline_to_slot_t():
    item = _item()
    arm = NullTrailingWhitespaceProbe(HP)
    docs, meta = arm.edit_context(item, _payload())
    assert docs[T - 1]["text"] == TARGET_TEXT + "\n"
    assert docs[T - 1]["title"] == item["context_docs"][T - 1]["title"]
    assert meta["changed_slots"] == [T]
    assert meta["plant"] == "\n"
    assert_only_slots_changed(item["context_docs"], docs, meta["changed_slots"])
    assert docs == null_edit(item["context_docs"], T)


def test_null_edit_drops_no_sentence_and_no_word():
    item = _item()
    docs, _meta = NullTrailingWhitespaceProbe(HP).edit_context(item, _payload())
    before = [s.strip() for s in split_sentences(TARGET_TEXT)]
    after = [s.strip() for s in split_sentences(docs[T - 1]["text"])]
    assert before == after
    assert docs[T - 1]["text"].split() == TARGET_TEXT.split()
    assert SPAN not in docs[T - 1]["text"]


def test_null_edit_does_not_mutate_the_input_item():
    item = _item()
    snapshot = copy.deepcopy(item)
    NullTrailingWhitespaceProbe(HP).edit_context(item, _payload())
    assert item == snapshot


def test_null_prepare_writes_nothing_and_excludes_nothing():
    payload = _payload()
    before = dict(payload)
    arm = NullTrailingWhitespaceProbe(HP)
    assert arm.prepare(_item(), {}, payload, SEED, _deps()) is None
    assert payload == before


def test_null_edit_is_seed_independent():
    arm = NullTrailingWhitespaceProbe(HP)
    p1, p2 = _payload(), _payload()
    assert arm.prepare(_item(), {}, p1, SEED, _deps()) is None
    assert arm.prepare(_item(), {}, p2, SEED + 1, _deps()) is None
    d1, _ = arm.edit_context(_item(), p1)
    d2, _ = arm.edit_context(_item(), p2)
    assert d1 == d2


def test_null_row_records_the_plan_edit():
    item, payload = _item(), _payload()
    arm = NullTrailingWhitespaceProbe(HP)
    _docs, meta = arm.edit_context(item, payload)
    row = arm._row_meta(item, {"margin": 0.2}, payload, meta, "edited")
    assert row["condition"] == NULL_ARM
    assert row["null_edit"] == NULL_EDIT_PLAN
    assert row["t"] == T
    assert row["plant"] == "\n"


def test_null_row_and_payload_carry_no_sentence_drop_fields():
    item, payload = _item(), _payload()
    arm = NullTrailingWhitespaceProbe(HP)
    assert arm.prepare(item, {}, payload, SEED, _deps()) is None
    _docs, meta = arm.edit_context(item, payload)
    row = arm._row_meta(item, {"margin": 0.2}, payload, meta, "edited")
    for key in list(row) + list(payload) + list(meta):
        for frag in FORBIDDEN_NULL_FIELD_FRAGMENTS:
            assert frag not in str(key), f"null arm carries sentence-drop field {key}"


def test_null_prepare_without_target_raises():
    arm = NullTrailingWhitespaceProbe(HP)
    with pytest.raises(ValueError):
        arm.prepare(_item(), {}, {"o": 1, "span": SPAN}, SEED, _deps())


def test_null_prepare_with_out_of_range_target_raises():
    arm = NullTrailingWhitespaceProbe(HP)
    with pytest.raises(ValueError):
        arm.prepare(_item(), {}, {"t": int(HP["n_passages"]) + 1, "o": 1}, SEED, _deps())