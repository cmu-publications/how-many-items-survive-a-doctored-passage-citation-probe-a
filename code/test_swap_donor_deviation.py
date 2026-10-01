from metrics import CONDITION_ORDER, SWAP_END
from probes import DONOR_POOL_RULE, SWAP_DONOR_PLAN, EntitySwapFoilProbe, probe_deviations


def _swap_entries():
    return [d for d in probe_deviations() if d["component"] == SWAP_END]


def test_swap_donor_population_is_a_run_level_deviation():
    entries = _swap_entries()
    assert len(entries) == 1
    dev = entries[0]
    assert dev["key"] == "donor_population"
    assert dev["planned"] == SWAP_DONOR_PLAN
    assert dev["used"] == DONOR_POOL_RULE
    assert "c7" in dev["reason"]
    assert SWAP_END in CONDITION_ORDER


def test_every_deviation_entry_has_all_fields():
    for dev in probe_deviations():
        for field in ("component", "key", "planned", "used", "reason"):
            assert isinstance(dev[field], str) and dev[field]


def _ner(text):
    out = []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        end = start + len(word)
        pos = end
        clean = word.strip(".,")
        if clean[:1].isupper():
            out.append((start, start + len(clean), "GPE", clean))
    return out


def test_donor_pool_includes_uncovered_items_as_the_deviation_states():
    pool = [
        {"sample_id": "a", "aliases": ["paris"], "gold_long_answer": "The answer is Paris.", "covered": True},
        {"sample_id": "b", "aliases": ["berlin"], "gold_long_answer": "The answer is Berlin.", "covered": False},
    ]
    donors = EntitySwapFoilProbe.build_donor_pool(pool, _ner, ["GPE"])
    donor_ids = {sid for sid, _text in donors.get("GPE", [])}
    assert donor_ids == {"a", "b"}