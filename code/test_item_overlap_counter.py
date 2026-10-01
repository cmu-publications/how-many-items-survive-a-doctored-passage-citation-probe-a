"""fallback_counters.item_overlap_across_seeds must equal item_reuse_across_seeds.n_reused."""
from main import CONDITION_NAMES, HYPERPARAMETERS, aggregate_and_write, build_config
from testing_fakes import FakeHarness


def _conds():
    return {c: {"itt_citation_migration_rate": 0.1, "target_cited_rate": 0.8} for c in CONDITION_NAMES}


def _record(seed, ids, stale_counter=0):
    return {"seed": seed, "conditions": _conds(),
            "extra": {"included_ids": list(ids), "counters": {"item_overlap_across_seeds": stale_counter}}}


def _run(records):
    harness = FakeHarness(records)
    return aggregate_and_write(harness, build_config(HYPERPARAMETERS), {"t_start": 0.0})


def test_overlap_counter_matches_reuse_field():
    out = _run([_record(0, ["a", "b", "c"]), _record(1, ["b", "c", "d"]), _record(2, ["c", "e"])])
    assert out["item_reuse_across_seeds"]["n_reused"] == 2
    assert out["fallback_counters"]["item_overlap_across_seeds"] == 2


def test_overlap_counter_ignores_per_seed_values():
    out = _run([_record(0, ["a"], stale_counter=5), _record(1, ["a"], stale_counter=5)])
    assert out["item_reuse_across_seeds"]["n_reused"] == 1
    assert out["fallback_counters"]["item_overlap_across_seeds"] == 1


def test_no_overlap_gives_zero_in_both():
    out = _run([_record(0, ["a"]), _record(1, ["b"])])
    assert out["item_reuse_across_seeds"]["n_reused"] == 0
    assert out["fallback_counters"]["item_overlap_across_seeds"] == 0