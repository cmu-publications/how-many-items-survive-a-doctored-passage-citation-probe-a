import json
from collections import Counter

import numpy as np
import pytest

from runtime_io import ReplayCache, _memo, finalize_results, sanitize_for_json


def test_replay_cache_counts_every_hit_and_survives_counter_reset():
    cache = ReplayCache()
    calls = []

    def compute(local):
        calls.append(1)
        local["ppl_too_short"] += 1
        return 2.0

    c = Counter()
    assert cache.get("k", compute, c) == 2.0
    assert cache.get("k", compute, c) == 2.0
    assert len(calls) == 1 and c["ppl_too_short"] == 2
    fresh = Counter()
    assert cache.get("k", compute, fresh) == 2.0
    assert fresh["ppl_too_short"] == 1 and len(cache) == 1


def test_memo_replays_ppl_too_short_on_hits():
    seen = []

    def nll(sentence, counters):
        seen.append(sentence)
        counters["ppl_too_short"] += 1
        return None

    memo = _memo(nll)
    c = Counter()
    assert memo("a", c) is None and memo("a", c) is None
    assert seen == ["a"] and c["ppl_too_short"] == 2


def test_replay_cache_does_not_cache_exceptions():
    cache = ReplayCache()
    state = {"fail": True}

    def compute(local):
        if state["fail"]:
            raise ValueError("boom")
        return 1.0

    with pytest.raises(ValueError):
        cache.get("k", compute, Counter())
    assert len(cache) == 0
    state["fail"] = False
    assert cache.get("k", compute, Counter()) == 1.0 and len(cache) == 1


def test_sanitize_records_non_finite_paths():
    obj = {"a": float("nan"), "b": [1, float("inf")], "c": np.float64(0.5), "d": object(), "e": {2, 1}}
    clean, non_finite, stringified = sanitize_for_json(obj)
    assert clean["a"] is None and clean["b"] == [1, None] and clean["c"] == 0.5 and clean["e"] == [1, 2]
    assert non_finite == ["$.a", "$.b[1]"] and stringified == ["$.d"]
    json.dumps(clean, allow_nan=False)


def test_finalize_never_prints_primary_line(capsys):
    """Fails while finalize_results prints its own PRIMARY / primary_metric line."""
    res = {"primary_metric_key": "itt_citation_migration_rate", "primary_metric": 0.25,
           "primary_condition": "topical_foil_end_injection", "x": float("nan")}
    clean = finalize_results(res)
    out = capsys.readouterr().out
    assert "PRIMARY" not in out and "primary_metric:" not in out
    assert "FLAG: non-finite value at $.x written as null" in out
    assert clean["non_finite_values_replaced"] == ["$.x"] and clean["x"] is None
    assert clean["primary_metric"] == 0.25
    json.dumps(clean, allow_nan=False)
    assert "PRIMARY" not in capsys.readouterr().out
    clean_none = finalize_results({"primary_metric": None})
    assert clean_none["primary_metric"] is None and "PRIMARY" not in capsys.readouterr().out


def test_finalize_writes_no_results_file(tmp_path, monkeypatch):
    """Fails while runtime_io keeps a local results.json writer next to experiment_harness: finalizing a
    full results dict, and a minimal one, must leave the working directory empty."""
    monkeypatch.chdir(tmp_path)
    full = {"primary_metric_key": "itt_citation_migration_rate", "primary_metric": 0.1,
            "condition_names": ["baseline_unedited_alce_citation"], "skipped_components": [],
            "reduced_components": [], "y": [float("inf")]}
    clean = finalize_results(full)
    assert clean["y"] == [None]
    finalize_results({"primary_metric": 0.1})
    assert list(tmp_path.iterdir()) == []