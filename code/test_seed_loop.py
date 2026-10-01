"""Seed-loop contracts: no write after an abort, guarded determinism reruns, metric failures keep rows."""
import time
from collections import Counter

import pytest

import main  # sets the CUDA / HF-offline environment before torch is imported
import seed_loop
from main import HYPERPARAMETERS, HarnessAborted, build_config, run_all_seeds, run_and_write
from metrics import determinism_rate
from seed_loop import DET_REGEN_FAILED, determinism_pairs
from testing_fakes import FakeHarness, answer_table, make_fake_deps, make_item, toy_datasets, toy_pool

assert main.HYPERPARAMETERS is HYPERPARAMETERS


def _design():
    return {"n_items": 10, "quota": {"single_source": 5, "redundant": 5}, "scan_cap": None, "det_items": 20,
            "est_sec": 100.0}


def _run_info():
    return {"t_start": time.time(), "est_sec": 100.0, "unrun_seeds": [], "seed_failures": []}


class _ExitHarness(FakeHarness):
    def should_stop(self):
        raise SystemExit(3)


class _BrokenRecordHarness(FakeHarness):
    def record_seed(self, seed, conditions, expected_conditions=None, extra=None):
        raise OSError("disk full")


def test_run_and_write_writes_nothing_after_system_exit():
    h = _ExitHarness()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    with pytest.raises(SystemExit):
        run_and_write(h, _design(), [0, 1], deps, toy_datasets(), build_config(HYPERPARAMETERS), _run_info(), 1e12)
    assert getattr(h, "written", None) is None


def test_run_and_write_writes_nothing_after_harness_abort(capsys):
    h = _BrokenRecordHarness()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    with pytest.raises(HarnessAborted):
        run_and_write(h, _design(), [0, 1], deps, toy_datasets(), build_config(HYPERPARAMETERS), _run_info(), 1e12)
    assert getattr(h, "written", None) is None
    assert "ABORTED" in capsys.readouterr().out


def test_determinism_regen_error_is_counted_not_raised():
    item = make_item("q1", "Paris")
    ok_item = make_item("q2", "Berlin")

    def gen(question, docs):
        if question == item["question"]:
            raise RuntimeError("CUDA out of memory")
        return "same"

    c = Counter()
    pairs, errors = determinism_pairs(["q1", "q2"], {"q1": item, "q2": ok_item},
                                      {"q1": {"answer": "Paris [1]."}, "q2": {"answer": "same"}}, gen, c, 0)
    assert c["determinism_regen_failed"] == 1
    assert errors == [{"sample_id": "q1", "error_class": "RuntimeError", "message": "CUDA out of memory"}]
    assert pairs[0][1] == DET_REGEN_FAILED
    assert determinism_rate(pairs) == pytest.approx(0.5)


def test_metric_failure_keeps_item_rows(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("non-finite metric")

    monkeypatch.setattr(seed_loop, "seed_condition_metrics", boom)
    h = FakeHarness()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(toy_pool()))
    run_info = _run_info()
    run_all_seeds([0], _design(), deps, toy_datasets(), build_config(HYPERPARAMETERS), h, run_info, 1e12)
    assert h.seed_records() == []
    failure = run_info["seed_failures"][0]
    assert failure["error_class"] == "SeedFailed"
    assert failure["partial"]["item_rows"] and failure["partial"]["included_ids"]