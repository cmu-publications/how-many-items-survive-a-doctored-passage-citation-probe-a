"""Model loading is a go/no-go precondition: a missing or unloadable model prints
`PRECONDITION model_load: FAIL ...` and exits 1 before any other precondition, seed or write."""
import pytest

import main
from main import HYPERPARAMETERS, MODEL_LOAD_PRECONDITION, build_config, load_deps_or_fail
from design import _hp
from testing_fakes import FakeHarness, answer_table, make_fake_deps, toy_datasets, toy_pool


def _hp_dict():
    return _hp(build_config(HYPERPARAMETERS))


def _missing_model(hp, datasets):
    raise OSError("Qwen/Qwen2.5-3B-Instruct not found in the local cache")


class _Reader:
    dtype_name = "float32"


def test_missing_model_prints_fail_line_and_exits_1(capsys):
    with pytest.raises(SystemExit) as info:
        load_deps_or_fail(_hp_dict(), toy_datasets(), loader=_missing_model)
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert f"PRECONDITION {MODEL_LOAD_PRECONDITION}: FAIL OSError" in out
    assert "not found in the local cache" in out
    assert "PRECONDITION_FAILED" in out


def test_loaded_models_print_pass_line(capsys):
    hp = _hp_dict()
    deps = make_fake_deps(hp, answer_table(toy_pool()))

    def ok_loader(hp_, datasets):
        return deps, _Reader()

    got_deps, reader = load_deps_or_fail(hp, toy_datasets(), loader=ok_loader)
    assert got_deps is deps and reader.dtype_name == "float32"
    assert f"PRECONDITION {MODEL_LOAD_PRECONDITION}: pass" in capsys.readouterr().out


def test_main_turns_a_model_load_error_into_a_precondition_fail(monkeypatch, capsys):
    harness = FakeHarness()
    monkeypatch.setattr(main.experiment_harness, "get_harness", lambda budget: harness, raising=False)
    monkeypatch.setattr(main.experiment_harness, "seeds", lambda s: list(s), raising=False)
    monkeypatch.setattr(main, "get_datasets", lambda root: toy_datasets())
    monkeypatch.setattr(main, "check_pools", lambda *a, **k: (True, []))
    monkeypatch.setattr(main, "build_real_deps", _missing_model)
    with pytest.raises(SystemExit) as info:
        main.main([])
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert f"PRECONDITION {MODEL_LOAD_PRECONDITION}: FAIL OSError" in out
    assert harness.seed_records() == []