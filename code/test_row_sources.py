import main  # noqa: F401  sets the CUDA / HF-offline environment before torch is imported
import row_sources
import seed_loop
from row_sources import preserved_rows, restore_extra, source_summary
from seed_loop import BUDGET_STOP_ROWS_KEY


def test_restore_extra_fills_missing_item_rows_from_process_copy(monkeypatch):
    monkeypatch.setattr(row_sources, "SEED_EXTRA_STORE", {"3": {"item_rows": [{"a": 1}], "counters": {"x": 1}}})
    extra, restored = restore_extra(3, {"design": {"n_items": 1}})
    assert restored is True
    assert extra["item_rows"] == [{"a": 1}]
    assert extra["design"] == {"n_items": 1}


def test_restore_extra_keeps_harness_rows(monkeypatch):
    monkeypatch.setattr(row_sources, "SEED_EXTRA_STORE", {"3": {"item_rows": [{"a": 1}]}})
    extra, restored = restore_extra(3, {"item_rows": []})
    assert restored is False
    assert extra["item_rows"] == []


def test_restore_extra_without_copy_is_unchanged(monkeypatch):
    monkeypatch.setattr(row_sources, "SEED_EXTRA_STORE", {})
    extra, restored = restore_extra(0, {})
    assert restored is False and extra == {}


def test_capture_stores_payload_after_record(monkeypatch):
    calls = []

    def fake_record(harness, seed, conds, extra):
        calls.append(seed)

    monkeypatch.setattr(seed_loop, "record_seed_payload", fake_record)
    monkeypatch.setattr(row_sources, "SEED_EXTRA_STORE", {})
    assert row_sources.install_seed_extra_capture() is True
    assert row_sources.install_seed_extra_capture() is False
    seed_loop.record_seed_payload(None, 5, {}, {"item_rows": [{"r": 1}]})
    assert calls == [5]
    assert row_sources.SEED_EXTRA_STORE["5"]["item_rows"] == [{"r": 1}]


def test_preserved_rows_from_failures_and_budget_stops():
    timing = {"seed_failures": [{"seed": 0, "partial": {"item_rows": [{"a": 1}, {"a": 2}],
                                                         "alias_offsets": [0.5, float("nan")]}}],
              "unrun_seeds": [{"seed": 1, BUDGET_STOP_ROWS_KEY: [{"b": 1}]}, {"seed": 2, "reason": "x"}]}
    rows, offsets, sources = preserved_rows(timing)
    assert len(rows) == 3
    assert offsets == [0.5]
    assert [s["seed"] for s in sources] == ["0", "1"]
    assert "seed_failed:seed=0:n=2" in source_summary(sources)


def test_preserved_rows_empty():
    rows, offsets, sources = preserved_rows({})
    assert rows == [] and offsets == [] and sources == []
    assert source_summary(sources) == "none"