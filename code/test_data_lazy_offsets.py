"""get_datasets must not run the spaCy alias-offset pass over the whole corpus at load time."""
import copy
import json
import re
from typing import List, Tuple

import pytest

from data import alias_offsets_computed, alias_sentence_offsets, get_datasets

N_ITEMS = 12
N_DOCS = 5
_SPLIT = re.compile(r"[^.]+\.\s*")


class CountingSplitter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> List[Tuple[int, int]]:
        self.calls += 1
        return [(m.start(), m.start() + len(m.group(0).rstrip())) for m in _SPLIT.finditer(text)]


def _write_fixture(root) -> None:
    alce = root / "alce"
    alce.mkdir()
    items = []
    for i in range(N_ITEMS):
        docs = [{"title": f"T{i}-{j}", "text": f"Filler sentence {j}. Paris is the answer here. More text {j}."}
                for j in range(N_DOCS)]
        items.append({"sample_id": f"s{i}", "question": f"What city {i}?", "answer": "Paris.",
                      "qa_pairs": [{"short_answers": ["Paris"]}], "docs": docs})
    (alce / "asqa_eval_gtr_top100.json").write_text(json.dumps(items), encoding="utf-8")
    (alce / "asqa_default.json").write_text(json.dumps(
        {"instruction": "Answer.", "demo_prompt": "", "doc_prompt": "", "demos": []}), encoding="utf-8")


def _all_items(ds):
    return list(ds["val"].items) + list(ds["test"].items)


def test_loading_runs_no_offset_pass(tmp_path):
    _write_fixture(tmp_path)
    splitter = CountingSplitter()
    before = alias_offsets_computed()
    ds = get_datasets(str(tmp_path), span_splitter=splitter)
    assert splitter.calls == 0
    assert alias_offsets_computed() == before
    items = _all_items(ds)
    assert len(items) == N_ITEMS
    assert all(it["covered"] for it in items)
    assert splitter.calls == 0


def test_offsets_computed_only_for_the_item_used(tmp_path):
    _write_fixture(tmp_path)
    splitter = CountingSplitter()
    ds = get_datasets(str(tmp_path), span_splitter=splitter)
    item = _all_items(ds)[0]
    before = alias_offsets_computed()
    offs = item["alias_sentence_offsets"]
    assert splitter.calls == N_DOCS
    assert alias_offsets_computed() == before + 1
    expected = alias_sentence_offsets(item["aliases"], item["context_docs"], CountingSplitter())
    assert offs == expected and len(offs) == N_DOCS
    assert item.get("alias_sentence_offsets") == offs
    assert splitter.calls == N_DOCS


def test_bulk_views_and_copies_include_offsets(tmp_path):
    _write_fixture(tmp_path)
    splitter = CountingSplitter()
    ds = get_datasets(str(tmp_path), span_splitter=splitter)
    a, b, c = _all_items(ds)[:3]
    assert "alias_sentence_offsets" in dict(a)
    assert "alias_sentence_offsets" in {**b}
    deep = copy.deepcopy(c)
    assert deep["alias_sentence_offsets"] == c["alias_sentence_offsets"]
    assert splitter.calls == 3 * N_DOCS


def test_unknown_key_still_raises(tmp_path):
    _write_fixture(tmp_path)
    ds = get_datasets(str(tmp_path), span_splitter=CountingSplitter())
    with pytest.raises(KeyError):
        _ = _all_items(ds)[0]["no_such_key"]