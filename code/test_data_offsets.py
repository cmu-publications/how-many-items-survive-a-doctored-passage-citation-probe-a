import json
from typing import List, Tuple

import pytest

from data import DATA_CONFIG, alias_sentence_offsets, get_datasets, sentence_starts

# The regex splitter breaks after "Dr."; a spaCy-like splitter keeps "Dr. Smith went home." whole.
TEXT = "Dr. Smith went home. Paris is big."


def probe_like_splitter(text: str) -> List[Tuple[int, int]]:
    """Stand-in for the probe's span splitter: one boundary, before 'Paris' only."""
    if "Paris" not in text:
        return [(0, len(text))] if text else []
    i = text.index("Paris")
    return [(0, i - 1), (i, len(text))]


def test_regex_and_probe_splitter_disagree_on_fixture():
    assert sentence_starts(TEXT) != [s for s, _ in probe_like_splitter(TEXT)]


def test_offsets_follow_the_probe_splitter_not_the_regex():
    docs = [{"title": "t", "text": TEXT}]
    offs = alias_sentence_offsets(["smith"], docs, probe_like_splitter)
    assert offs == [0.0]
    regex_start = sentence_starts(TEXT)[1] / len(TEXT)
    assert regex_start not in offs


def test_offsets_require_a_splitter():
    with pytest.raises(TypeError):
        alias_sentence_offsets(["smith"], [{"title": "t", "text": TEXT}], None)  # type: ignore[arg-type]


def test_offsets_reject_out_of_range_spans():
    with pytest.raises(ValueError):
        alias_sentence_offsets(["smith"], [{"title": "t", "text": TEXT}], lambda t: [(0, len(t) + 5)])


def test_get_datasets_uses_supplied_probe_splitter(tmp_path):
    root = tmp_path / "data"
    (root / "alce").mkdir(parents=True)
    docs = [{"title": "d1", "text": TEXT}] + [{"title": f"d{i}", "text": "Filler text here."} for i in range(2, 21)]
    items = [{"sample_id": f"s{i}", "question": "Who went home?", "answer": "Smith.",
              "qa_pairs": [{"short_answers": ["Smith"]}], "docs": docs} for i in range(6)]
    (root / "alce" / DATA_CONFIG["eval_file"]).write_text(json.dumps(items), encoding="utf-8")
    (root / "alce" / DATA_CONFIG["demo_file"]).write_text(json.dumps({"demos": []}), encoding="utf-8")
    ds = get_datasets(str(root), span_splitter=probe_like_splitter)
    all_items = ds["val"].items + ds["test"].items
    assert len(all_items) == 6
    for it in all_items:
        assert it["alias_sentence_offsets"] == [0.0]
    assert DATA_CONFIG["alias_offset_splitter"] == "caller-supplied probe span splitter"