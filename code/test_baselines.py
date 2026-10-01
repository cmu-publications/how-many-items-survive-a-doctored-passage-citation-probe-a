from collections import Counter

import pytest

from baselines import (blank_passage, citation_positions, count_truncation, extract_self_span, mean_nll_or_none,
                       parse_citations, score_outcome, select_dtype, slots_from_logprobs, split_sentences)


def test_imports_pure_modules():
    import analysis
    import metrics
    import probes
    assert metrics.itt_rate([]) is None
    assert analysis.holm({}) == {}
    assert probes.content_tokens("The cat", {"the"}) == {"cat"}


def test_split_sentences_with_fake_splitter():
    assert split_sentences("a|b", lambda t: t.split("|")) == ["a", "b"]
    assert split_sentences("One. Two.") == ["One.", "Two."]


def test_select_dtype_by_capability():
    assert select_dtype((7, 5)) == "float16"
    assert select_dtype((8, 0)) == "bfloat16"
    assert select_dtype(None) == "float32"


def test_citation_slots_from_fake_logits():
    slots = slots_from_logprobs([[-0.1, -2.0, -3.0, -4.0, -5.0]], [1])
    assert slots[0]["margin"] == pytest.approx(1.9)
    with pytest.raises(FloatingPointError):
        slots_from_logprobs([[float("nan")] * 5], [1])


def test_citation_positions_merged_digit_counter():
    c = Counter()
    offs = [(0, 10), (10, 11), (11, 13), (13, 14), (14, 15), (15, 17), (17, 19), (19, 21), (21, 22)]
    pos, digits = citation_positions("A [1] B [2].", 10, offs, c)
    assert pos == [2] and digits == [1] and c["digit_merged_slot_skipped"] == 1


def test_blank_passage_only_slot_k():
    docs = [{"title": f"t{i}", "text": f"x{i}"} for i in range(5)]
    out = blank_passage(docs, 3)
    assert out[2] == {"title": "", "text": ""}
    assert [d for i, d in enumerate(out) if i != 2] == [d for i, d in enumerate(docs) if i != 2]


def test_parse_citations_and_score_outcome_move_rule():
    assert parse_citations("x [1,2] y [3]") == ({1, 2, 3}, True)
    assert parse_citations("none") == (set(), False)
    base = {"cited": [1], "alias_set": ["paris"]}
    r = score_outcome(base, "Paris is it [1][2].", 2, ["paris"])
    assert r["move"] and r["parse_ok"] and r["target_cited"] and r["answer_kept"]
    r = score_outcome(base, "Berlin [2].", 2, ["paris"])
    assert r["target_cited"] and not r["move"] and r["answer_flip"]
    r = score_outcome(base, "Paris [2][7].", 2, ["paris"])
    assert not r["parse_ok"] and not r["target_cited"] and not r["move"]


def test_score_outcome_move_rule():
    base = {"cited": [1, 2], "alias_set": ["paris"]}
    r = score_outcome(base, "Paris [2].", 2, ["paris"])
    assert not r["target_cited"] and not r["move"]


def test_extract_self_span_and_source():
    span, ids = extract_self_span("Intro sentence [2]. Paris is big [1][3]. More.", ["paris"])
    assert span == "Paris is big." and ids == [1, 3] and min(ids) == 1
    assert extract_self_span("Nothing here [1].", ["paris"]) == (None, [])


def test_mean_nll_and_truncation_counters():
    c = Counter()
    assert mean_nll_or_none([], c) is None and c["ppl_too_short"] == 1
    assert mean_nll_or_none([-1.0, -3.0], c) == pytest.approx(2.0)
    assert count_truncation(600, 512, c) and c["nli_truncated"] == 1