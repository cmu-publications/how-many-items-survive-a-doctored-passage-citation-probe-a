import math
from collections import Counter

import pytest

from data import item_rng
from main import HYPERPARAMETERS
from metrics import CONDITION_ORDER
from probes import (BaselineUneditedArm, EntitySwapFoilProbe, MatchedTopicalFoilInjectionProbe,
                    RandomBoundarySelfSpanProbe, RandomBoundaryTopicalFoilProbe, alce_citation_proxy,
                    check_eligibility, delta_logp_target, null_edit, relocate_evidence, reliance_audit, run_item,
                    select_target)
from baselines import score_outcome
from testing_fakes import answer_table, fake_ner, fake_nli, make_fake_deps, make_item, toy_pool

HP = HYPERPARAMETERS


def _ready(item_id="q1"):
    pool = toy_pool()
    deps = make_fake_deps(HP, answer_table(pool))
    deps.set_seed_pool(pool, EntitySwapFoilProbe.build_donor_pool(pool, deps.ner, HP["entity_types"]))
    item = {it["sample_id"]: it for it in pool}[item_id]
    base = BaselineUneditedArm.generate_baseline(item, deps.generate, deps.tf_slots)
    return pool, deps, item, base


def test_eligibility_first_failing_criterion():
    pool, deps, item, base = _ready()
    assert check_eligibility(dict(item, covered=False), base, 0, deps)[1] == "c1_uncovered"
    assert check_eligibility(item, base, 0, deps, quota_state={"single_source": [1, 1]})[1] == "c2_quota_full"
    assert check_eligibility(item, dict(base, parse_ok=False), 0, deps)[1] == "c3_parse"
    assert check_eligibility(item, dict(base, answer="Something else [1]."), 0, deps)[1] == "c4_self_span"
    deps.donors = {}
    assert check_eligibility(item, base, 0, deps)[1] == "c7_swap"
    deps.set_seed_pool([item], EntitySwapFoilProbe.build_donor_pool(pool, deps.ner, HP["entity_types"]))
    assert check_eligibility(item, base, 0, deps)[1] == "c8_poscontrol"
    _p, deps2, _i, _b = _ready()
    ok, label, payload = check_eligibility(item, base, 0, deps2)
    assert ok and label is None and payload["t"] not in base["cited"]


def test_select_target_uncited_and_three_sentences():
    docs = [{"title": "", "text": "A a. B b. C c."} for _ in range(5)]
    docs[4]["text"] = "A a. B b."
    for seed in range(10):
        t, stratum = select_target([1, 4], docs, item_rng(seed, "x", "coin"), item_rng(seed, "x", "target"))
        assert (stratum == "early" and t == 2) or (stratum == "late" and t is None)


def test_mine_candidates_ranks_6_20_only():
    item = make_item("q1", "Paris")
    cands = MatchedTopicalFoilInjectionProbe(HP).mine_candidate_sentences(item)
    assert {r for _s, r in cands} == set(range(6, 21))
    ctx = " ".join(d["text"] for d in item["context_docs"])
    assert all(s not in ctx for s, _r in cands)


def _foil_item():
    item = make_item("q1", "Paris")
    txt = ("Paris lorem ipsum sit. An answer lorem ipsum. One two three four five six seven eight. "
           "Lorem ipsum dolor sit.")
    item["foil_docs"] = [{"title": "F", "text": txt}] * 15
    return item


def test_match_foil_filters_with_fakes():
    _p, deps, _i, _b = _ready()
    foil, drops = MatchedTopicalFoilInjectionProbe(HP).match_foil(_foil_item(), "Paris is the answer.",
                                                                  "What is the capital q1?", 0, deps)
    assert foil == "Lorem ipsum dolor sit."
    assert drops["alias"] == 1 and drops["overlap"] == 1 and drops["length"] == 1


def test_match_foil_order_invariant():
    _p, deps, _i, _b = _ready()
    probe = MatchedTopicalFoilInjectionProbe(HP)
    args = (_foil_item(), "Paris is the answer.", "What is the capital q1?", 0, deps)
    a, _ = probe.match_foil(*args)
    b, _ = probe.match_foil(*args, filter_order=("nli", "ppl", "cosine", "length", "overlap", "alias"))
    assert a == b


def test_entity_swap_with_fake_ner_nli():
    probe, c = EntitySwapFoilProbe(HP), Counter()
    span = "Paris is the answer."
    s2, meta = probe.swap(span, ["paris"], {"GPE": [("q2", "Berlin")]}, 0, "q1", fake_nli, fake_ner, c)
    assert s2 == "Berlin is the answer." and s2[6:] == span[5:] and meta["donor"] == "q2"
    none, _ = probe.swap(span, ["paris"], {"GPE": [("q2", "Berlin")]}, 0, "q1",
                         lambda p, h: {"contradiction": 0.1}, fake_ner, c)
    assert none is None and c["swap_nli_rejected"] == 1


def test_internal_boundaries_and_inject_at():
    text = "A b. C d. E f."
    bs = RandomBoundarySelfSpanProbe.internal_boundaries(text)
    assert bs == [5, 10]
    out = RandomBoundarySelfSpanProbe.inject_at(text, 5, "PLANT X.")
    assert out == "A b. PLANT X. C d. E f." and out.count("PLANT X.") == 1
    off, obin = RandomBoundarySelfSpanProbe.offset_bin(5, len(text))
    assert 0 < off < 1 and obin == 1


def test_random_foil_shares_foil_and_boundary():
    _p, deps, item, base = _ready()
    ok, _l, payload = check_eligibility(item, base, 0, deps)
    assert ok
    docs_s, meta_s = RandomBoundarySelfSpanProbe(HP).edit_context(item, payload)
    docs_f, meta_f = RandomBoundaryTopicalFoilProbe(HP).edit_context(item, payload)
    t = payload["t"]
    assert payload["foil"] in docs_f[t - 1]["text"] and meta_f["offset"] == meta_s["offset"]
    assert docs_f[t - 1]["text"].index(payload["foil"]) == docs_s[t - 1]["text"].index(payload["span"])


def test_null_arm_returns_input_unchanged():
    item = make_item("q1", "Paris")
    docs = null_edit(item["context_docs"], 2)
    assert docs[1]["text"].rstrip("\n") == item["context_docs"][1]["text"]
    assert [d for i, d in enumerate(docs) if i != 1] == [d for i, d in enumerate(item["context_docs"]) if i != 1]
    base = {"cited": [1], "alias_set": ["paris"]}
    assert score_outcome(base, "Paris is the answer [1].", 2, ["paris"])["move"] is False


def test_positive_control_relocation():
    docs = make_item("q1", "Paris")["context_docs"]
    donor = {"title": "D", "text": "donor text"}
    out = relocate_evidence(docs, 4, 1, donor)
    assert out[3] == docs[0] and out[0] == donor and out[1:3] == docs[1:3] and out[4] == docs[4]


def test_run_item_fake_generate_eight_rows():
    _p, deps, item, base = _ready()
    ok, _l, payload = check_eligibility(item, base, 0, deps)
    rows = run_item(item, base, payload, deps.generate, deps.tf_slots, deps.answer_lp, deps.nli, hp=HP,
                    splitter=deps.splitter, counters=deps.counters)
    assert ok and len(rows) == 8 and {c for c, _s in rows} == set(CONDITION_ORDER)
    assert {r["sample_id"] for r in rows.values()} == {"q1"} and len({r["t"] for r in rows.values()}) == 1


def test_delta_logp_target_and_mismatch():
    c = Counter()
    base = [{"logp": [-3.0, -3.0, 0, 0, 0]}, {"logp": [0, -2.0, 0, 0, 0]}]
    ed = [{"logp": [-1.5, -1.5, 0, 0, 0]}, {"logp": [0, -0.5, 0, 0, 0]}]
    assert delta_logp_target(base, ed, 2, c) == pytest.approx(1.5)
    assert delta_logp_target(base, ed[:1], 2, c) is None and c["slot_mismatch"] == 1


def _lp_table(e, b, eo, bo):
    def lp(q, docs, a):
        blank = docs[0]["text"] == ""
        edited = "plant" in docs[1]["text"]
        return {(True, False): e, (False, False): b, (True, True): eo, (False, True): bo}[(edited, blank)]
    return lp


def test_reliance_classification_thresholds():
    base = [{"title": "A", "text": "orig"}, {"title": "B", "text": "orig"}]
    ctx = [dict(base[0]), {"title": "B", "text": "orig plant"}]
    row = {"move": True, "edited_answer": "x"}
    c = Counter()
    r = reliance_audit(row, _lp_table(-0.7, -1.0, -1.1, -2.0), ctx, base, 1, question="q", hp=HP, counters=c)
    assert r["rel_class"] == "re_sourcing" and r["dp"] == pytest.approx(0.3) and r["red"] == pytest.approx(0.6)
    r = reliance_audit(row, _lp_table(-0.95, -1.0, -1.85, -2.0), ctx, base, 1, question="q", hp=HP, counters=c)
    assert r["rel_class"] == "post_rationalization"
    r = reliance_audit(row, _lp_table(-0.7, -1.0, -1.1, -0.5), ctx, base, 1, question="q", hp=HP, counters=c)
    assert r["rel_class"] == "undefined" and c["loo_base_nonpositive"] == 1
    assert reliance_audit({"move": False}, None, ctx, base, 1, question="q", hp=HP)["rel_class"] is None


def test_alce_proxy_with_fake_nli():
    docs = [{"title": "A", "text": "paris is capital"}, {"title": "B", "text": "berlin city"}]
    out = alce_citation_proxy("Paris is capital [1]. Berlin city [1][2].", docs, fake_nli)
    assert out["alce_recall"] == pytest.approx(1.0) and out["alce_precision"] == pytest.approx(2 / 3)
    assert alce_citation_proxy("", docs, fake_nli) == {"alce_recall": None, "alce_precision": None}
    recall = out["alce_recall"]
    assert recall is not None and math.isfinite(float(recall))