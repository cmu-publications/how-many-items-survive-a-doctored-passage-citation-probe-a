import json
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import baselines
from analysis import INFERENCE_TEST, PRIMARY_CONTRASTS, compute_contrasts, holm, mcnemar_paired
from data import DATA_CONFIG, get_datasets
from main import CONDITION_NAMES, HYPERPARAMETERS, PLAN_ENTITY_TYPES, build_config
from metrics import (BASELINE, NULL_ARM, POSITIVE_ARM, SELF_END, SELF_RANDOM, SWAP_END, TOPICAL_END,
                     aggregate_over_seeds, answer_flip_rate, conditional_rate, determinism_rate, itt_rate,
                     margin_terciles, natural_location_rate, natural_weights, near_tie_concentration, parse_rate,
                     reliance_shares, target_cited_rate)
from probes import (ELIGIBILITY_LABELS, NULL_EDIT_PLAN, POSCONTROL_DONOR_POPULATION, PREPARE_ORDER,
                    BaselineUneditedArm, EntitySwapFoilProbe, NullTrailingWhitespaceProbe,
                    PositiveControlEvidenceRelocatedProbe, build_arms, check_eligibility, delta_logp_target,
                    null_edit, probe_deviations, relocate_evidence)
import seed_loop
from design import should_start_seed
from testing_fakes import FakeHarness, answer_table, fake_ner, make_fake_deps, self_check_rows, toy_pool


def _five_docs():
    return [{"title": f"T{k}", "text": f"Passage {k} first. Passage {k} second."} for k in range(1, 6)]


NULL_TARGET_SENTENCES = ["Alpha one is here.", "Beta two is there.", "Gamma three is near."]
RETENTION_KEYS = frozenset({"null_retention_rate", "null_keep_mask", "null_trailing_chars", "null_dropped_chars"})


def _null_item():
    docs = _five_docs()
    docs[1] = {"title": "T2", "text": " ".join(NULL_TARGET_SENTENCES)}
    return {"sample_id": "q1", "question": "What is the capital?", "context_docs": docs, "aliases": ["paris"],
            "redundancy": "single_source"}


class _WriteRecordingPayload(dict):
    """A payload dict that logs every mutation, so a prepare that writes any key is caught."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.writes: List[str] = []

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes.append(str(key))
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        self.writes.append(f"del:{key}")
        super().__delitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.writes.extend(str(k) for k in dict(*args, **kwargs))
        super().update(*args, **kwargs)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self.writes.append(str(key))
        return super().setdefault(key, default)

    def pop(self, key: Any, *default: Any) -> Any:  # type: ignore[override]
        self.writes.append(f"pop:{key}")
        return super().pop(key, *default)


# ---- plan item 49: the seed-start check warns below 1.1 x measured seed cost, never drops a seed ---
class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


def _run_seed_loop(monkeypatch, budget: float, seed_cost: float, seeds=(0, 1, 2, 3)):
    clock = _Clock()
    monkeypatch.setattr(seed_loop, "time", clock)
    started: List[int] = []

    def fake_run_seed(seed, design, deps, datasets, cfg, harness):
        started.append(int(seed))
        clock.now += seed_cost
        conds = {c: {"itt_citation_migration_rate": 0.1} for c in CONDITION_NAMES}
        return conds, {"n_included": 3}

    monkeypatch.setattr(seed_loop, "run_seed", fake_run_seed)
    harness = FakeHarness()
    run_info: Dict[str, Any] = {"t_start": 0.0, "unrun_seeds": [], "seed_failures": []}
    seed_loop.run_all_seeds(list(seeds), {"est_sec": 1000.0}, None, {}, build_config(HYPERPARAMETERS),
                            harness, run_info, budget)
    return started, harness, run_info


def test_should_start_seed_rule_uses_the_safety_factor():
    factor = float(HYPERPARAMETERS["seed_start_safety_factor"])
    assert factor == pytest.approx(1.1)
    assert should_start_seed(150.0, 100.0, factor)
    assert should_start_seed(111.0, 100.0, factor)
    assert not should_start_seed(109.0, 100.0, factor)
    assert not should_start_seed(50.0, 100.0, factor)


def test_seed_below_safety_factor_is_started_with_a_warning(monkeypatch, capsys):
    """Budget 250s, each seed costs 100s: seeds 0 and 1 pass the check (no measurement yet; 150 >= 110).
    Seeds 2 (50 < 110) and 3 (-50 < 110) fall below it: each is FLAGged and listed in seed_start_warnings,
    but still started and recorded. No seed is listed unrun and no seeds reduction is recorded, because
    only the harness hard stop may end the loop."""
    started, harness, run_info = _run_seed_loop(monkeypatch, 250.0, 100.0)
    assert started == [0, 1, 2, 3]
    assert sorted(int(r["seed"]) for r in harness.seed_records()) == [0, 1, 2, 3]
    assert run_info["unrun_seeds"] == []
    assert not run_info.get("reduced_components")
    assert "time_warnings" not in run_info
    warnings = run_info["seed_start_warnings"]
    assert {w["reason"] for w in warnings} == {seed_loop.SEED_START_WARNING}
    assert [w["seed"] for w in warnings] == [2, 3]
    assert all(w["started"] is True for w in warnings)
    assert warnings[0]["remaining_sec"] == pytest.approx(50.0)
    assert warnings[0]["measured_seed_sec"] == pytest.approx(100.0)
    assert warnings[0]["required_sec"] == pytest.approx(110.0)
    assert warnings[0]["safety_factor"] == pytest.approx(1.1)
    out = capsys.readouterr().out
    assert "FLAG: seed 2: remaining 50s < 1.1 x measured seed cost 100s = 110s; starting it anyway" in out
    assert "recalibrated_from=1000s to=" in out
    assert run_info["recalibrated_to_sec"] == pytest.approx(100.0 + 3 * 100.0 + HYPERPARAMETERS["overhead_sec"])


def test_every_seed_runs_when_budget_suffices(monkeypatch):
    started, harness, run_info = _run_seed_loop(monkeypatch, 10000.0, 100.0)
    assert started == [0, 1, 2, 3]
    assert len(harness.seed_records()) == 4
    assert run_info["unrun_seeds"] == []
    assert not run_info.get("reduced_components")
    assert not run_info.get("seed_start_warnings")


# ---- plan item 41: Holm and the verdicts use the exact McNemar p ---------------------------------
def _mrow(cond: str, seed: int, sid: str, move: bool) -> Dict[str, Any]:
    return {"condition": cond, "seed": seed, "sample_id": sid, "move": move}


def test_mcnemar_p_is_inference_and_item_sign_test_is_sensitivity():
    """Two items repeated under three seeds, x moves and y never does: b = 6, c = 0. McNemar p = 2 * 0.5**6
    feeds inference; the clustered sign test (2 items) gives 0.5 and is only a sensitivity check."""
    xs = [_mrow(SELF_END, s, sid, True) for s in (0, 1, 2) for sid in ("a", "b")]
    ys = [_mrow(TOPICAL_END, s, sid, False) for s in (0, 1, 2) for sid in ("a", "b")]
    mc = mcnemar_paired(xs, ys)
    assert (mc["b"], mc["c"], mc["n"]) == (6, 0, 6)
    assert mc["p"] == pytest.approx(2 * 0.5 ** 6)
    assert mc["p_inference"] == mc["p_row_mcnemar"] == mc["p"]
    assert mc["inference_test"] == INFERENCE_TEST == "mcnemar_exact"
    assert mc["p_item_sign_test"] == pytest.approx(0.5)
    assert mc["n_repeated_items"] == 2


def test_compute_contrasts_holm_runs_on_mcnemar_p():
    rows = self_check_rows()
    pooled = [r for cond_rows in rows.values() for r in cond_rows]
    res = compute_contrasts(pooled, HYPERPARAMETERS, Counter())
    for k in ("P1a", "P1b", "P1c", "P1d", "P2a_self", "P2a_topical", "P2c", "P2e", "F3", "F4"):
        assert k in res
    expected = holm({k: res[k]["p_row_mcnemar"] for k in PRIMARY_CONTRASTS})
    sens = holm({k: res[k]["p_item_sign_test"] for k in PRIMARY_CONTRASTS})
    for k in PRIMARY_CONTRASTS:
        assert res[k]["p_inference"] == res[k]["p_row_mcnemar"]
        assert res[k]["p_holm"] == expected[k]
        assert res[k]["p_holm_item_sign_test"] == sens[k]
    assert res["holm"] == expected
    assert "mcnemar" in res["holm_input"]
    assert set(res["inference_caveat"]["n_repeated_items"]) == set(PRIMARY_CONTRASTS)
    assert res["P1b"]["mcnemar_direction"] in (None, "self_gt_topical", "topical_gt_self", "none")


# ---- plan items 20, 21: control arm condition keys and edits -------------------------------------
def test_control_arm_keys_are_the_plan_names_with_mandatory_prefixes():
    assert NULL_ARM == "null_random_drop_trailing_whitespace_edit"
    assert POSITIVE_ARM == "positive_control_planted_evidence_relocated"
    assert NULL_ARM.startswith("null_random_drop")
    assert POSITIVE_ARM.startswith("positive_control_planted")
    assert NULL_ARM in CONDITION_NAMES and POSITIVE_ARM in CONDITION_NAMES
    arms = build_arms(HYPERPARAMETERS)
    assert isinstance(arms[NULL_ARM], NullTrailingWhitespaceProbe) and arms[NULL_ARM].condition == NULL_ARM
    assert isinstance(arms[POSITIVE_ARM], PositiveControlEvidenceRelocatedProbe)
    assert arms[POSITIVE_ARM].condition == POSITIVE_ARM


def test_null_edit_helper_appends_only_trailing_whitespace():
    docs = _five_docs()
    new = null_edit(docs, 2)
    assert new[1]["text"] == docs[1]["text"] + "\n"
    assert new[1]["text"].rstrip("\n") == docs[1]["text"] and new[1]["title"] == docs[1]["title"]
    assert all(new[k] == docs[k] for k in (0, 2, 3, 4))
    assert docs == _five_docs()


def test_null_arm_returns_input_unchanged():
    """Plan item 20 through the arm itself: slot t is the original text + '\\n', stripping it gives the
    original, and every other slot is identical."""
    item = _null_item()
    arm = build_arms(HYPERPARAMETERS)[NULL_ARM]
    for t in range(1, 6):
        payload: Dict[str, Any] = {"t": t, "o": 1}
        assert arm.prepare(item, {}, payload, 0, None) is None
        edited, meta = arm.edit_context(item, payload)
        orig = item["context_docs"]
        assert edited[t - 1]["text"] == orig[t - 1]["text"] + "\n"
        assert edited[t - 1]["text"].strip() == orig[t - 1]["text"].strip()
        assert edited[t - 1]["title"] == orig[t - 1]["title"]
        assert all(edited[k] == orig[k] for k in range(5) if k != t - 1)
        assert meta["changed_slots"] == [t] and meta["plant"] == "\n"
        assert edited == null_edit(orig, t)
    assert item == _null_item()


def test_null_arm_prepare_never_writes_to_the_payload():
    """Regression for the contract breach: prepare must not write retention fields (or anything else) into
    the shared eligibility payload. A write-recording payload fails on any mutation, for every slot and
    whether or not slot t carries trailing whitespace."""
    arm = NullTrailingWhitespaceProbe(HYPERPARAMETERS)
    for tail in ("", "  ", "\t\n"):
        item = _null_item()
        for doc in item["context_docs"]:
            doc["text"] = doc["text"] + tail
        for t in range(1, 6):
            payload = _WriteRecordingPayload({"t": t, "o": 1, "span": "Paris is the capital."})
            snapshot = dict(payload)
            assert arm.prepare(item, {}, payload, 3, None) is None
            assert payload.writes == [], f"null prepare wrote {payload.writes} (tail={tail!r}, t={t})"
            assert dict(payload) == snapshot
            assert not RETENTION_KEYS & set(payload)


def test_null_arm_keeps_trailing_whitespace_and_records_no_retention_rate():
    """Slot t with trailing whitespace: the arm returns text + '\\n' verbatim (the former random-drop edit
    stripped the whitespace) and neither payload nor row carries a retention rate or keep mask."""
    item = _null_item()
    for k, tail in enumerate(("  ", "\t", " \n ", "\n", "   ")):
        item["context_docs"][k]["text"] = item["context_docs"][k]["text"] + tail
    arm = build_arms(HYPERPARAMETERS)[NULL_ARM]
    baseline = {"answer": "Paris is the capital [1].", "cited": [1], "parse_ok": True, "alias_set": ["paris"],
                "citation_slots": [], "margin": None}
    for t in range(1, 6):
        payload: Dict[str, Any] = {"t": t, "o": 1, "span": "Paris is the capital."}
        before = dict(payload)
        assert arm.prepare(item, baseline, payload, 3, None) is None
        assert payload == before
        edited, _meta = arm.edit_context(item, payload)
        assert edited[t - 1]["text"] == item["context_docs"][t - 1]["text"] + "\n"
        assert edited == null_edit(item["context_docs"], t)
        row = arm.run(item, baseline, payload, lambda q, d: baseline["answer"], lambda q, d, a: [], Counter())
        assert row["null_edit"] == NULL_EDIT_PLAN
        assert not RETENTION_KEYS & set(row)


def test_null_arm_edit_is_seed_independent_and_drops_nothing():
    """Behavioural replacement for the old attribute check: a seeded random-drop edit would give
    seed-dependent contexts or lose target sentences. Under every seed the null arm's context is the same
    null_edit output, every target sentence survives, and the row carries no retention field."""
    item = _null_item()
    arm = build_arms(HYPERPARAMETERS)[NULL_ARM]
    baseline = {"answer": "Paris is the capital [1].", "cited": [1], "parse_ok": True, "alias_set": ["paris"],
                "citation_slots": [], "margin": None}
    expected = null_edit(item["context_docs"], 2)
    for seed in (0, 1, 2, 3, 7, 11):
        payload: Dict[str, Any] = {"t": 2, "o": 1, "span": "Paris is the capital."}
        assert arm.prepare(item, baseline, payload, seed, None) is None
        edited, _meta = arm.edit_context(item, payload)
        assert edited == expected
        assert all(s in edited[1]["text"] for s in NULL_TARGET_SENTENCES)
        row = arm.run(item, baseline, payload, lambda q, d: baseline["answer"], lambda q, d, a: [], Counter())
        assert row["_docs"] == expected
        assert not RETENTION_KEYS & set(row)
    assert item == _null_item()


def test_null_arm_prepare_adds_no_eligibility_exclusion():
    """The null arm never returns an eligibility label and writes nothing to the payload, even without a
    treatment span."""
    arm = build_arms(HYPERPARAMETERS)[NULL_ARM]
    for payload in ({"t": 2}, {"t": 2, "span": "Paris is the capital."}):
        before = dict(payload)
        assert arm.prepare(_null_item(), {}, payload, 0, None) is None
        assert payload == before
    one_sentence = _null_item()
    one_sentence["context_docs"][1]["text"] = "Only one sentence here."
    assert arm.prepare(one_sentence, {}, {"t": 2, "span": "Paris is it."}, 0, None) is None
    with pytest.raises(ValueError):
        arm.prepare(_null_item(), {}, {}, 0, None)


def test_null_arm_excludes_no_item():
    """The eligible set is identical with the null arm's prepare replaced by a no-op."""
    pool = toy_pool()

    def eligible_ids(arms):
        deps = make_fake_deps(HYPERPARAMETERS, answer_table(pool))
        deps.set_seed_pool(pool, EntitySwapFoilProbe.build_donor_pool(pool, fake_ner, PLAN_ENTITY_TYPES))
        out = []
        for it in pool:
            base = BaselineUneditedArm.generate_baseline(it, deps.generate, deps.tf_slots, 5)
            ok, _label, _p = check_eligibility(it, base, 0, deps, arms=arms)
            if ok:
                out.append(str(it["sample_id"]))
        return out

    with_null = eligible_ids(build_arms(HYPERPARAMETERS))
    noop = build_arms(HYPERPARAMETERS)
    noop[NULL_ARM].prepare = lambda *a, **k: None  # type: ignore[method-assign]
    assert with_null and with_null == eligible_ids(noop)


def test_null_arm_identical_answer_scores_zero_move():
    item = _null_item()
    baseline = BaselineUneditedArm.generate_baseline(item, lambda q, d: "Paris is the capital [1].",
                                                     lambda q, d, a: [], 5)
    arm = build_arms(HYPERPARAMETERS)[NULL_ARM]
    counters: Counter = Counter()
    row = arm.run(item, baseline, {"t": 2, "o": 1}, lambda q, d: baseline["answer"], lambda q, d, a: [], counters)
    assert not row["move"]
    assert row["null_edit"] == NULL_EDIT_PLAN and row["plant"] == "\n"
    assert row["_docs"][1]["text"] == item["context_docs"][1]["text"] + "\n"


# ---- plan item 13: eligibility reaches every arm, first failure in label order --------------------
def test_check_eligibility_reaches_every_arm_on_the_toy_pool():
    pool = toy_pool()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(pool))
    deps.set_seed_pool(pool, EntitySwapFoilProbe.build_donor_pool(pool, fake_ner, PLAN_ENTITY_TYPES))
    arms = build_arms(HYPERPARAMETERS)
    eligible = []
    for it in pool:
        base = BaselineUneditedArm.generate_baseline(it, deps.generate, deps.tf_slots, 5)
        ok, label, payload = check_eligibility(it, base, 0, deps, arms=arms)
        if ok:
            assert label is None
            eligible.append(payload)
        else:
            assert label in ELIGIBILITY_LABELS
    assert eligible, "no toy item became eligible"
    for p in eligible:
        assert not any(k.startswith("null_") for k in p)
        assert "donor_passage" in p and p["donor_population"] == POSCONTROL_DONOR_POPULATION


def test_eligibility_reports_null_failure_before_foil_failure(monkeypatch):
    assert PREPARE_ORDER.index(SELF_END) < PREPARE_ORDER.index(NULL_ARM) < PREPARE_ORDER.index(TOPICAL_END)
    pool = toy_pool()
    deps = make_fake_deps(HYPERPARAMETERS, answer_table(pool))
    deps.set_seed_pool(pool, {})
    arms = build_arms(HYPERPARAMETERS)
    monkeypatch.setattr(arms[NULL_ARM], "prepare", lambda *a, **k: "c5_target")
    monkeypatch.setattr(arms[TOPICAL_END], "prepare", lambda *a, **k: "c6_foil")
    item = next(it for it in pool if it.get("covered"))
    base = BaselineUneditedArm.generate_baseline(item, deps.generate, deps.tf_slots, 5)
    ok, label, _p = check_eligibility(item, base, 0, deps, arms=arms)
    assert not ok and label in ("c4_self_span", "c5_target")


def test_positive_control_records_donor_population_deviation():
    dev = probe_deviations()
    assert dev and dev[0]["component"] == POSITIVE_ARM and dev[0]["used"] == POSCONTROL_DONOR_POPULATION
    arm = build_arms(HYPERPARAMETERS)[POSITIVE_ARM]
    item = {"sample_id": "q1", "redundancy": "single_source", "context_docs": _five_docs()}
    payload = {"t": 4, "o": 1, "donor_id": "q9", "donor_population": POSCONTROL_DONOR_POPULATION}
    meta_row = arm._row_meta(item, {"margin": 0.1}, payload, {"plant": "relocated_evidence"}, "x")
    assert meta_row["donor_population"] == POSCONTROL_DONOR_POPULATION and meta_row["donor_id"] == "q9"


def test_positive_control_planted_evidence_relocated_swaps_only_t_and_o():
    docs = _five_docs()
    donor = {"title": "Donor", "text": "Unrelated donor passage."}
    new = relocate_evidence(docs, 4, 1, donor)
    assert new[3] == docs[0] and new[0] == donor
    assert all(new[k] == docs[k] for k in (1, 2, 4))
    arm = build_arms(HYPERPARAMETERS)[POSITIVE_ARM]
    edited_docs, meta = arm.edit_context({"context_docs": docs}, {"t": 4, "o": 1, "donor_passage": donor})
    assert edited_docs == new and meta["changed_slots"] == [1, 4]


def test_self_check_toy_pool():
    rows = self_check_rows()
    s = rows[SELF_END]
    assert itt_rate(s) == pytest.approx(1 / 3) and target_cited_rate(s) == pytest.approx(2 / 3)
    assert answer_flip_rate(s) == pytest.approx(1 / 3) and conditional_rate(s) == pytest.approx(0.5)
    assert parse_rate(rows[TOPICAL_END]) == pytest.approx(2 / 3)
    assert conditional_rate(rows[SWAP_END]) is None
    assert itt_rate(rows[NULL_ARM]) == 0 and target_cited_rate(rows[NULL_ARM]) == 0
    assert answer_flip_rate(rows[NULL_ARM]) == 0
    assert target_cited_rate(rows[POSITIVE_ARM]) == pytest.approx(1.0)
    mc = mcnemar_paired(s, rows[TOPICAL_END])
    assert (mc["p"], mc["rd"], mc["odds_ratio"]) == (1.0, 0.0, 1.0)
    assert mc["p_inference"] == mc["p"]
    assert itt_rate(rows[TOPICAL_END]) / itt_rate(s) == pytest.approx(1.0)
    assert near_tie_concentration(s, margin_terciles(rows[BASELINE])) == pytest.approx(1.0)
    assert reliance_shares(s)["re_sourcing_share"] == pytest.approx(1.0)
    base = [{"logp": [-3.0, -3.0, 0, 0, 0]}]
    assert delta_logp_target(base, [{"logp": [-1.5, -1.5, 0, 0, 0]}], 2) == pytest.approx(1.5)
    assert natural_location_rate(rows[SELF_RANDOM], natural_weights([0.1, 0.5])) == pytest.approx(0.5)
    assert determinism_rate([("a", "a"), ("b", "b"), ("c", "d")]) == pytest.approx(2 / 3)
    agg = aggregate_over_seeds({0: {SELF_END: {"k": 1 / 3}}, 1: {SELF_END: {"k": 0.0}}}, "k")[SELF_END]
    assert agg["mean"] == pytest.approx(1 / 6) and agg["std"] == pytest.approx(0.2357, abs=1e-4)


# ---- plan item 17: entity-swap foil donors ------------------------------------------------------
def test_entity_swap_tries_up_to_ten_donors():
    donors = {"GPE": [(f"q{i}", "Berlin") for i in range(2, 14)]}
    c: Counter = Counter()
    probe = EntitySwapFoilProbe(HYPERPARAMETERS)
    out, meta = probe.swap("Paris is it.", ["paris"], donors, 0, "q1", lambda p, h: {"contradiction": 0.0},
                           fake_ner, c)
    assert out is None and meta["reason"] == "nli_rejected_all" and c["swap_nli_rejected"] == 10
    out, meta = probe.swap("Paris is it.", ["paris"], donors, 0, "q1", lambda p, h: {"contradiction": 0.9},
                           fake_ner, Counter())
    start, end = meta["entity_span"]
    assert out == "Berlin" + "Paris is it."[end:] and start == 0


def _first_word_ner(text):
    word = text.split()[0].rstrip(".,")
    return [(0, len(word), "GPE", word)]


def test_donor_pool_draws_from_every_other_pool_item():
    """Plan item 17: donors come from every pool item, not only covered ones; the current item is excluded
    at swap time and the accepted swap differs from the span only inside the entity range."""
    pool = [{"sample_id": "a", "covered": False, "aliases": ["berlin"], "gold_long_answer": "Berlin is big."},
            {"sample_id": "b", "covered": True, "aliases": ["rome"], "gold_long_answer": "Rome is old."},
            {"sample_id": "c", "covered": True, "aliases": ["paris"], "gold_long_answer": "Paris is it."},
            {"sample_id": "d", "covered": False, "aliases": ["x"], "gold_long_answer": "Madrid is far."}]
    donors = EntitySwapFoilProbe.build_donor_pool(pool, _first_word_ner, PLAN_ENTITY_TYPES)
    assert donors["GPE"] == [("a", "Berlin"), ("b", "Rome"), ("c", "Paris")]
    probe = EntitySwapFoilProbe(HYPERPARAMETERS)
    c: Counter = Counter()
    out, meta = probe.swap("Paris is it.", ["paris"], donors, 0, "c", lambda p, h: {"contradiction": 0.9},
                           _first_word_ner, c)
    assert meta["donor"] in {"a", "b"} and out in {"Berlin is it.", "Rome is it."}
    start, end = meta["entity_span"]
    assert out[:start] == "Paris is it."[:start] and out.endswith("Paris is it."[end:])
    assert c["swap_nli_rejected"] == 0


# ---- plan item 7: default alias offsets use the lazy spaCy splitter, never the regex fallback ----
# The regex fallback splits after "Dr." (alias "smith" would sit at 4/34); spaCy keeps
# "Dr. Smith went home." as one sentence (alias at 0/34).
ABBREV_TEXT = "Dr. Smith went home. Paris is big."
_FAKE_SPACY_BOUNDARY = re.compile(r"(?<!\bDr)\.\s+")


class _FakeSentNlp:
    """Stand-in for a spaCy pipeline: doc.sents spans that do not split after the 'Dr.' abbreviation."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, text: str) -> Any:
        self.calls.append(text)
        sents, start = [], 0
        for m in _FAKE_SPACY_BOUNDARY.finditer(text):
            sents.append(SimpleNamespace(start_char=start, end_char=m.start() + 1))
            start = m.end()
        if start < len(text):
            sents.append(SimpleNamespace(start_char=start, end_char=len(text)))
        return SimpleNamespace(sents=sents)


def _write_alce_fixture(root) -> None:
    alce = root / "alce"
    alce.mkdir()
    docs = [{"title": f"D{k}", "text": ABBREV_TEXT} for k in range(20)]
    items = [{"sample_id": "s1", "question": "Who went home?", "answer": "Smith.",
              "qa_pairs": [{"short_answers": ["Smith"]}], "docs": docs}]
    (alce / DATA_CONFIG["eval_file"]).write_text(json.dumps(items), encoding="utf-8")
    (alce / DATA_CONFIG["demo_file"]).write_text(json.dumps({"demos": [], "instruction": "I"}), encoding="utf-8")


def test_regex_fallback_splits_the_abbreviation_fixture():
    assert baselines.sentence_spans(ABBREV_TEXT) == [(0, 3), (4, 20), (21, 34)]


def test_default_alias_offsets_use_the_lazy_spacy_splitter(tmp_path, monkeypatch):
    """With no splitter supplied, get_datasets must obtain spans through baselines.get_sent_nlp; the old
    regex fallback gives offset 4/34 and never calls the spaCy loader, so this test fails on it."""
    fake = _FakeSentNlp()
    requested: List[str] = []

    def fake_get_sent_nlp(model_name: str = baselines.SPACY_SENT_MODEL) -> Any:
        requested.append(model_name)
        return fake

    monkeypatch.setattr(baselines, "get_sent_nlp", fake_get_sent_nlp)
    _write_alce_fixture(tmp_path)
    ds = get_datasets(str(tmp_path))
    items = ds["val"].items + ds["test"].items
    assert len(items) == 1
    assert items[0]["alias_sentence_offsets"] == [0.0] * 5
    assert fake.calls and set(fake.calls) == {ABBREV_TEXT}
    assert set(requested) == {baselines.SPACY_SENT_MODEL}
    recorded = DATA_CONFIG["alias_offset_splitter"]
    assert "spacy_sentence_spans" in recorded and baselines.SPACY_SENT_MODEL in recorded


def test_spacy_sentence_spans_goes_through_the_loader(monkeypatch):
    fake = _FakeSentNlp()
    monkeypatch.setattr(baselines, "get_sent_nlp", lambda model_name=baselines.SPACY_SENT_MODEL: fake)
    assert baselines.spacy_sentence_spans(ABBREV_TEXT) == [(0, 20), (21, 34)]
    assert baselines.spacy_sentence_spans("   ") == []
    assert fake.calls == [ABBREV_TEXT]