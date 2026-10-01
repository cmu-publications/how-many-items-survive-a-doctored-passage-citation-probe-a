from collections import Counter

import pytest

from analysis import (cluster_bootstrap, compute_contrasts, decision_verdicts, deviance_drop, holm,
                      mcnemar_paired, rank_biserial, seed_validity, seed_wilcoxon, tost_equivalence)
from main import HYPERPARAMETERS
from metrics import (BASELINE, CONDITION_ORDER, NULL_ARM, POSITIVE_ARM, SELF_END, SELF_RANDOM, TOPICAL_END,
                     TOPICAL_RANDOM, itt_rate)
from testing_fakes import row, self_check_rows


def test_mcnemar_toy_and_pairing_by_id():
    rows = self_check_rows()
    out = mcnemar_paired(rows[SELF_END], list(reversed(rows[TOPICAL_END])))
    assert (out["b"], out["c"], out["p"], out["rd"], out["odds_ratio"]) == (1, 1, 1.0, 0.0, 1.0)


def test_holm_known_values():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert out["a"] == pytest.approx(0.03) and out["c"] == pytest.approx(0.06)
    assert out["b"] == pytest.approx(0.06) and out["d"] is None


def test_cluster_bootstrap_deterministic_and_undefined_counter():
    rows = [row("x", s, move=(s in "ab")) for s in "abcd"]
    assert cluster_bootstrap(rows, itt_rate, B=200) == cluster_bootstrap(rows, itt_rate, B=200)
    c = Counter()
    res = cluster_bootstrap(rows, lambda r: None, B=50, counters=c)
    assert c["bootstrap_undefined_resamples"] == 50 and res["ci95"] is None


def test_tost_decision():
    assert tost_equivalence((-0.02, 0.03)) is True
    assert tost_equivalence((-0.06, 0.01)) is False
    assert tost_equivalence(None) is None


def test_rank_biserial_and_wilcoxon_none():
    assert rank_biserial([1.0, -2.0, 3.0]) == pytest.approx(1 / 3)
    assert rank_biserial([]) is None and rank_biserial([0.0, 0.0]) is None
    assert seed_wilcoxon([0.1], [0.2]) is None and seed_wilcoxon([0.1, 0.2], [0.1, 0.2]) is None


def test_deviance_drop_on_toy_frame():
    rows, cell = [], 0
    for cond in (SELF_END, TOPICAL_END, SELF_RANDOM, TOPICAL_RANDOM):
        for slot in (1, 2, 4, 5):
            for rep in range(3):
                r = row(cond, f"i{cell}_{rep}", move=(rep == (0 if slot < 3 else 2)))
                r.update(t=slot, tercile=cell % 3, redundancy="single_source" if cell % 2 else "redundant")
                rows.append(r)
            cell += 1
    out = deviance_drop(rows)
    assert "error" not in out, out
    assert out["slot_drop"] >= -1e-6 and out["span_drop"] >= -1e-6


def test_compute_contrasts_toy_pool():
    pooled = [r for rows in self_check_rows().values() for r in rows]
    out = compute_contrasts(pooled, HYPERPARAMETERS, Counter(), [0.1, 0.5])
    for k in ("P1a", "P1b", "P1c", "P1d", "P2a_self", "P2a_topical", "P2c", "P2e", "F3", "F4",
              "foil_share_ratio", "holm", "redundancy_gap", "delta_logp_rank_biserial", "seed_wilcoxon"):
        assert k in out
    assert out["P1b"]["p"] == 1.0 and out["P1b"]["rd"] == 0.0
    assert out["foil_share_ratio"]["point"] == pytest.approx(1.0)
    assert out["P2e"]["self"]["natural"] == pytest.approx(0.5)


def test_decision_verdicts_rules():
    contrasts = {"P1a": {"p_holm": 0.01, "rd": 0.05, "ratio_ci95": (0.4, 0.9)},
                 "P1b": {"tost_equivalent": False, "rd": 0.04}, "P2a_self": {"p_holm": 0.2, "rd": 0.01},
                 "redundancy_gap": {"point": 0.0, "ci95": None}, "F4": {"near_tie_concentration_self": 0.7}}
    ok = {"broken_null": False, "broken_positive": False, "null_itt": 0.04}
    labels = {v["label"] for v in decision_verdicts(contrasts, ok, HYPERPARAMETERS)}
    assert labels == {"foil_subtracted", "any_perturbation", "near_tie_fragility"}
    broken = decision_verdicts(contrasts, dict(ok, broken_null=True), HYPERPARAMETERS)
    assert [v["label"] for v in broken] == ["broken_null"]


def _cond_metrics(pos):
    m = {c: {"citation_parse_rate": 1.0, "itt_citation_migration_rate": 0.0, "target_cited_rate": 0.0}
         for c in CONDITION_ORDER}
    m[BASELINE]["baseline_determinism_rate"] = 1.0
    m[NULL_ARM]["itt_citation_migration_rate"] = 0.0
    m[POSITIVE_ARM]["target_cited_rate"] = pos
    return m


def test_seed_validity_positive_control_below_030():
    bad = seed_validity(_cond_metrics(0.2), HYPERPARAMETERS)
    assert not bad["valid"] and not bad["success"] and any("positive" in r for r in bad["reasons"])
    good = seed_validity(_cond_metrics(0.6), HYPERPARAMETERS)
    assert good["valid"] and good["success"]