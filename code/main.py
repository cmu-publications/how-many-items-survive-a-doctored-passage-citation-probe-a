"""
Experiment entry point: doctored-passage RAG citation probe (inference-only).

Reader Qwen/Qwen2.5-3B-Instruct on ASQA (ALCE format). Eight paired conditions per eligible item; every
edited arm is compared with the unedited baseline on the same (seed, sample_id). Each plan seed scans the
whole development pool in its own seeded order (confirmation seeds: the confirmation pool); inference is
clustered by sample_id. Go/no-go preconditions (model_load first) run at full size and exit 1 on FAIL.
PRIMARY METRIC: itt_citation_migration_rate (minimize); primary condition topical_foil_end_injection;
baseline condition self_span_end_injection. primary_metric / baseline_metric / aggregates are means over
every record in harness.seed_records(); contrasts use the headline seeds (valid, with item rows), falling
back to a flagged descriptive run over every recorded seed's rows, then over the rows preserved from seeds
that were not recorded (failed or budget-stopped); broken_null / broken_positive are decided run-level over
every recorded seed. A harness record without its item rows is completed from the process copy of the
payload (row_sources; FLAG line and results.json item_rows_restored_from_process).
A missing ALCE prompt file is a recorded zero-shot deviation
(FLAG line; results.json deviations.demonstrations, reduced_components and the precondition detail).
Nothing is trained; setup.py downloads, this script runs offline. Results go only through
experiment_harness, and only after the seed loop returns normally. This script never prints a PRIMARY line.
RC_SMOKE_TEST=1 turns on smoke mode; RC_TIME_BUDGET_SEC sets the budget. Smoke mode applies the recorded
SMOKE_OVERRIDES and the smoke citation reminder (each a recorded smoke-only protocol deviation) and runs at
most SMOKE_MAX_RUN_SEEDS seeds (FLAG lines; results.json unrun seeds and a reduced component); a full run
never applies them. Every eligibility decision is tallied by outcome and printed (ELIGIBILITY_TALLY lines)
together with the first baseline answers seen per outcome (ELIGIBILITY_SAMPLE lines). Every non-zero stop
before the seed loop prints its reason on stdout and stderr.

Usage:  python main.py [--smoke] [--data-root ./data]
"""
import argparse
import copy
import functools
import importlib
import math
import os
import sys
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Must be set before torch (imported via data) is loaded.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_flag] = "1"

import numpy as np  # noqa: E402

import data as data_module  # noqa: E402
import pilot as pilot_module  # noqa: E402
import probes as probes_module  # noqa: E402
import seed_loop as seed_loop_module  # noqa: E402
from analysis import compute_contrasts, decision_verdicts, seed_validity, success_rate  # noqa: E402
from data import get_datasets  # noqa: E402
from design import (ANALYSIS_COMPONENTS, CONDITION_NAMES, PILOT_CACHE, REDUNDANCY_STRATA, _hp,  # noqa: E402
                    check_pools, choose_n_items, finalize_design, parse_rc_seeds, pilot_and_design,
                    pool_sufficiency, quota_for, should_start_seed, sizing_basis, smoke_config)
from hp_schema import HP_VALIDATORS, validate_hyperparameters  # noqa: E402
from metrics import (BASELINE, NULL_ARM, POSITIVE_ARM, SELF_END, TOPICAL_END,  # noqa: E402
                     aggregate_over_seeds, new_counters)
from pilot import plan_run_seeds  # noqa: E402
from preconditions import (Check, data_summary, nli_sanity_correct, precondition_checks,  # noqa: E402
                           reader_numerics_check)
from probes import Deps, probe_deviations  # noqa: E402
from real_deps import backend_info, build_real_deps, spacy_ner_labels  # noqa: E402
from row_sources import install_seed_extra_capture, preserved_rows, restore_extra, source_summary  # noqa: E402
from runtime_io import finalize_results  # noqa: E402
from seed_loop import (HarnessAborted, SeedFailed, SeedStopped, determinism_pairs,  # noqa: E402
                       record_seed_payload, run_all_seeds, run_seed)

__all__ = ["HYPERPARAMETERS", "PREREGISTERED", "PLAN_ENTITY_TYPES", "CONDITION_NAMES", "ANALYSIS_COMPONENTS",
           "REDUNDANCY_STRATA", "PILOT_CACHE", "SeedStopped", "SeedFailed", "HarnessAborted", "build_config",
           "protocol_deviations", "format_precondition", "run_preconditions", "load_deps_or_fail",
           "MODEL_LOAD_PRECONDITION", "quota_for", "pool_sufficiency",
           "choose_n_items", "should_start_seed", "smoke_config", "sizing_basis", "parse_rc_seeds",
           "record_seed_payload", "determinism_pairs", "control_checks", "item_overlap", "run_seed",
           "run_all_seeds", "print_metric_lines", "aggregate_and_write", "run_and_write", "build_real_deps",
           "backend_info", "pilot_and_design", "finalize_design", "check_pools", "main",
           "CONTRAST_COMPONENTS", "CONTRASTS_BASIS_HEADLINE", "CONTRASTS_BASIS_ALL", "CONTRASTS_BASIS_NONE",
           "CONTRASTS_BASIS_PRESERVED",
           "headline_descriptive_controls", "HARNESS_MODULE", "HARNESS_IMPORT_ERROR",
           "SMOKE_OVERRIDES", "smoke_hyperparameters", "SMOKE_MAX_RUN_SEEDS", "SMOKE_SEED_REASON",
           "smoke_seed_subset", "report_stop", "precondition_hyperparameters", "ZERO_SHOT_PRECONDITION_NOTE",
           "ELIGIBILITY_TALLY", "ELIGIBILITY_SAMPLES", "eligibility_label", "baseline_answer_of",
           "install_eligibility_tally", "print_eligibility_tally", "SMOKE_CITATION_REMINDER",
           "QUESTION_SUFFIX_METHODS", "install_question_suffix"]

HARNESS_MODULE = "experiment_harness"


def _harness_unavailable(reason: str) -> SimpleNamespace:
    """Stand-in for a missing experiment_harness: every entry point raises RuntimeError naming the missing
    module, so a run without the harness stops non-zero and writes nothing."""
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{HARNESS_MODULE} is not importable ({reason}); it is supplied by the run sandbox "
                           f"and seeds cannot be recorded or results written without it")

    return SimpleNamespace(get_harness=fail, seeds=fail, available=False, reason=reason)


def _load_harness() -> Tuple[Any, Optional[str]]:
    """Imports experiment_harness. Only the module's own absence is tolerated (recorded and printed); an
    import error raised inside the harness propagates unchanged."""
    try:
        return importlib.import_module(HARNESS_MODULE), None
    except ModuleNotFoundError as exc:
        if exc.name != HARNESS_MODULE:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        print(f"FLAG: {HARNESS_MODULE} not importable ({reason}); get_harness/seeds will raise if called",
              flush=True)
        return _harness_unavailable(reason), reason


experiment_harness, HARNESS_IMPORT_ERROR = _load_harness()

DATA_CONFIG = data_module.DATA_CONFIG

PLAN_ENTITY_TYPES = ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "WORK_OF_ART", "EVENT", "NORP", "LOC"]

HYPERPARAMETERS = {
    # models and decoding
    "reader_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "embed_model_id": "BAAI/bge-small-en-v1.5",
    "nli_model_id": "cross-encoder/nli-deberta-v3-base",
    "spacy_model": "en_core_web_sm",
    "decoding": "greedy",
    "batch_size": 1,
    # design
    "seeds": [0, 1, 2, 3, 4],
    "n_passages": 5,
    "n_demos": 2,
    "max_new_tokens": 300,
    "separator": " ",
    "slot_strata": {"early": [1, 2], "late": [4, 5]},
    "min_target_sentences": 3,
    # topical foil matching (pre-registered)
    "foil_mining_ranks": [6, 20],
    "foil_length_tol": 0.1,
    "foil_cosine_tol": 0.05,
    "foil_ppl_tol_nats": 0.3,
    "topical_nli_contradiction_max": 0.5,
    # entity swap (pre-registered type list)
    "entity_types": list(PLAN_ENTITY_TYPES),
    "swap_max_tries": 10,
    "swap_nli_contradiction_min": 0.5,
    # random-position arms
    "offset_bins": 5,
    # reliance audit
    "reliance_delta_plant_nats_per_token": 0.1,
    "reliance_orig_drop_resourcing": 0.5,
    "reliance_orig_drop_postrat": 0.2,
    # ALCE citation proxy / NLI
    "alce_entail_threshold": 0.5,
    "nli_max_length": 512,
    # statistics
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20260928,
    "effect_threshold": 0.03,
    "tost_margin": 0.05,
    "alpha": 0.05,
    "ratio_ci_lower_min": 0.3,
    "near_tie_share_threshold": 0.6,
    # validity gates and controls
    "null_expected_max": 0.03,
    "null_broken_above": 0.05,
    "pos_expected_min": 0.5,
    "pos_broken_below": 0.30,
    "parse_rate_gate": 0.9,
    "determinism_gate": 1.0,
    # preconditions (never shrunk in smoke)
    "expected_eval_items": 948,
    "manifest_glob": "*manifest*.json",
    "nli_sanity_min_correct": 7,
    # budget: 50 eligible items per seed (25 single_source + 25 redundant); never fewer than 20
    "n_max_items": 50,
    "n_min_items": 20,
    "det_items": 20,
    "eligible_rate_prior": 0.3,
    "time_budget_sec": 43200,
    "overhead_sec": 600,
    "seed_start_safety_factor": 1.1,
    "pilot_scan_items": 10,
    "max_item_exceptions": 10,
    # smoke mode: n = min(5, pilot-fitted), never fewer than 2 (smoke override: 1, recorded)
    "smoke_n_items": 5,
    "smoke_min_items": 2,
    "smoke_scan_cap": 40,
}

# Values locked by the experiment plan. Any difference from HYPERPARAMETERS is written to
# results.json as a protocol deviation; nothing is changed silently.
PREREGISTERED = {
    "reader_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "nli_model_id": "cross-encoder/nli-deberta-v3-base",
    "decoding": "greedy",
    "batch_size": 1,
    "seeds": [0, 1, 2, 3, 4],
    "foil_mining_ranks": [6, 20],
    "foil_length_tol": 0.1,
    "foil_cosine_tol": 0.05,
    "foil_ppl_tol_nats": 0.3,
    "topical_nli_contradiction_max": 0.5,
    "swap_max_tries": 10,
    "swap_nli_contradiction_min": 0.5,
    "entity_types": list(PLAN_ENTITY_TYPES),
    "offset_bins": 5,
    "reliance_delta_plant_nats_per_token": 0.1,
    "reliance_orig_drop_resourcing": 0.5,
    "reliance_orig_drop_postrat": 0.2,
    "bootstrap_resamples": 2000,
    "effect_threshold": 0.03,
    "tost_margin": 0.05,
    "ratio_ci_lower_min": 0.3,
    "near_tie_share_threshold": 0.6,
    "null_expected_max": 0.03,
    "null_broken_above": 0.05,
    "pos_expected_min": 0.5,
    "pos_broken_below": 0.30,
    "parse_rate_gate": 0.9,
    "determinism_gate": 1.0,
    "n_max_items": 50,
    "n_min_items": 20,
    "det_items": 20,
    "smoke_n_items": 5,
    "smoke_min_items": 2,
    "smoke_scan_cap": 40,
    "pilot_scan_items": 10,
    "max_new_tokens": 300,
    "n_passages": 5,
    "n_demos": 2,
}

# Smoke-only settings (RC_SMOKE_TEST=1 or --smoke). A smoke run checks the pipeline end to end; it is never a
# scientific result. History of smoke failures:
#  1. The pre-registered foil tolerances admitted no eligible item in the 40-item scan (relaxed below).
#  2. 3B reader: load + preconditions + pilot took 1226 s of 1500 s; 2 items needed 965 s (1.5B reader below).
#  3. 1.5B reader: the answers dropped the [n] markers (c3_parse 52 of 69), so the smoke run appends
#     SMOKE_CITATION_REMINDER to the question of every reader call (all arms alike; install_question_suffix).
#  4. With the reminder the pilot found 1 eligible item; n_min=2 needed 743 s of 683 s left (smoke min: 1).
#  5. With the no-document-list reminder the pilot found none in 40 items (c1_uncovered 11, c4_self_span 28,
#     c5_target 1). The reminder now asks for at most three short sentences, the value copied from the
#     documents and one supporting document per sentence; max_new_tokens 200 and scan cap 80 below.
# Each override is printed as a FLAG line and written to results.json (smoke_overrides, reduced_components,
# flags; the pre-registered ones also appear under deviations.hyperparameters). A full run never applies them.
SMOKE_OVERRIDES: Dict[str, Any] = {
    "reader_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "overhead_sec": 60,
    "foil_length_tol": 0.3,
    "foil_cosine_tol": 0.15,
    "foil_ppl_tol_nats": 1.0,
    "min_target_sentences": 2,
    "smoke_min_items": 1,
    "max_new_tokens": 200,
    "smoke_scan_cap": 80,
}
_SMOKE_FOIL_REASON = ("smoke eligibility: the pre-registered matched-foil tolerances admitted no eligible item "
                      "in the 40-item smoke scan; relaxed for the pipeline check only")
SMOKE_OVERRIDE_REASONS: Dict[str, str] = {
    "reader_model_id": "smoke generation cost: with the 3B reader the pilot left 274 s while 2 items needed "
                       "965 s; the 1.5B model of the same family (cached locally, same tokenizer and chat "
                       "template) roughly halves every generation; the full run uses the pre-registered 3B "
                       "reader",
    "overhead_sec": "smoke aggregation covers a few items on one seed (measured well under a minute); the "
                    "full-run 600 s reserve would exhaust the smoke budget before the minimum item count fits",
    "foil_length_tol": _SMOKE_FOIL_REASON,
    "foil_cosine_tol": _SMOKE_FOIL_REASON,
    "foil_ppl_tol_nats": _SMOKE_FOIL_REASON,
    "min_target_sentences": "smoke eligibility: a shorter target passage minimum lets the 40-item smoke scan "
                            "find an eligible item; relaxed for the pipeline check only",
    "smoke_min_items": "smoke budget: the pilot fitted n_fit=1 item/seed and 2 items needed 743 s of the 683 s "
                       "left; one item per seed still runs every arm, metric and analysis component end to "
                       "end; the full run keeps n_min_items=20",
    "max_new_tokens": "smoke generation cost: the smoke reminder asks for at most three short sentences "
                      "(well under 200 tokens); the lower cap bounds the 17.6 s/answer measured at 300 tokens "
                      "for every arm alike; the full run keeps 300",
    "smoke_scan_cap": "smoke eligibility: the 40-item scan found no eligible item (c4_self_span 28, c5_target 1, "
                      "c1_uncovered 11); 80 items give the pilot more candidates within the smoke budget; the "
                      "pilot still stops at the first eligible item",
}
# Smoke-only prompt addition: appended to the question of every reader call (all arms identically).
SMOKE_CITATION_REMINDER = (" Answer in at most three short sentences, copying the exact name, date or number "
                           "that answers the question from the documents. End every sentence with the number "
                           "of the one document that best supports it in square brackets, for example [2]. "
                           "Do not list the documents or their titles after your answer.")
SMOKE_CITATION_REASON = ("smoke eligibility: with the 1.5B reader 52 of 69 eligibility decisions failed the "
                         "citation parse (answers without any [n] marker), later 27 of 69 failed the self-span "
                         "check on answers ending in a document list, and then 28 of 40 scanned items failed it on "
                         "long answers stating an unsupported value; the reminder (short answer, value copied "
                         "from the documents, one cited document per sentence) is appended to the question for "
                         "generation and both teacher-forced passes of every arm alike; the full run uses the "
                         "unmodified ALCE prompt")
# Reader-facing Deps methods whose first argument is the question; all get the same suffix.
QUESTION_SUFFIX_METHODS = ("generate", "tf_slots", "answer_lp")
# Smoke-only replication cut: one seed; the seeds not run are recorded as unrun seeds and a reduced component.
SMOKE_MAX_RUN_SEEDS = 1
SMOKE_SEED_REASON = "smoke_replication_reduced"

# Detail appended to the data_and_models precondition when the ALCE prompt file (the source of the
# in-context demonstrations) is absent: the run proceeds zero-shot as a recorded deviation.
ZERO_SHOT_PRECONDITION_NOTE = ("zero-shot: ALCE demonstrations unavailable; demonstration minimum set to 0 for "
                               "this check only (recorded deviation: deviations.demonstrations)")

EXCLUDED_SELECTIONS = {
    "CIFAR-10": "image data; not relevant to citation behaviour",
    "LSTM LM": "architecture, not a citation-measurement method",
    "DANN": "architecture, not a citation-measurement method",
}
CONTROL_ARM_MAPPING = {"null": NULL_ARM, "positive_control": POSITIVE_ARM, "treatment": TOPICAL_END,
                       "published_probe": SELF_END, "paired_reference": BASELINE}
AGGREGATE_SEED_SET = ("every record in harness.seed_records(); used for aggregates, primary_metric, "
                      "baseline_metric and the printed per-seed and mean lines (same basis as the harness "
                      "PRIMARY line)")
HEADLINE_SEED_SET = ("seeds passing seed_validity (arms present, parse rate, determinism, defined null ITT, "
                     "positive control not broken) with item rows; used for aggregates_headline_seeds, "
                     "*_headline_seeds metrics, control_checks_headline_descriptive, contrasts and the novelty "
                     "precondition. The null ITT never excludes a seed")
NULL_CONTROL_BASIS = ("run-level: mean null ITT and mean positive-control target_cited rate over every recorded "
                      "seed, excluded seeds included (controls, control_checks, control_checks_all_seeds and "
                      "controls_for_verdicts all hold this block); decides broken_null and broken_positive for "
                      "the verdicts and scientific_validity.controls_ok. Headline seeds already passed the "
                      "per-seed validity gates, so they cannot reveal a broken control")
# The analysis components compute_contrasts runs (contrasts with bootstrap CIs, the GLMM, the deviance drop).
CONTRAST_COMPONENTS = ("contrasts", "glmm", "deviance_drop")
CONTRASTS_BASIS_HEADLINE = "headline_seeds"
CONTRASTS_BASIS_ALL = "all_recorded_seeds_descriptive"
CONTRASTS_BASIS_PRESERVED = "preserved_rows_of_unrecorded_seeds_descriptive"
CONTRASTS_BASIS_NONE = "none_no_item_rows_recorded"
# Name of the go/no-go line printed for loading the reader, embedder, NLI and spaCy models.
MODEL_LOAD_PRECONDITION = "model_load"
# Keys of a control_checks block that state a broken-control decision; the headline-only block never carries
# them under these names (it cannot show a broken control, see NULL_CONTROL_BASIS).
_BROKEN_KEYS = ("broken_null", "broken_positive", "null_broken", "positive_broken")

# Run-wide tally of eligibility outcomes (label -> count), filled by install_eligibility_tally().
ELIGIBILITY_TALLY: Counter = Counter()
# First baseline answers seen per outcome label (length and tail), for diagnosing rejections.
ELIGIBILITY_SAMPLES: Dict[str, List[Dict[str, Any]]] = {}
ELIGIBILITY_SAMPLE_MAX = 3
ELIGIBILITY_SAMPLE_CHARS = 240
_ELIGIBILITY_LABEL_KEYS = ("label", "reason", "status", "first_failure", "failed", "criterion")
_BASELINE_ANSWER_KEYS = ("answer", "text", "generation", "output")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _budget_sec(hp: Dict[str, Any]) -> float:
    for name in ("RC_TIME_BUDGET_SEC", "TIME_BUDGET_SEC"):
        value = os.environ.get(name)
        if value and value.strip():
            return float(value)
    return float(hp["time_budget_sec"])


SEEDS = list(HYPERPARAMETERS["seeds"])
SMOKE = _env_flag("RC_SMOKE_TEST")
TIME_BUDGET = _budget_sec(HYPERPARAMETERS)


def report_stop(message: str) -> None:
    """Prints a run-stopping reason on stdout and stderr, so a non-zero exit is explained in both streams."""
    print(message, flush=True)
    print(message, file=sys.stderr, flush=True)


def eligibility_label(result: Any) -> str:
    """A short label for one check_eligibility return value, whatever its shape (str, bool, tuple, dict)."""
    if result is None:
        return "returned_None"
    if isinstance(result, bool):
        return "eligible" if result else "ineligible"
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in _ELIGIBILITY_LABEL_KEYS:
            if key in result:
                return str(result[key])
        return f"dict_without_label:{sorted(map(str, result.keys()))[:4]}"
    if isinstance(result, (tuple, list)):
        for part in result:
            if isinstance(part, str):
                return part
        for part in result:
            if isinstance(part, bool):
                return "eligible" if part else "ineligible"
        return f"{type(result).__name__}_without_label"
    return type(result).__name__


def baseline_answer_of(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Optional[str]:
    """The baseline answer text passed to check_eligibility(item, baseline, ...), or None when the call
    carries no baseline dict with a text field."""
    baseline = kwargs.get("baseline", args[1] if len(args) > 1 else None)
    if not isinstance(baseline, Mapping):
        return None
    for key in _BASELINE_ANSWER_KEYS:
        value = baseline.get(key)
        if isinstance(value, str):
            return value
    return None


def _record_sample(label: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    samples = ELIGIBILITY_SAMPLES.setdefault(label, [])
    if len(samples) >= ELIGIBILITY_SAMPLE_MAX:
        return
    answer = baseline_answer_of(args, kwargs)
    item = args[0] if args else kwargs.get("item")
    sid = str(item.get("sample_id")) if isinstance(item, Mapping) else None
    samples.append({"sample_id": sid, "answer_chars": None if answer is None else len(answer),
                    "answer_tail": None if answer is None else answer[-ELIGIBILITY_SAMPLE_CHARS:]})


def install_eligibility_tally() -> List[str]:
    """Wraps check_eligibility in probes, pilot and seed_loop so every call adds its outcome label (or the
    class of the exception it raised, which is then re-raised unchanged) to ELIGIBILITY_TALLY and keeps the
    first baseline answers per label in ELIGIBILITY_SAMPLES. Returns the module names patched. Behaviour of
    the wrapped function is unchanged."""
    original = probes_module.check_eligibility
    if getattr(original, "_eligibility_tally", False):
        return []

    @functools.wraps(original)
    def tallied(*args: Any, **kwargs: Any) -> Any:
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            label = f"raised:{type(exc).__name__}"
            ELIGIBILITY_TALLY[label] += 1
            _record_sample(label, args, kwargs)
            raise
        label = eligibility_label(result)
        ELIGIBILITY_TALLY[label] += 1
        _record_sample(label, args, kwargs)
        return result

    setattr(tallied, "_eligibility_tally", True)
    patched = []
    for module in (probes_module, pilot_module, seed_loop_module):
        if getattr(module, "check_eligibility", None) is original:
            setattr(module, "check_eligibility", tallied)
            patched.append(module.__name__)
    return patched


def print_eligibility_tally(stage: str, to_stderr: bool = False) -> Dict[str, int]:
    """Prints the eligibility outcome counts so far and the sampled baseline answers per outcome; returns
    the counts as a plain dict."""
    tally = {k: int(v) for k, v in sorted(ELIGIBILITY_TALLY.items())}
    lines = [f"ELIGIBILITY_TALLY after {stage}: total={sum(tally.values())} outcomes={tally}"]
    for label in sorted(ELIGIBILITY_SAMPLES):
        for i, s in enumerate(ELIGIBILITY_SAMPLES[label]):
            lines.append(f"ELIGIBILITY_SAMPLE {label}[{i}]: sample_id={s['sample_id']} "
                         f"answer_chars={s['answer_chars']} answer_tail={s['answer_tail']!r}")
    for line in lines:
        print(line, flush=True)
        if to_stderr:
            print(line, file=sys.stderr, flush=True)
    return tally


def _suffixed(fn: Callable[..., Any], suffix: str) -> Callable[..., Any]:
    @functools.wraps(fn)
    def call(question: str, *args: Any, **kwargs: Any) -> Any:
        return fn(f"{question}{suffix}", *args, **kwargs)

    return call


def install_question_suffix(deps: Any, suffix: str, reason: str) -> Dict[str, Any]:
    """Replaces deps.generate / tf_slots / answer_lp by wrappers that append suffix to the question, so the
    generation and both teacher-forced passes of every arm read the same prompt. Returns the smoke-override
    record. A Deps object that refuses the replacement stops the run (FAIL line, exit 1)."""
    for name in QUESTION_SUFFIX_METHODS:
        try:
            setattr(deps, name, _suffixed(getattr(deps, name), suffix))
        except AttributeError as exc:
            report_stop(format_precondition("reader_prompt", False, f"cannot wrap Deps.{name} for the smoke "
                                            f"citation reminder: {type(exc).__name__}: {exc}"))
            report_stop("PRECONDITION_FAILED: stopping before any seed; no results written")
            sys.exit(1)
    return {"component": "prompt:reader_question", "key": "citation_reminder_suffix", "planned": "",
            "used": suffix, "reason": reason, "scope": "smoke_only", "preregistered": True,
            "applies_to": list(QUESTION_SUFFIX_METHODS)}


def build_config(hp: Mapping[str, Any]) -> SimpleNamespace:
    """Checks the key set, then reads and validates every key once through hp_schema.HP_VALIDATORS."""
    schema_gap = sorted(set(HP_VALIDATORS) ^ set(HYPERPARAMETERS))
    if schema_gap:
        raise KeyError(f"hp_schema.HP_VALIDATORS and HYPERPARAMETERS disagree on: {schema_gap}")
    given = set(hp.keys())
    missing = sorted(set(HYPERPARAMETERS) - given)
    if missing:
        raise KeyError(f"missing hyperparameters: {missing}")
    extra = sorted(given - set(HYPERPARAMETERS))
    if extra:
        raise KeyError(f"unknown hyperparameters: {extra}")
    values = validate_hyperparameters(hp)
    return SimpleNamespace(**copy.deepcopy(values))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if all(isinstance(x, str) for x in list(a) + list(b)):
            return sorted(a) == sorted(b)
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def smoke_hyperparameters(hp: Mapping[str, Any], smoke: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns a copy of hp with SMOKE_OVERRIDES applied when smoke is set, and one record per value that
    changed (pre-registered keys are marked preregistered=True and also appear as protocol deviations). A
    full run gets an unchanged copy and an empty list."""
    out: Dict[str, Any] = copy.deepcopy(dict(hp))
    applied: List[Dict[str, Any]] = []
    if not smoke:
        return out, applied
    for key, value in SMOKE_OVERRIDES.items():
        if key not in out:
            raise KeyError(f"smoke override {key!r} is not a hyperparameter")
        if key not in SMOKE_OVERRIDE_REASONS:
            raise KeyError(f"smoke override {key!r} has no recorded reason")
        if _same(out[key], value):
            continue
        applied.append({"component": f"hyperparameter:{key}", "key": key, "planned": out[key], "used": value,
                        "reason": SMOKE_OVERRIDE_REASONS[key], "scope": "smoke_only",
                        "preregistered": key in PREREGISTERED})
        out[key] = value
    return out, applied


def smoke_seed_subset(run_seeds: Sequence[int], confirmation_seeds: Optional[Sequence[int]], smoke: bool
                      ) -> Tuple[List[int], List[int], Optional[Dict[str, Any]]]:
    """In smoke mode keeps the first SMOKE_MAX_RUN_SEEDS run seeds (in harness order) and returns a
    reduced-component record naming the seeds not run; a full run returns the seeds unchanged and None."""
    run = [int(s) for s in run_seeds]
    conf = [int(s) for s in (confirmation_seeds or [])]
    if not smoke or len(run) <= SMOKE_MAX_RUN_SEEDS:
        return run, conf, None
    kept = run[:SMOKE_MAX_RUN_SEEDS]
    dropped = run[SMOKE_MAX_RUN_SEEDS:]
    record = {"component": "replication_seeds", "key": "run_seeds", "planned": run, "used": kept,
              "not_run": dropped, "scope": "smoke_only",
              "reason": f"smoke run checks the pipeline end to end on {SMOKE_MAX_RUN_SEEDS} seed(s); the full "
                        f"run executes every seed"}
    return kept, [s for s in conf if s in kept], record


def protocol_deviations(hp: Any) -> List[Dict[str, Any]]:
    """Every pre-registered setting whose used value differs from the plan."""
    used = dict(_hp(hp))
    out = []
    for k, plan_v in PREREGISTERED.items():
        if k not in used:
            out.append({"key": k, "plan": plan_v, "used": None, "reason": "missing from HYPERPARAMETERS"})
        elif not _same(used[k], plan_v):
            out.append({"key": k, "plan": plan_v, "used": used[k], "reason": "differs from pre-registered value"})
    return out


def format_precondition(name: str, ok: bool, detail: str = "") -> str:
    if ok:
        return f"PRECONDITION {name}: pass"
    return f"PRECONDITION {name}: FAIL {detail}".rstrip()


def _stage(name: str, t_start: float, budget: float) -> None:
    """Prints the elapsed time after a startup stage so a stop before the seed loop shows where the budget went."""
    elapsed = time.time() - t_start
    print(f"STAGE {name} done: elapsed={elapsed:.1f}s remaining={budget - elapsed:.1f}s of {budget:.0f}s",
          flush=True)


def load_deps_or_fail(hp: Dict[str, Any], datasets: Dict[str, Any],
                      loader: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Tuple[Deps, Any]]] = None
                      ) -> Tuple[Deps, Any]:
    """Loads every model as a go/no-go precondition; any loading error prints
    `PRECONDITION model_load: FAIL <error class>: <message>` and exits 1 before any other precondition or seed.
    loader defaults to this module's build_real_deps, looked up at call time."""
    load = loader if loader is not None else build_real_deps
    models = ", ".join(str(hp[k]) for k in ("reader_model_id", "embed_model_id", "nli_model_id", "spacy_model"))
    try:
        deps, reader = load(hp, datasets)
        dtype_name = str(reader.dtype_name)
    except Exception as exc:  # recorded on the FAIL line; the run stops here
        detail = f"{type(exc).__name__}: {exc} (models: {models})"
        report_stop(format_precondition(MODEL_LOAD_PRECONDITION, False, detail))
        report_stop("PRECONDITION_FAILED: stopping before any seed; no results written")
        sys.exit(1)
    print(format_precondition(MODEL_LOAD_PRECONDITION, True), flush=True)
    print(f"models loaded: {models} (reader dtype {dtype_name})", flush=True)
    return deps, reader


def precondition_hyperparameters(hp: Mapping[str, Any], demo_dev: Optional[Dict[str, Any]]
                                 ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Hyperparameters the precondition checks read. With the ALCE prompt file present they equal hp. When
    data.demo_deviation() reports the demonstrations unavailable (recorded zero-shot deviation), the
    demonstration minimum of the data check is 0; the run's own hyperparameters are not changed. Returns the
    check copy and a record of the relaxation (None when nothing changed)."""
    out = copy.deepcopy(dict(hp))
    if demo_dev is None or int(out["n_demos"]) == 0:
        return out, None
    record = {"component": "precondition:data_and_models", "key": "n_demos", "planned": int(out["n_demos"]),
              "used": 0, "scope": "precondition_check_only",
              "reason": "ALCE prompt file with demonstrations unavailable; the run proceeds zero-shot as the "
                        "recorded deviation deviations.demonstrations"}
    out["n_demos"] = 0
    return out, record


def run_preconditions(datasets: Dict[str, Any], data_root: str, hp: Dict[str, Any], deps: Deps,
                      ner_labels_fn: Callable[[], Sequence[str]], dtype_name: str) -> List[Check]:
    """Full-size go/no-go checks (not shrunk in smoke). Prints PRECONDITION lines; exits 1 on any FAIL. A
    missing ALCE prompt file is a recorded zero-shot deviation, not a FAIL (printed as a FLAG line and noted in
    the data_and_models detail)."""
    errors: Dict[str, str] = {}

    def attempt(name: str, fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # any failure becomes a FAIL line carrying its message
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    n = int(hp["n_passages"])
    check_hp, demo_relax = precondition_hyperparameters(hp, data_module.demo_deviation())
    if demo_relax is not None:
        print(f"FLAG: precondition data_and_models: demonstrations planned={demo_relax['planned']} "
              f"required=0 ({demo_relax['reason']})", flush=True)
    data = attempt("data_and_models", lambda: data_summary(datasets, data_root, str(hp["manifest_glob"])), {})
    reader = attempt("reader_numerics", lambda: reader_numerics_check(deps.answer_lp, deps.tf_slots,
                                                                      datasets["val"].items, dtype_name), (False, ""))
    digits = attempt("digit_tokens", lambda: {d: deps.token_len(str(d)) for d in range(1, n + 1)}, {})
    nli_correct = attempt("nli_sanity", lambda: nli_sanity_correct(deps.nli), 0)
    labels = attempt("spacy_ner", lambda: list(ner_labels_fn()), [])
    raw_checks = precondition_checks(data, reader, digits, nli_correct, labels, check_hp, errors)
    checks: List[Check] = []
    for name, ok, detail in raw_checks:
        if demo_relax is not None and name == "data_and_models":
            detail = f"{detail} {ZERO_SHOT_PRECONDITION_NOTE}".strip()
        checks.append((name, ok, detail))
    for name, ok, detail in checks:
        line = format_precondition(name, ok, detail)
        if ok:
            print(line, flush=True)
        else:
            report_stop(line)
    if not all(ok for _n, ok, _d in checks):
        report_stop("PRECONDITION_FAILED: stopping before any seed; no results written")
        sys.exit(1)
    return checks


def _control_flags(null: Optional[float], pos: Optional[float], hp: Dict[str, Any]) -> Dict[str, Any]:
    """None means the rate was undefined (empty group); it is neither met nor broken."""
    return {"null_itt": null, "positive_target_cited": pos,
            "null_expected_met": None if null is None else bool(null <= float(hp["null_expected_max"])),
            "null_broken": bool(null is not None and null > float(hp["null_broken_above"])),
            "positive_expected_met": None if pos is None else bool(pos >= float(hp["pos_expected_min"])),
            "positive_broken": bool(pos is not None and pos < float(hp["pos_broken_below"]))}


def control_checks(per_seed: Dict[Any, Dict[str, Dict[str, Any]]], hp: Any) -> Dict[str, Any]:
    """Null ITT vs 0.03 (expected) / 0.05 (broken); positive target_cited vs 0.50 / 0.30; per seed and pooled."""
    hp = _hp(hp)
    per: Dict[str, Dict[str, Any]] = {}
    nulls, poss = [], []
    for seed in sorted(per_seed, key=str):
        conds = per_seed[seed] or {}
        null = conds.get(NULL_ARM, {}).get("itt_citation_migration_rate")
        pos = conds.get(POSITIVE_ARM, {}).get("target_cited_rate")
        per[str(seed)] = _control_flags(null, pos, hp)
        if null is not None:
            nulls.append(float(null))
        if pos is not None:
            poss.append(float(pos))
    pooled = _control_flags(float(np.mean(nulls)) if nulls else None, float(np.mean(poss)) if poss else None, hp)
    return {"per_seed": per, **pooled, "broken_null": pooled["null_broken"],
            "broken_positive": pooled["positive_broken"],
            "thresholds": {k: hp[k] for k in ("null_expected_max", "null_broken_above", "pos_expected_min",
                                              "pos_broken_below")},
            "pooling": "mean over the seeds passed in"}


def headline_descriptive_controls(block: Dict[str, Any]) -> Dict[str, Any]:
    """The headline-seed control block without any unqualified broken-control key: the pooled broken flags are
    renamed *_headline_descriptive and broken_null / broken_positive are dropped, so the only unqualified
    broken_* values in results.json are the run-level ones the verdicts use."""
    out = {k: v for k, v in block.items() if k not in _BROKEN_KEYS}
    out["null_broken_headline_descriptive"] = bool(block["null_broken"])
    out["positive_broken_headline_descriptive"] = bool(block["positive_broken"])
    out["decides_verdicts"] = False
    return out


def item_overlap(ids_by_seed: Dict[str, Sequence[str]]) -> List[str]:
    """sample_ids included under more than one seed (expected: seeds share one pool)."""
    seen = Counter(i for ids in ids_by_seed.values() for i in set(ids))
    return sorted(i for i, c in seen.items() if c > 1)


# ---------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------
def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def print_metric_lines(per_seed: Dict[Any, Dict[str, Dict[str, Any]]],
                       aggregates: Dict[str, Dict[str, Any]]) -> None:
    """Per-seed values and means over the given aggregates; undefined (None) values are not printed."""
    for seed in sorted(per_seed, key=str):
        for cond in sorted(per_seed[seed]):
            for key, v in sorted(per_seed[seed][cond].items()):
                if _finite_number(v):
                    print(f"condition={cond} seed={seed} {key}: {v}", flush=True)
    for cond in sorted(aggregates):
        for key, agg in sorted(aggregates[cond].items()):
            m, s = agg.get("mean"), agg.get("std")
            if not _finite_number(m):
                continue
            line = f"condition={cond} {key}_mean: {m}"
            if _finite_number(s):
                line += f" {key}_std: {s}"
            print(line, flush=True)


def _aggregate(per_seed: Dict[Any, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    metric_keys = sorted({k for conds in per_seed.values() for m in conds.values() for k in m})
    aggregates: Dict[str, Dict[str, Any]] = {c: {} for c in sorted({c for cs in per_seed.values() for c in cs})}
    for key in metric_keys:
        for c, agg in aggregate_over_seeds(per_seed, key).items():
            aggregates.setdefault(c, {})[key] = agg
    return aggregates


def _run_level_controls_ok(controls_all: Dict[str, Any]) -> Optional[bool]:
    """False when either run-level control is broken; None when either rate is undefined; else True."""
    if controls_all["broken_null"] or controls_all["broken_positive"]:
        return False
    if controls_all["null_itt"] is None or controls_all["positive_target_cited"] is None:
        return None
    return True


def aggregate_and_write(harness: Any, cfg: Any, timing: Dict[str, Any]) -> Dict[str, Any]:
    """Reads only harness.seed_records(); timing carries run-level info (unrun seeds, failures, backend,
    smoke overrides). Means are over every record; contrasts use headline seeds (descriptive fallback over
    every recorded seed, then over rows preserved from unrecorded seeds, flagged); controls are run-level.
    Never prints a PRIMARY line (the harness does at write_results)."""
    hp = _hp(cfg)
    records = harness.seed_records()
    counters = new_counters()
    per_seed: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    headline: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    excluded: List[Dict[str, Any]] = []
    validities: Dict[str, Dict[str, Any]] = {}
    designs: Dict[str, Any] = {}
    pooled_valid: List[Dict[str, Any]] = []
    alias_offsets: List[float] = []
    all_rows: List[Dict[str, Any]] = []
    all_alias_offsets: List[float] = []
    skipped: List[str] = []
    restored_seeds: List[str] = []
    reduced: List[Dict[str, Any]] = list(timing.get("reduced_components", []))
    smoke_overrides: List[Dict[str, Any]] = list(timing.get("smoke_overrides", []))
    for override in smoke_overrides:
        if override not in reduced:
            reduced.append(override)
    ids_by_seed: Dict[str, List[str]] = {}
    shortfalls: Dict[str, int] = {}
    budget_dev = False
    for rec in records:
        seed = rec["seed"]
        conds = rec.get("conditions") or {}
        extra, restored = restore_extra(seed, rec.get("extra") or {})
        if restored:
            restored_seeds.append(str(seed))
        per_seed[seed] = conds
        for k, v in (extra.get("counters") or {}).items():
            counters[k] += int(v)
        validity = seed_validity(conds, hp)
        validities[str(seed)] = validity
        design = extra.get("design") or {}
        designs[str(seed)] = design
        budget_dev = budget_dev or bool(design.get("budget_deviation"))
        ids_by_seed[str(seed)] = list(extra.get("included_ids", []))
        if extra.get("shortfall"):
            shortfalls[str(seed)] = int(extra["shortfall"])
        for comp in design.get("skipped_components", []):
            if comp not in skipped:
                skipped.append(comp)
        for comp in design.get("reduced_components", []):
            if comp not in reduced:
                reduced.append(comp)
        has_rows = "item_rows" in extra
        if not has_rows:
            counters["item_rows_missing"] += 1
        else:
            all_rows.extend(extra["item_rows"] or [])
            all_alias_offsets.extend(extra.get("alias_offsets", []))
        if validity.get("valid") and has_rows:
            headline[seed] = conds
            pooled_valid.extend(extra["item_rows"] or [])
            alias_offsets.extend(extra.get("alias_offsets", []))
        else:
            reasons = ([] if validity.get("valid") else ["seed_validity failed"]) + \
                      ([] if has_rows else ["item_rows missing"])
            excluded.append({"seed": str(seed), "reasons": reasons, "validity": validity})
    if restored_seeds:
        print(f"FLAG: item rows restored from the process copy of the recorded payload for seeds "
              f"{restored_seeds} (the harness record carried no item_rows)", flush=True)
    reused = item_overlap(ids_by_seed)
    # Run-level quantity: set (not summed) so it always equals item_reuse_across_seeds.n_reused.
    counters["item_overlap_across_seeds"] = len(reused)
    print(f"item reuse: {len(reused)} sample_ids included under more than one seed (seeds share one pool; "
          f"arms paired by (seed, sample_id), inference clustered by sample_id)", flush=True)
    print(f"headline seeds: {sorted(map(str, headline))}; excluded: "
          f"{[(e['seed'], e['reasons']) for e in excluded]}", flush=True)
    aggregates = _aggregate(per_seed)
    aggregates_headline = _aggregate(headline)
    print_metric_lines(per_seed, aggregates)
    primary = aggregates.get(TOPICAL_END, {}).get("itt_citation_migration_rate", {})
    reference = aggregates.get(SELF_END, {}).get("itt_citation_migration_rate", {})
    primary_h = aggregates_headline.get(TOPICAL_END, {}).get("itt_citation_migration_rate", {})
    reference_h = aggregates_headline.get(SELF_END, {}).get("itt_citation_migration_rate", {})
    if _finite_number(primary_h.get("mean")):
        print(f"condition={TOPICAL_END} itt_citation_migration_rate_headline_seeds_mean: {primary_h['mean']}",
              flush=True)
    controls_headline_raw = control_checks(headline, hp)
    controls_headline_raw["pooling"] = "mean over headline seeds, descriptive only (" + HEADLINE_SEED_SET + ")"
    controls_headline = headline_descriptive_controls(controls_headline_raw)
    controls_all = control_checks(per_seed, hp)
    controls_all["pooling"] = "mean over every recorded seed, including excluded seeds (" + NULL_CONTROL_BASIS + ")"
    run_null_broken = bool(controls_all["broken_null"])
    run_pos_broken = bool(controls_all["broken_positive"])
    # Both controls are judged over every recorded seed (headline seeds already passed the validity gates).
    verdict_controls = dict(controls_all)
    print(f"controls (run-level, {len(per_seed)} recorded seeds): null_itt={controls_all['null_itt']} "
          f"broken_null={run_null_broken} positive_target_cited={controls_all['positive_target_cited']} "
          f"broken_positive={run_pos_broken}", flush=True)
    contrasts = compute_contrasts(pooled_valid, hp, counters, alias_offsets) if pooled_valid else None
    # Without headline seeds the contrasts, GLMM and deviance drop still run descriptively: on every recorded
    # row, else on the rows preserved from seeds that failed or stopped before being recorded.
    contrasts_descriptive: Optional[Dict[str, Any]] = None
    descriptive_counters = new_counters()
    preserved: List[Dict[str, Any]] = []
    preserved_offsets: List[float] = []
    preserved_sources: List[Dict[str, Any]] = []
    if not pooled_valid and not all_rows:
        preserved, preserved_offsets, preserved_sources = preserved_rows(timing)
    if pooled_valid:
        contrasts_basis = CONTRASTS_BASIS_HEADLINE
    elif all_rows:
        contrasts_basis = CONTRASTS_BASIS_ALL
        contrasts_descriptive = compute_contrasts(all_rows, hp, descriptive_counters, all_alias_offsets)
        for comp in CONTRAST_COMPONENTS:
            reduced.append({"component": comp, "planned": "headline seeds (passing seed_validity)",
                            "used": f"every recorded seed with item rows ({len(all_rows)} rows); descriptive, "
                                    f"verdicts not drawn from it",
                            "reason": "no recorded seed passed seed_validity"})
    elif preserved:
        contrasts_basis = CONTRASTS_BASIS_PRESERVED
        contrasts_descriptive = compute_contrasts(preserved, hp, descriptive_counters, preserved_offsets)
        for comp in CONTRAST_COMPONENTS:
            reduced.append({"component": comp, "planned": "headline seeds (passing seed_validity)",
                            "used": f"item rows preserved from seeds that were not recorded ({len(preserved)} "
                                    f"rows: {source_summary(preserved_sources)}); descriptive, verdicts not "
                                    f"drawn from it",
                            "reason": "no recorded seed carried item rows"})
    else:
        contrasts_basis = CONTRASTS_BASIS_NONE
        report_stop(f"FLAG: no item row in any recorded seed ({len(records)} records), failed seed or "
                    f"budget-stopped seed; {list(CONTRAST_COMPONENTS)} cannot run and are listed as skipped")
        for comp in CONTRAST_COMPONENTS:
            if comp not in skipped:
                skipped.append(comp)
    print(f"contrasts basis: {contrasts_basis} (headline rows={len(pooled_valid)}, "
          f"all recorded rows={len(all_rows)}, preserved rows={len(preserved)}; "
          f"components {list(CONTRAST_COMPONENTS)})", flush=True)
    roles = sorted({r.get("pool", "dev") for r in pooled_valid})
    contrasts_by_pool: Dict[str, Any] = {}
    # Each pool counts its own fallbacks (the pooled contrasts already counted these rows run-level).
    pool_counters: Dict[str, Dict[str, int]] = {}
    if len(roles) > 1:
        for role in roles:
            role_counters = new_counters()
            contrasts_by_pool[role] = compute_contrasts([r for r in pooled_valid if r.get("pool", "dev") == role],
                                                        hp, role_counters, alias_offsets)
            pool_counters[role] = dict(role_counters)
            print(f"contrasts_by_pool[{role}] fallback counters (separate from run-level): "
                  f"{ {k: v for k, v in pool_counters[role].items() if v} }", flush=True)
    verdicts = decision_verdicts(contrasts, verdict_controls, hp)
    hp_devs = protocol_deviations(hp)
    probe_devs = probe_deviations()
    demo_dev = data_module.demo_deviation()
    unrun = list(timing.get("unrun_seeds", []))
    failures = list(timing.get("seed_failures", []))
    smoke = bool(timing.get("smoke", False))
    harness_info = dict(timing.get("harness", {}))
    flags = [f"protocol_deviation:{d['key']}" for d in hp_devs]
    flags += [f"protocol_deviation:positive_control_{d['key']}" for d in probe_devs]
    flags += [f"smoke_override:{o['key']}:{o['planned']}->{o['used']}" for o in smoke_overrides]
    if demo_dev is not None:
        flags.append("protocol_deviation:in_context_demonstrations_unavailable")
        if demo_dev not in reduced:
            reduced.append(demo_dev)
    flags += [f"seed_shortfall:seed={s}:missing={n}" for s, n in sorted(shortfalls.items())]
    flags += [f"unrun_seed:{u['seed']}:{u['reason']}" for u in unrun]
    flags += [f"seed_failed:{f['seed']}:{f['error_class']}" for f in failures]
    flags += [f"item_rows_restored_from_process:seed={s}" for s in restored_seeds]
    for f in failures:
        rows = (f.get("partial") or {}).get("item_rows") or []
        if rows:
            flags.append(f"seed_failed_item_rows_preserved:{f['seed']}:n={len(rows)}")
    flags += [f"seed_excluded_from_headline:{e['seed']}:{'+'.join(e['reasons'])}" for e in excluded]
    for s, f in sorted(controls_all["per_seed"].items()):
        if f["null_broken"]:
            flags.append(f"null_control_broken:seed={s}")
        if f["positive_broken"]:
            flags.append(f"positive_control_broken:seed={s}")
    for cond, key in ((budget_dev, "budget_deviation"),
                      (run_null_broken, "null_control_broken_run_level"),
                      (run_pos_broken, "positive_control_broken_run_level"),
                      (controls_headline["null_broken_headline_descriptive"], "null_control_broken_headline_seeds"),
                      (controls_headline["positive_broken_headline_descriptive"],
                       "positive_control_broken_headline_seeds"),
                      (controls_all["null_expected_met"] is False, "null_control_above_expected_0.03"),
                      (controls_all["positive_expected_met"] is False, "positive_control_below_expected_0.50"),
                      (counters["item_rows_missing"] > 0, "item_rows_missing"),
                      (counters["determinism_regen_failed"] > 0, "determinism_regen_failed"),
                      (bool(excluded), "primary_metric_includes_seeds_excluded_from_headline"),
                      (contrasts_basis == CONTRASTS_BASIS_ALL, "contrasts_computed_on_seeds_excluded_from_headline"),
                      (contrasts_basis == CONTRASTS_BASIS_PRESERVED, "contrasts_computed_on_rows_of_unrecorded_seeds"),
                      (contrasts_basis == CONTRASTS_BASIS_NONE, "contrasts_glmm_deviance_skipped_no_item_rows"),
                      (smoke, "smoke_run"),
                      (any(o.get("preregistered") for o in smoke_overrides),
                       "smoke_overrides_change_preregistered_values"),
                      (not records, "no_seed_recorded"), (bool(records) and not headline, "no_valid_seed")):
        if cond:
            flags.append(key)
    for f in flags:
        print(f"FLAG: {f}", flush=True)
    n_valid = sum(1 for v in validities.values() if v.get("valid"))
    controls_ok = _run_level_controls_ok(controls_all)
    self_m, null_m = reference_h.get("mean"), controls_headline["null_itt"]
    novelty = {"published_probe_condition": SELF_END, "null_condition": NULL_ARM, "self_itt_mean": self_m,
               "null_itt_mean": null_m, "seed_set": "headline seeds",
               "rule": "self-span ITT minus null ITT >= effect_threshold",
               "published_probe_moves_citations": (None if self_m is None or null_m is None
                                                   else bool(self_m - null_m >= float(hp["effect_threshold"])))}
    deviations = {"hyperparameters": hp_devs, "probes": probe_devs, "demonstrations": demo_dev,
                  "smoke_overrides": smoke_overrides,
                  "seed_shortfalls": shortfalls, "budget_deviation": budget_dev, "unrun_seeds": unrun,
                  "seed_failures": failures}
    eligibility_tally = {k: int(v) for k, v in sorted(ELIGIBILITY_TALLY.items())}
    results = {
        "primary_metric_key": "itt_citation_migration_rate", "direction": "minimize",
        "primary_condition": TOPICAL_END, "baseline_condition": SELF_END, "condition_names": list(CONDITION_NAMES),
        "aggregate_seed_set": AGGREGATE_SEED_SET, "headline_seed_set": HEADLINE_SEED_SET,
        "null_control_basis": NULL_CONTROL_BASIS,
        "recorded_seeds": sorted(map(str, per_seed)), "headline_seeds": sorted(map(str, headline)),
        "excluded_seeds": excluded,
        "primary_metric": primary.get("mean"), "primary_metric_std": primary.get("std"),
        "baseline_metric": reference.get("mean"), "baseline_metric_std": reference.get("std"),
        "aggregates": aggregates,
        "primary_metric_headline_seeds": primary_h.get("mean"),
        "primary_metric_std_headline_seeds": primary_h.get("std"),
        "baseline_metric_headline_seeds": reference_h.get("mean"),
        "baseline_metric_std_headline_seeds": reference_h.get("std"),
        "aggregates_headline_seeds": aggregates_headline, "per_seed": {str(s): c for s, c in per_seed.items()},
        "seed_validity": validities, "success_rate": success_rate(list(validities.values())),
        # One run-level control block (every recorded seed) used by the verdicts; headline block is descriptive.
        "control_checks": controls_all, "controls": controls_all, "control_checks_all_seeds": controls_all,
        "controls_for_verdicts": verdict_controls,
        "control_checks_headline_descriptive": controls_headline,
        "control_arm_mapping": CONTROL_ARM_MAPPING,
        "contrasts": contrasts, "contrasts_basis": contrasts_basis,
        "contrasts_descriptive_all_recorded_seeds": contrasts_descriptive,
        "contrasts_preserved_row_sources": preserved_sources,
        "item_rows_restored_from_process": restored_seeds,
        "contrasts_by_pool": contrasts_by_pool, "verdicts": verdicts,
        "novelty_precondition": novelty, "flags": flags, "smoke": smoke,
        "smoke_overrides": smoke_overrides,
        "eligibility_outcomes": {"counts": eligibility_tally,
                                 "samples": copy.deepcopy(ELIGIBILITY_SAMPLES),
                                 "scope": "every check_eligibility call in this process (pilot and seed loop); "
                                          f"samples: first {ELIGIBILITY_SAMPLE_MAX} baseline answers per outcome, "
                                          f"last {ELIGIBILITY_SAMPLE_CHARS} characters"},
        "scientific_validity": {"n_seeds_recorded": len(records), "n_valid_seeds": n_valid,
                                "n_headline_seeds": len(headline), "n_excluded_seeds": len(excluded),
                                "null_control_broken_run_level": run_null_broken,
                                "positive_control_broken_run_level": run_pos_broken,
                                "null_control_basis": NULL_CONTROL_BASIS, "contrasts_basis": contrasts_basis,
                                "controls_ok": controls_ok, "smoke": smoke,
                                "is_scientific_result": bool(not smoke and len(headline) >= 3
                                                             and controls_ok is True)},
        "deviations": deviations, "protocol_deviations": deviations,
        "unrun_seeds": unrun, "seed_failures": failures,
        "item_reuse_across_seeds": {"n_reused": len(reused), "sample_ids": reused,
                                    "pairing": "(seed, sample_id)", "inference": "clustered by sample_id",
                                    "fallback_counter": "fallback_counters.item_overlap_across_seeds "
                                                        "holds the same count"},
        "fallback_counters": dict(counters),
        "fallback_counters_descriptive_contrasts": dict(descriptive_counters),
        "fallback_counters_by_pool": pool_counters,
        "skipped_components": skipped,
        "reduced_components": reduced, "designs": designs,
        "backend": dict(timing.get("backend", {})), "harness": harness_info,
        "preconditions": list(timing.get("preconditions", [])),
        "time": {"elapsed_sec": time.time() - float(timing.get("t_start", time.time())),
                 "estimate_sec": timing.get("est_sec"), "recalibrated_to_sec": timing.get("recalibrated_to_sec"),
                 "measured_vs_estimate": timing.get("measured_vs_estimate"),
                 "time_warnings": list(timing.get("time_warnings", []))},
        "hyperparameters": hp, "preregistered": PREREGISTERED, "excluded_selections": EXCLUDED_SELECTIONS,
        "data_config": {k: v for k, v in DATA_CONFIG.items() if k not in ("instruction", "demo_prompt", "doc_prompt")},
    }
    clean = finalize_results(results)
    harness.write_results(clean)
    print(f"verdicts: {[v['label'] for v in verdicts]}", flush=True)
    return clean


def run_and_write(harness: Any, design: Optional[Dict[str, Any]], run_seeds: Sequence[int], deps: Optional[Deps],
                  datasets: Dict[str, Any], cfg: Any, run_info: Dict[str, Any], budget: float) -> Dict[str, Any]:
    """Runs the seed loop, then aggregates and writes. No exception from the loop is caught: it propagates
    unchanged, a finally block prints the ABORTED line, and nothing is written."""
    t_start = float(run_info["t_start"])
    est = float(run_info.get("est_sec") or 0.0)
    completed = False
    try:
        if design is not None:
            run_all_seeds(run_seeds, design, deps, datasets, cfg, harness, run_info, budget)
        completed = True
    finally:
        if not completed:
            exc = sys.exc_info()[1]
            what = f"{type(exc).__name__}: {exc}" if exc is not None else "an escaping exception"
            report_stop(f"ABORTED: the seed loop stopped on {what}; results.json is not written; "
                        f"seeds already recorded stay in the harness store")
    elapsed = time.time() - t_start
    run_info["measured_vs_estimate"] = elapsed / est if est > 0 else None
    print(f"measured_vs_estimate={run_info['measured_vs_estimate']} (actual {elapsed:.0f}s / "
          f"estimated {est:.0f}s)", flush=True)
    print_eligibility_tally("seed_loop")
    return aggregate_and_write(harness, cfg, run_info)


def _design_or_stop(harness: Any, deps: Deps, datasets: Dict[str, Any], cfg: Any, run_seeds: Sequence[int],
                    confirmation_seeds: Sequence[int], smoke: bool, budget: float, t_start: float,
                    dtype_name: str) -> Dict[str, Any]:
    """Runs the pilot and fixes the design. A refusal inside pilot_and_design (budget or pilot failure) keeps
    its exit code; its reason is echoed to stderr together with the time arithmetic, the eligibility outcome
    counts and the sampled rejected answers, then re-raised."""
    try:
        return pilot_and_design(harness, deps, datasets, cfg, run_seeds, confirmation_seeds, smoke, budget,
                                t_start, dtype_name)
    except SystemExit as exc:
        if exc.code not in (0, None):
            elapsed = time.time() - t_start
            print_eligibility_tally("pilot (refused)", to_stderr=True)
            report_stop(f"STOPPED in pilot_and_design (exit {exc.code}): the pilot-fitted design was refused "
                        f"(budget or pilot failure, see the preceding stdout lines); elapsed={elapsed:.1f}s "
                        f"budget={budget:.0f}s run_seeds={list(run_seeds)} smoke={smoke}; no results written")
        raise


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="same as RC_SMOKE_TEST=1")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args(argv)
    t_start = time.time()
    # RC_SMOKE_TEST / RC_TIME_BUDGET_SEC are read from the environment at import (SMOKE, TIME_BUDGET).
    smoke = bool(SMOKE or args.smoke)
    budget = float(TIME_BUDGET)
    run_hp, smoke_overrides = smoke_hyperparameters(HYPERPARAMETERS, smoke)
    cfg = build_config(run_hp)
    hp = _hp(cfg)
    for o in smoke_overrides:
        tag = " [pre-registered value, smoke only]" if o.get("preregistered") else ""
        print(f"FLAG: smoke override {o['key']}: planned={o['planned']} used={o['used']}{tag} ({o['reason']})",
              flush=True)
    patched = install_eligibility_tally()
    print(f"eligibility outcomes tallied in: {patched}", flush=True)
    print(f"seed payload capture installed: {install_seed_extra_capture()} (item rows kept in-process in case "
          f"the harness record omits them)", flush=True)
    plan = smoke_config(cfg, smoke)
    harness = experiment_harness.get_harness(budget)
    all_seeds = [int(s) for s in experiment_harness.seeds(SEEDS)]
    done = sorted({int(r["seed"]) for r in harness.seed_records()})
    planned_run_seeds, planned_confirmation = plan_run_seeds(all_seeds, hp["seeds"], smoke, done)
    run_seeds, confirmation_seeds, seed_cut = smoke_seed_subset(planned_run_seeds, planned_confirmation, smoke)
    smoke_unrun: List[Dict[str, Any]] = []
    if seed_cut is not None:
        smoke_unrun = [{"seed": s, "reason": SMOKE_SEED_REASON} for s in seed_cut["not_run"]]
        print(f"FLAG: smoke replication reduced: planned run seeds={seed_cut['planned']} "
              f"used={seed_cut['used']} not run={seed_cut['not_run']} ({seed_cut['reason']})", flush=True)
    rc_env = parse_rc_seeds(os.environ.get("RC_SEEDS"))
    if rc_env is not None and sorted(rc_env) != sorted(all_seeds):
        print(f"FLAG: RC_SEEDS={rc_env} differs from experiment_harness.seeds(SEEDS)={all_seeds}; the harness "
              f"list is used", flush=True)
    for d in protocol_deviations(hp):
        print(f"FLAG: protocol deviation {d['key']}: plan={d['plan']} used={d['used']}", flush=True)
    for d in probe_deviations():
        print(f"FLAG: protocol deviation {d['component']}.{d['key']}: plan={d['planned']} used={d['used']} "
              f"({d['reason']})", flush=True)

    import torch
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
    print(f"backend: cuda_available={cuda_ok} device={device_name} smoke={smoke} budget={budget:.0f}s "
          f"reader={hp['reader_model_id']} max_new_tokens={hp['max_new_tokens']} "
          f"seeds={all_seeds} already_recorded={done} to_run={run_seeds}", flush=True)
    datasets = get_datasets(args.data_root)
    demo_dev = data_module.demo_deviation()
    if demo_dev is not None:
        print(f"FLAG: protocol deviation in_context_demonstrations_unavailable: {demo_dev}", flush=True)
    elig = float(hp["eligible_rate_prior"])
    pools_ok, pool_lines = check_pools(run_seeds, hp["seeds"], datasets, int(plan["n_max"]), elig)
    for line in pool_lines:
        print(line, flush=True)
    pool_line = format_precondition("pool_sufficient", pools_ok, "see POOL_INSUFFICIENT lines")
    if not pools_ok:
        report_stop(pool_line)
        for line in pool_lines:
            print(line, file=sys.stderr, flush=True)
        report_stop("POOL_INSUFFICIENT: refusing to start; no results written")
        sys.exit(1)
    print(pool_line, flush=True)
    _stage("data_load", t_start, budget)
    # Model loading is a precondition: a missing or unloadable model prints a FAIL line and exits 1.
    deps, reader = load_deps_or_fail(hp, datasets)
    _stage("model_load", t_start, budget)
    if smoke:
        reminder = install_question_suffix(deps, SMOKE_CITATION_REMINDER, SMOKE_CITATION_REASON)
        smoke_overrides.append(reminder)
        print(f"FLAG: smoke override {reminder['key']}: planned={reminder['planned']!r} used={reminder['used']!r} "
              f"on {reminder['applies_to']} [pre-registered prompt, smoke only] ({reminder['reason']})", flush=True)
    checks = run_preconditions(datasets, args.data_root, hp, deps, lambda: spacy_ner_labels(hp["spacy_model"]),
                               reader.dtype_name)
    _stage("preconditions", t_start, budget)
    checks = [(MODEL_LOAD_PRECONDITION, True, "")] + list(checks)
    backend = backend_info(hp, reader.dtype_name, cuda_ok, device_name)
    harness_info = {"backend": HARNESS_MODULE, "resumed_design": bool(getattr(harness, "resumed_design", None)),
                    "already_recorded_seeds": done}
    design: Optional[Dict[str, Any]] = None
    if run_seeds:
        design = _design_or_stop(harness, deps, datasets, cfg, run_seeds, confirmation_seeds, smoke, budget,
                                 t_start, reader.dtype_name)
        print_eligibility_tally("pilot")
        design = finalize_design(design, hp, run_seeds, smoke)
        design.setdefault("pool_check", pool_lines)
        print(f"design: n_items/seed={design['n_items']} est={float(design['est_sec']):.0f}s seeds={run_seeds} "
              f"confirmation={design.get('confirmation_seeds')} primary={design['primary_condition']} "
              f"baseline={design['baseline_condition']} sizing={design.get('sizing')} "
              f"reduced={[r.get('component') for r in design['reduced_components']]}", flush=True)
        _stage("pilot_and_design", t_start, budget)
    else:
        print(f"all seeds {all_seeds} already recorded; aggregating only", flush=True)
    est = float(design["est_sec"]) if design else 0.0
    reduced_run: List[Dict[str, Any]] = list(design["reduced_components"]) if design else []
    if seed_cut is not None:
        reduced_run.append(seed_cut)
    _, demo_relax = precondition_hyperparameters(hp, demo_dev)
    if demo_relax is not None:
        reduced_run.append(demo_relax)
    run_info: Dict[str, Any] = {"t_start": t_start, "smoke": smoke, "backend": backend, "est_sec": est,
                                "harness": harness_info,
                                "preconditions": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                                "reduced_components": reduced_run,
                                "smoke_overrides": list(smoke_overrides),
                                "unrun_seeds": list(smoke_unrun), "seed_failures": []}
    return run_and_write(harness, design, run_seeds, deps, datasets, cfg, run_info, budget)


if __name__ == "__main__":
    main()