"""
Item-row sources for the analysis components (contrasts, GLMM, deviance drop).

1. Process-local copy of every seed payload passed to seed_loop.record_seed_payload, so a harness whose
   seed_records() does not return the recorded `extra` block (item rows, alias offsets, counters, design)
   cannot silently turn the analysis into a skipped component. A restored payload is reported by the caller
   (FLAG line and results.json item_rows_restored_from_process).
2. Item rows preserved from seeds that were not recorded (a failed seed's partial rows, a budget stop's
   preserved rows). They feed a descriptive-only analysis when no recorded seed carries any item row; the
   caller records it as a reduced component and never draws verdicts from it.
"""
import functools
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import seed_loop as seed_loop_module
from seed_loop import BUDGET_STOP_ROWS_KEY

__all__ = ["SEED_EXTRA_STORE", "install_seed_extra_capture", "restore_extra", "preserved_rows"]

# str(seed) -> shallow copy of the extra block passed to record_seed_payload in this process.
SEED_EXTRA_STORE: Dict[str, Dict[str, Any]] = {}
_CAPTURE_TAG = "_seed_extra_capture"


def install_seed_extra_capture() -> bool:
    """Wraps seed_loop.record_seed_payload so every payload it records is also kept in SEED_EXTRA_STORE.
    The copy is taken only after the wrapped call returns (a seed the harness refused is not stored).
    Returns False when the wrapper is already installed."""
    original = seed_loop_module.record_seed_payload
    if getattr(original, _CAPTURE_TAG, False):
        return False

    @functools.wraps(original)
    def capturing(harness: Any, seed: int, conds: Dict[str, Dict[str, Any]], extra: Dict[str, Any]) -> None:
        original(harness, seed, conds, extra)
        SEED_EXTRA_STORE[str(seed)] = dict(extra or {})

    setattr(capturing, _CAPTURE_TAG, True)
    seed_loop_module.record_seed_payload = capturing
    return True


def restore_extra(seed: Any, extra: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The harness extra block, with keys it lacks filled from the process copy of the same seed when the
    harness block carries no item_rows. Returns (extra, restored)."""
    given = dict(extra or {})
    if "item_rows" in given:
        return given, False
    stored = SEED_EXTRA_STORE.get(str(seed))
    if not stored or "item_rows" not in stored:
        return given, False
    return {**stored, **given}, True


def _row_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(r) for r in value if isinstance(r, Mapping)]


def _finite_offsets(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    return [float(o) for o in value
            if isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(float(o))]


def _entry_rows(entry: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[float]]:
    partial = entry.get("partial")
    partial = partial if isinstance(partial, Mapping) else {}
    for source in (partial, entry):
        for key in ("item_rows", BUDGET_STOP_ROWS_KEY):
            rows = _row_list(source.get(key))
            if rows:
                return rows, _finite_offsets(source.get("alias_offsets"))
    return [], []


def preserved_rows(timing: Mapping[str, Any]
                   ) -> Tuple[List[Dict[str, Any]], List[float], List[Dict[str, Any]]]:
    """Item rows kept from seeds that were not recorded: seed_failures[*].partial, unrun_seeds entries and a
    run-level budget-stop block. Returns (rows, alias offsets, one source record per contributing seed)."""
    rows: List[Dict[str, Any]] = []
    offsets: List[float] = []
    sources: List[Dict[str, Any]] = []
    entries: List[Tuple[str, Any]] = [("seed_failed", f) for f in (timing.get("seed_failures") or [])]
    entries += [("unrun_seed", u) for u in (timing.get("unrun_seeds") or [])]
    stop_block = timing.get(BUDGET_STOP_ROWS_KEY)
    if isinstance(stop_block, Mapping):
        entries += [("budget_stop", {"seed": s, "item_rows": v}) for s, v in stop_block.items()]
    elif isinstance(stop_block, (list, tuple)):
        entries.append(("budget_stop", {"seed": None, "item_rows": stop_block}))
    for kind, entry in entries:
        if not isinstance(entry, Mapping):
            continue
        found, found_offsets = _entry_rows(entry)
        if not found:
            continue
        rows.extend(found)
        offsets.extend(found_offsets)
        sources.append({"kind": kind, "seed": str(entry.get("seed")), "n_rows": len(found)})
    return rows, offsets, sources


def source_summary(sources: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(f"{s['kind']}:seed={s['seed']}:n={s['n_rows']}" for s in sources) or "none"