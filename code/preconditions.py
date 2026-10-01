"""
Go/no-go checks run once before any seed. They never shrink in smoke mode.
Each check returns (name, ok, detail); main.run_preconditions prints them and exits 1 on any FAIL.
"""
import glob
import math
import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

Check = Tuple[str, bool, str]
PRECONDITION_NAMES = ("data_and_models", "reader_numerics", "digit_tokens", "nli_sanity", "spacy_ner")

# Eight hand-written pairs with obvious labels (fixed before any data is seen).
NLI_SANITY_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("A man is playing a guitar on stage.", "A man is playing an instrument.", "entailment"),
    ("A man is playing a guitar on stage.", "Nobody is playing any music.", "contradiction"),
    ("The cat is sleeping on the sofa.", "The cat is awake and running outside.", "contradiction"),
    ("The cat is sleeping on the sofa.", "An animal is on the sofa.", "entailment"),
    ("Paris is the capital of France.", "The capital of France is Paris.", "entailment"),
    ("Paris is the capital of France.", "Berlin is the capital of France.", "contradiction"),
    ("A woman is reading a book in the park.", "The woman is waiting for her best friend.", "neutral"),
    ("Two children are building a sandcastle on the beach.", "The two children are siblings.", "neutral"),
)
# A citing answer used to probe the reader's numerics on a full-length ALCE prompt.
READER_PROBE_ANSWER = "The documents state the answer directly [1][2]."


def manifest_paths(data_root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_root, "**", pattern), recursive=True))


def data_summary(datasets: Dict[str, Any], data_root: str, pattern: str) -> Dict[str, Any]:
    dev = list(datasets["val"].items)
    conf_ds = datasets.get("test")
    conf = list(conf_ds.items) if conf_ds is not None else []
    train = datasets.get("train")
    ids = {str(it["sample_id"]) for it in dev + conf}
    return {"n_items": len(ids), "n_demos": len(train) if train is not None else 0, "n_dev": len(dev),
            "n_conf": len(conf), "manifests": manifest_paths(data_root, pattern)}


def longest_covered_item(items: Sequence[Dict[str, Any]]) -> Any:
    covered = [it for it in items if it.get("covered")]
    if not covered:
        return None
    return max(covered, key=lambda it: (sum(len(d["title"]) + len(d["text"]) for d in it["context_docs"]),
                                        str(it["sample_id"])))


def reader_numerics_check(answer_lp: Callable, tf_slots: Callable, items: Sequence[Dict[str, Any]],
                          dtype_name: str) -> Tuple[bool, str]:
    item = longest_covered_item(items)
    if item is None:
        return False, "no covered item to build a full-length prompt from"
    lp = float(answer_lp(item["question"], item["context_docs"], READER_PROBE_ANSWER))
    slots = tf_slots(item["question"], item["context_docs"], READER_PROBE_ANSWER)
    vals = [float(v) for s in slots for v in s["logp"]]
    ok = math.isfinite(lp) and bool(vals) and all(math.isfinite(v) for v in vals)
    return ok, (f"dtype={dtype_name} item={item['sample_id']} answer_logp={lp} "
                f"slot_logps={len(vals)} all_finite={ok}")


def nli_sanity_correct(nli_fn: Callable[[str, str], Dict[str, float]]) -> int:
    correct = 0
    for premise, hypothesis, label in NLI_SANITY_PAIRS:
        scores = nli_fn(premise, hypothesis)
        # Bound method is taken per iteration, so no closure over the loop variable.
        correct += int(max(scores, key=scores.__getitem__) == label)
    return correct


def precondition_checks(data: Dict[str, Any], reader: Tuple[bool, str], digit_lens: Dict[int, int],
                        nli_correct: int, ner_labels: Sequence[str], hp: Dict[str, Any],
                        errors: Dict[str, str]) -> List[Check]:
    n_exp = int(hp["expected_eval_items"])
    d_ok = (data.get("n_items") == n_exp and int(data.get("n_demos", 0)) >= int(hp["n_demos"])
            and int(data.get("n_dev", 0)) > 0 and int(data.get("n_conf", 0)) > 0 and bool(data.get("manifests")))
    d_detail = (f"items={data.get('n_items')} (need {n_exp}) demos={data.get('n_demos')} (need >= {hp['n_demos']}) "
                f"dev={data.get('n_dev')} conf={data.get('n_conf')} manifests={len(data.get('manifests') or [])}")
    n = int(hp["n_passages"])
    bad = {d: ln for d, ln in digit_lens.items() if ln != 1}
    g_ok = len(digit_lens) == n and not bad
    need = int(hp["nli_sanity_min_correct"])
    missing = sorted(set(hp["entity_types"]) - set(ner_labels))
    raw = [("data_and_models", d_ok, d_detail),
           ("reader_numerics", bool(reader[0]), reader[1]),
           ("digit_tokens", g_ok, f"digits 1..{n} token lengths {digit_lens}; multi-token {bad}"),
           ("nli_sanity", int(nli_correct) >= need,
            f"{nli_correct}/{len(NLI_SANITY_PAIRS)} argmax correct (need >= {need})"),
           ("spacy_ner", not missing, f"entity types missing from NER labels: {missing}")]
    return [(name, bool(ok) and name not in errors, errors.get(name, detail)) for name, ok, detail in raw]