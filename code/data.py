"""
Data loading for the doctored-passage RAG citation probe.

CIFAR-10 is deliberately NOT loaded: image data cannot test citation behaviour. The data is
ASQA in ALCE format (princeton-nlp/ALCE-data, asqa_eval_gtr_top100.json), downloaded by
setup.py into ./data/alce and read offline here.

Split policy: 'train' = the ALCE in-context demonstrations (prompt only, never scored);
'val' = development pool (sha256(sample_id) bucket >= 0.40); 'test' = confirmation pool
(bucket < 0.40). The dev/confirmation split is deterministic and seed-independent, so no
item ever appears in both pools.

Demonstrations: the ALCE prompt file (asqa_default.json) ships with the ALCE code repository, not
with the HF dataset that setup.py downloads. When it is absent under data_root the run proceeds
zero-shot (0 demonstrations); this is a recorded deviation (DATA_CONFIG["demo_deviation"], a FLAG
line at load, and demo_deviation() for results.json), never a silent substitution. A prompt file
that exists but cannot be parsed is still an error.

Per-seed pool (plan item 4): pool_for_seed returns the WHOLE development pool for a seed in
plan_seeds and the WHOLE confirmation pool otherwise. The choice depends only on the seed id.
Seeds of the same pool therefore draw overlapping per-seed subsets (each seed scans in its own
seeded order); analyses pair arms by (seed, sample_id) and inference is clustered by sample_id.

P2e natural-location weights: the alias sentence offsets are computed with the SAME spaCy
sentence span splitter the random-boundary arms use to place and bin their injections
(baselines.spacy_sentence_spans, loaded lazily through baselines.get_sent_nlp), never with a
regex splitter, so a weight and a bin always refer to the same sentence boundaries.

The offsets are computed LAZILY, per item, the first time an item's "alias_sentence_offsets"
is read (or the item is iterated / copied). Loading the corpus runs no spaCy pass at all, so
the spaCy cost scales with the items a run actually touches (smoke mode shrinks it with the
scan). alias_offsets_computed() reports how many items have been split so far.
"""
import copy
import glob
import hashlib
import json
import os
import re
import string
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

__all__ = ["DATA_CONFIG", "PURPOSE_CODE", "normalize_answer", "gold_aliases", "alias_in_text", "sentence_starts",
           "rough_sentence_count", "alias_sentence_offsets", "AsqaAlceDataset", "get_datasets", "pool_for_seed",
           "item_rng", "scan_order", "alias_offsets_computed", "demo_deviation"]

SpanSplitter = Callable[[str], Iterable[Tuple[int, int]]]

DATA_CONFIG = {
    "name": "ASQA-ALCE",
    "hf_id": "princeton-nlp/ALCE-data",
    "eval_file": "asqa_eval_gtr_top100.json",
    "demo_file": "asqa_default.json",
    "task": "long-form QA with passage-ID citations (RAG citation probe)",
    "modality": "text",
    "num_classes": None,
    "input_shape": None,
    "n_in_context_passages": 5,
    "foil_mining_ranks": (6, 20),
    "min_target_sentences": 3,
    "confirmation_fraction": 0.40,
    "expected_num_items": 948,
    "per_seed_pool": "whole val pool for plan seeds, whole test pool otherwise (overlapping across seeds)",
    "reader_model": "Qwen/Qwen2.5-3B-Instruct",
    "embedder": "BAAI/bge-small-en-v1.5",
    "nli_model": "cross-encoder/nli-deberta-v3-base",
    "regime_factors": {"evidence_redundancy": ["single_source", "redundant"],
                       "target_slot": ["early_slot_1_2", "late_slot_4_5"]},
    "excluded_datasets": {"CIFAR-10": "image classification; irrelevant to RAG citation behaviour"},
    "demo_path": None,
    "demo_deviation": None,
}

# ITEM 5
# Every random draw has its own purpose code; "null_drop" seeds the null arm's random sentence drop.
PURPOSE_CODE = {"coin": 1, "target": 2, "foil": 3, "swap": 4, "boundary": 5, "donor": 6, "poscontrol": 7,
                "null_drop": 8}

OFFSETS_KEY = "alias_sentence_offsets"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# Number of items whose spaCy alias offsets have been computed in this process.
_OFFSET_STATS = {"items_computed": 0}


def alias_offsets_computed() -> int:
    """How many items have had their spaCy alias sentence offsets computed so far."""
    return int(_OFFSET_STATS["items_computed"])


def demo_deviation() -> Optional[Dict[str, Any]]:
    """The recorded deviation when the ALCE prompt file was missing (zero-shot run), else None."""
    dev = DATA_CONFIG.get("demo_deviation")
    return dict(dev) if dev else None


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def gold_aliases(item: Dict[str, Any]) -> List[str]:
    aliases = []
    for qa in item.get("qa_pairs", []) or []:
        for a in qa.get("short_answers", []) or []:
            n = normalize_answer(a)
            if n:
                aliases.append(n)
    return sorted(set(aliases))


def alias_in_text(aliases: List[str], text: str) -> bool:
    t = " " + normalize_answer(text) + " "
    return any((" " + a + " ") in t for a in aliases)


def sentence_starts(text: str) -> List[int]:
    """Coarse regex sentence starts. Never used for P2e offsets (those use the probe's span splitter)."""
    return [0] + [m.end() for m in _SENT_SPLIT.finditer(text)]


def rough_sentence_count(text: str) -> int:
    return len([s for s in _SENT_SPLIT.split(text.strip()) if s.strip()])


def _find_file(data_root: str, filename: str) -> Optional[str]:
    hits = glob.glob(os.path.join(data_root, "**", filename), recursive=True)
    return sorted(hits, key=len)[0] if hits else None


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _hash_bucket(sample_id: str) -> float:
    h = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class AsqaAlceDataset(Dataset):
    """ASQA items in ALCE format with pre-treatment coverage fields."""

    def __init__(self, items: List[Dict[str, Any]], role: str):
        self.role = role
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def coverage_rate(self) -> Optional[float]:
        if not self.items:
            return None
        return sum(1 for it in self.items if it.get("covered")) / len(self.items)

    def covered_items(self) -> List[Dict[str, Any]]:
        return [it for it in self.items if it.get("covered")]

    def by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(it["sample_id"]): it for it in self.items}


def _probe_span_splitter() -> Tuple[SpanSplitter, str]:
    """The spaCy sentence spans the random-boundary arms use, and the name recorded for them.

    Calls baselines.spacy_sentence_spans (lazy spaCy load through get_sent_nlp); the regex
    fallback of baselines.sentence_spans is never used here.
    """
    import baselines  # lazy: baselines imports this module

    model_name = baselines.SPACY_SENT_MODEL

    def split(text: str) -> List[Tuple[int, int]]:
        return [(int(s), int(e)) for s, e in baselines.spacy_sentence_spans(text, model_name)]

    return split, f"baselines.spacy_sentence_spans(spaCy {model_name}, same splitter as random-boundary arms)"


def alias_sentence_offsets(aliases: List[str], docs: List[Dict[str, str]],
                           span_splitter: SpanSplitter) -> List[float]:
    """Relative start offsets of sentences containing a gold alias, using the probe's span splitter."""
    if span_splitter is None:
        raise TypeError("alias_sentence_offsets needs the probe's span splitter; the regex splitter is not allowed")
    offs: List[float] = []
    for d in docs:
        text = d["text"]
        if not text or not aliases:
            continue
        for start, end in span_splitter(text):
            s, e = int(start), int(end)
            if not 0 <= s < e <= len(text):
                raise ValueError(f"span splitter returned span ({s}, {e}) outside text of length {len(text)}")
            if alias_in_text(aliases, text[s:e]):
                offs.append(s / len(text))
    return offs


class _LazyOffsetsItem(dict):
    """An item dict whose "alias_sentence_offsets" is computed with the probe splitter on first use.

    Reading the key (item[k], item.get(k)) or any bulk view (iteration, keys/items/values, len,
    copy, dict(item), {**item}) materialises it, so no consumer can observe the item without it.
    """

    def __init__(self, data: Dict[str, Any], span_splitter: SpanSplitter):
        super().__init__(data)
        self._span_splitter = span_splitter

    def _materialize(self) -> "_LazyOffsetsItem":
        if not dict.__contains__(self, OFFSETS_KEY):
            offs = alias_sentence_offsets(dict.__getitem__(self, "aliases"),
                                          dict.__getitem__(self, "context_docs"), self._span_splitter)
            dict.__setitem__(self, OFFSETS_KEY, offs)
            _OFFSET_STATS["items_computed"] += 1
        return self

    def __missing__(self, key: Any) -> Any:
        if key == OFFSETS_KEY:
            self._materialize()
            return dict.__getitem__(self, key)
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if key == OFFSETS_KEY:
            return self[key]
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        return key == OFFSETS_KEY or dict.__contains__(self, key)

    def __iter__(self) -> Iterator[Any]:
        self._materialize()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._materialize()
        return dict.__len__(self)

    def keys(self):  # type: ignore[override]
        self._materialize()
        return dict.keys(self)

    def items(self):  # type: ignore[override]
        self._materialize()
        return dict.items(self)

    def values(self):  # type: ignore[override]
        self._materialize()
        return dict.values(self)

    def __eq__(self, other: object) -> bool:
        self._materialize()
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        self._materialize()
        return dict.__repr__(self)

    def copy(self) -> Dict[str, Any]:  # type: ignore[override]
        self._materialize()
        return dict(dict.items(self))

    def __copy__(self) -> Dict[str, Any]:
        return self.copy()

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[str, Any]:
        self._materialize()
        return copy.deepcopy(dict(dict.items(self)), memo)

    def __reduce__(self) -> Tuple[Any, Tuple[Dict[str, Any]]]:
        self._materialize()
        return (dict, (dict(dict.items(self)),))


def _prepare_item(raw: Dict[str, Any], idx: int, span_splitter: SpanSplitter) -> Dict[str, Any]:
    k = DATA_CONFIG["n_in_context_passages"]
    lo, hi = DATA_CONFIG["foil_mining_ranks"]
    docs = raw.get("docs", []) or []
    if len(docs) < k:
        raise ValueError(f"item {idx} has only {len(docs)} docs (< {k})")

    def _doc(d: Dict[str, Any]) -> Dict[str, str]:
        return {"title": d.get("title", ""), "text": d.get("text", "")}

    context_docs = [_doc(d) for d in docs[:k]]
    foil_docs = [_doc(d) for d in docs[lo - 1:hi]]
    aliases = gold_aliases(raw)
    alias_slots = [i + 1 for i, d in enumerate(context_docs) if aliases and alias_in_text(aliases, d["text"])]
    covered = len(alias_slots) > 0
    redundancy = "uncovered" if not covered else ("single_source" if len(alias_slots) == 1 else "redundant")
    data = {
        "sample_id": str(raw.get("sample_id", f"asqa_{idx}")),
        "question": raw["question"],
        "gold_long_answer": raw.get("answer", ""),
        "aliases": aliases,
        "context_docs": context_docs,
        "foil_docs": foil_docs,
        "covered": covered,
        "alias_doc_slots": alias_slots,
        "redundancy": redundancy,
        "doc_sentence_counts": [rough_sentence_count(d["text"]) for d in context_docs],
    }
    # alias_sentence_offsets (spaCy) is computed on first use, never for the whole corpus at load time.
    return _LazyOffsetsItem(data, span_splitter)


def _load_prompt_config(data_root: str) -> Dict[str, Any]:
    """The ALCE prompt file, or {} (zero-shot) when it is absent; the absence is recorded and printed."""
    demo_path = _find_file(data_root, DATA_CONFIG["demo_file"])
    if demo_path is None:
        dev = {"component": "in_context_demonstrations",
               "planned": f"ALCE demonstrations and instruction from {DATA_CONFIG['demo_file']}",
               "used": "0 demonstrations (zero-shot); empty instruction/demo_prompt/doc_prompt fields passed "
                       "to the reader",
               "reason": f"{DATA_CONFIG['demo_file']} not found under {data_root}; setup.py downloads only "
                         f"{DATA_CONFIG['eval_file']} (the prompt file ships with the ALCE code repository)"}
        DATA_CONFIG["demo_path"] = None
        DATA_CONFIG["demo_deviation"] = dev
        print(f"FLAG: protocol deviation in_context_demonstrations: {dev['reason']}; run proceeds with "
              f"{dev['used']}", flush=True)
        return {}
    try:
        prompt_cfg = _load_json(demo_path)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to parse ALCE prompt file {demo_path}: {e}") from e
    if not isinstance(prompt_cfg, dict):
        raise RuntimeError(f"ALCE prompt file {demo_path} is not a JSON object.")
    DATA_CONFIG["demo_path"] = demo_path
    DATA_CONFIG["demo_deviation"] = None
    return prompt_cfg


# ITEM 4
def get_datasets(data_root: str = "./data", span_splitter: Optional[SpanSplitter] = None) -> dict:
    """
    span_splitter: the sentence span splitter the random-boundary arms use. When None, the spaCy
    splitter baselines.spacy_sentence_spans is used (the arms' production splitter); the choice
    is recorded in DATA_CONFIG["alias_offset_splitter"]. The splitter is NOT run here: each item
    computes its offsets lazily on first use, so loading costs no spaCy pass.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root '{data_root}' does not exist. Run setup.py first.")
    eval_path = _find_file(data_root, DATA_CONFIG["eval_file"])
    if eval_path is None:
        raise FileNotFoundError(f"Could not find {DATA_CONFIG['eval_file']} under {data_root}; run setup.py.")
    try:
        raw_items = _load_json(eval_path)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to parse {eval_path}: {e}") from e
    if not isinstance(raw_items, list) or not raw_items:
        raise RuntimeError(f"{eval_path} did not contain a non-empty list of items.")
    if span_splitter is None:
        span_splitter, splitter_name = _probe_span_splitter()
    else:
        splitter_name = "caller-supplied probe span splitter"
    DATA_CONFIG["alias_offset_splitter"] = splitter_name
    DATA_CONFIG["alias_offset_timing"] = "lazy per item on first use (no corpus-wide spaCy pass at load)"
    print(f"alias offsets (P2e natural weights) use: {splitter_name}; computed lazily per used item", flush=True)
    items, bad = [], []
    for i, raw in enumerate(raw_items):
        try:
            items.append(_prepare_item(raw, i, span_splitter))
        except (KeyError, ValueError) as e:
            bad.append(f"{i}: {type(e).__name__}: {e}")
    if bad:
        print(f"FLAG: {len(bad)} unparseable ASQA items, first: {bad[0]}", flush=True)
    if not items:
        raise RuntimeError("No usable ASQA items after parsing.")
    conf_frac = DATA_CONFIG["confirmation_fraction"]
    dev_items = [it for it in items if _hash_bucket(it["sample_id"]) >= conf_frac]
    conf_items = [it for it in items if _hash_bucket(it["sample_id"]) < conf_frac]
    prompt_cfg = _load_prompt_config(data_root)
    demos = list(prompt_cfg.get("demos", []) or [])
    DATA_CONFIG["instruction"] = prompt_cfg.get("instruction", "")
    DATA_CONFIG["demo_prompt"] = prompt_cfg.get("demo_prompt", "")
    DATA_CONFIG["doc_prompt"] = prompt_cfg.get("doc_prompt", "")
    train_ds = AsqaAlceDataset(demos, role="demonstrations")
    val_ds = AsqaAlceDataset(dev_items, role="development_pool")
    test_ds = AsqaAlceDataset(conf_items, role="confirmation_pool")
    DATA_CONFIG.update({
        "eval_path": eval_path, "num_items_total": len(items), "num_items_unparseable": len(bad),
        "unparseable_items": bad,
        "num_dev_pool": len(dev_items), "num_confirmation_pool": len(conf_items),
        "coverage_rate_dev": val_ds.coverage_rate(), "coverage_rate_confirmation": test_ds.coverage_rate(),
        "num_demos": len(demos),
        "redundancy_counts_dev": {r: sum(1 for it in dev_items if it["redundancy"] == r)
                                  for r in ("single_source", "redundant", "uncovered")},
    })
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def pool_for_seed(seed: int, plan_seeds: Sequence[int], datasets: dict) -> List[Dict[str, Any]]:
    """
    Whole development pool ('val') when seed is in plan_seeds, whole confirmation pool ('test')
    otherwise. Depends only on the seed id; the two pools are hash-disjoint by construction.
    """
    split = "val" if int(seed) in {int(s) for s in plan_seeds} else "test"
    return list(datasets[split].items)


def item_rng(seed: int, sample_id: str, purpose: str) -> np.random.Generator:
    if purpose not in PURPOSE_CODE:
        raise KeyError(f"unregistered random purpose '{purpose}'; known: {sorted(PURPOSE_CODE)}")
    h = int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([int(seed), h, PURPOSE_CODE[purpose]])


# ITEM 6
def scan_order(pool: List[Dict[str, Any]], seed: int) -> List[str]:
    ids = sorted(str(it["sample_id"]) for it in pool)
    perm = np.random.default_rng(int(seed)).permutation(len(ids))
    return [ids[int(j)] for j in perm]