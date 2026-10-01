"""Toy items and fake scorers for the pytest suite only; main.py never imports this module."""
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baselines import sentence_spans, slots_from_logprobs, split_sentences
from data import AsqaAlceDataset, normalize_answer
from probes import Deps

KNOWN_NAMES = ("Paris", "Berlin", "Rome", "Madrid")
TEST_STOP_WORDS = frozenset({"is", "the", "a", "what", "of", "in", "it"})
FILLER = "Alpha one is here. Beta two is there. Gamma three is near."
FOIL_TEXT = "Lorem ipsum dolor sit. Amet consectetur adipiscing elit. Sed do eiusmod tempor."


def make_item(sid: str, city: str, redundancy: str = "single_source") -> Dict[str, Any]:
    ctx = [{"title": f"{city} page", "text": f"{city} is the capital. It is big. It is old."}]
    ctx += [{"title": f"Filler {i}", "text": FILLER} for i in range(2, 6)]
    slots = [1]
    if redundancy == "redundant":
        ctx[2] = {"title": f"{city} river", "text": f"{city} has a river. The river is wide. The river is long."}
        slots = [1, 3]
    return {"sample_id": sid, "question": f"What is the capital {sid}?", "gold_long_answer": f"The answer is {city}.",
            "aliases": [city.lower()], "context_docs": ctx,
            "foil_docs": [{"title": f"Foil {r}", "text": FOIL_TEXT} for r in range(6, 21)],
            "covered": True, "alias_doc_slots": slots, "redundancy": redundancy,
            "doc_sentence_counts": [3] * 5, "alias_sentence_offsets": [0.0]}


def toy_pool() -> List[Dict[str, Any]]:
    return [make_item("q1", "Paris"), make_item("q2", "Berlin"), make_item("q3", "Rome", "redundant"),
            make_item("q4", "Madrid", "redundant")]


def toy_datasets() -> Dict[str, AsqaAlceDataset]:
    return {"train": AsqaAlceDataset([], "demonstrations"), "val": AsqaAlceDataset(toy_pool(), "dev"),
            "test": AsqaAlceDataset([make_item("z9", "Paris")], "confirmation")}


def answer_table(pool: List[Dict[str, Any]]) -> Dict[str, str]:
    return {it["question"]: f"{it['gold_long_answer'].split()[-1].rstrip('.')} is the answer [1]." for it in pool}


def fake_tf(question: str, docs: Any, answer: str, counters: Counter) -> List[Dict[str, Any]]:
    digits = [int(m) for m in re.findall(r"\[(\d)\]", answer)]
    return slots_from_logprobs([[-0.1, -3.0, -3.0, -3.0, -3.0] for _ in digits], digits)


def fake_sent_nll(sentence: str, counters: Counter) -> Optional[float]:
    if len(sentence.split()) < 2:
        counters["ppl_too_short"] += 1
        return None
    return 2.0


def fake_embed(texts: List[str], is_query: bool = False) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0]), (len(texts), 1))


def fake_nli(premise: str, hypothesis: str, counters: Optional[Counter] = None) -> Dict[str, float]:
    names_p = {n for n in KNOWN_NAMES if n in premise}
    names_h = {n for n in KNOWN_NAMES if n in hypothesis}
    contra = 0.9 if names_h and names_h != names_p else 0.05
    hw, pw = set(normalize_answer(hypothesis).split()), set(normalize_answer(premise).split())
    ent = 0.9 if hw and hw <= pw else 0.05
    return {"contradiction": contra, "entailment": ent, "neutral": max(0.0, 1.0 - contra - ent)}


def fake_ner(text: str) -> List[Tuple[int, int, str, str]]:
    ents = [(m.start(), m.end(), "GPE", n) for n in KNOWN_NAMES for m in re.finditer(r"\b%s\b" % n, text)]
    return sorted(ents)


def make_fake_deps(hp: Dict[str, Any], table: Dict[str, str], generate: Any = None) -> Deps:
    def gen(question: str, docs: Any) -> str:
        return table[question]

    return Deps(hp, generate=generate or gen, tf_slots=fake_tf, answer_lp=lambda q, d, a: -1.0,
                sent_nll=fake_sent_nll, token_len=lambda s: len(s.split()), embed=fake_embed, nli=fake_nli,
                splitter=split_sentences, span_splitter=sentence_spans, ner=fake_ner, stop_words=TEST_STOP_WORDS)


class FakeHarness:
    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self.records = list(records or [])
        self.written: Optional[Dict[str, Any]] = None
        self.resumed_design = None

    def should_stop(self) -> bool:
        return False

    def check_value(self, value: float, name: str) -> bool:
        return bool(np.isfinite(value))

    def record_seed(self, seed: int, conditions: Dict[str, Any], expected_conditions: Any = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
        self.records.append({"seed": seed, "conditions": conditions, "extra": extra or {}})

    def seed_records(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def write_results(self, extra: Dict[str, Any]) -> None:
        self.written = extra


def row(cond: str, sid: str, move: bool = False, tc: bool = False, kept: bool = True, parse: bool = True,
        margin: Optional[float] = None, rel: Optional[str] = None, delta: Optional[float] = None,
        obin: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    return {"condition": cond, "sample_id": sid, "seed": seed, "move": move, "target_cited": tc, "answer_kept": kept,
            "answer_flip": not kept, "parse_ok": parse, "margin": margin, "rel_class": rel,
            "delta_logp_target_id": delta, "offset_bin": obin, "redundancy": "single_source", "stratum": "early",
            "t": 2, "next_exists": True, "prev_exists": True, "neighbor_next_new": False, "neighbor_prev_new": False}


def self_check_rows() -> Dict[str, List[Dict[str, Any]]]:
    """The Self-check pool; every list is deliberately out of id order (q2, q1, q3)."""
    from metrics import (BASELINE, NULL_ARM, POSITIVE_ARM, SELF_END, SELF_RANDOM, SWAP_END, TOPICAL_END,
                         TOPICAL_RANDOM)
    return {
        BASELINE: [row(BASELINE, "q2", margin=0.5), row(BASELINE, "q1", margin=0.1), row(BASELINE, "q3", margin=0.9)],
        SELF_END: [row(SELF_END, "q2", tc=True, kept=False), row(SELF_END, "q1", move=True, tc=True, rel="re_sourcing",
                                                                 delta=1.5), row(SELF_END, "q3")],
        TOPICAL_END: [row(TOPICAL_END, "q2", parse=False), row(TOPICAL_END, "q1"),
                      row(TOPICAL_END, "q3", move=True, tc=True, rel="mixed")],
        SWAP_END: [row(SWAP_END, s, kept=False) for s in ("q2", "q1", "q3")],
        SELF_RANDOM: [row(SELF_RANDOM, "q2", obin=4), row(SELF_RANDOM, "q1", move=True, tc=True, obin=0,
                                                          rel="mixed"), row(SELF_RANDOM, "q3", obin=2)],
        TOPICAL_RANDOM: [row(TOPICAL_RANDOM, s) for s in ("q2", "q1", "q3")],
        NULL_ARM: [row(NULL_ARM, s) for s in ("q2", "q1", "q3")],
        POSITIVE_ARM: [row(POSITIVE_ARM, s, move=True, tc=True) for s in ("q2", "q1", "q3")],
    }