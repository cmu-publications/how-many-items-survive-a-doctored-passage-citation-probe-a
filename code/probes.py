"""
Condition strategies for the doctored-passage citation probe. Each arm implements its own
prepare() (pre-treatment payload; eligibility contribution) and edit_context() (a distinct
document edit). All randomness comes from item_rng(seed, sample_id, purpose).
"""
import copy
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from baselines import (blank_passage, extract_self_span, parse_citations, score_outcome, sentence_spans,
                       split_sentences, strip_citations)
from data import alias_in_text, item_rng, normalize_answer
from metrics import (BASELINE, CONDITION_ORDER, NULL_ARM, POSITIVE_ARM, SELF_END, SELF_RANDOM, SWAP_END,
                     TOPICAL_END, TOPICAL_RANDOM, neighbor_excess, new_counters)

DEFAULT_STRATA = {"early": [1, 2], "late": [4, 5]}
FOIL_FILTER_ORDER = ("alias", "overlap", "length", "cosine", "ppl", "nli")
ELIGIBILITY_LABELS = ("c1_uncovered", "c2_quota_full", "c3_parse", "c4_self_span", "c5_target",
                      "c5_boundary", "c6_foil", "c7_swap", "c8_poscontrol", "c9_margin")
# Plan: swap accepted iff NLI contradiction(baseline answer without citations -> swapped span) >= threshold.
SWAP_NLI_PREMISE = "baseline_answer_citations_stripped"
# Recorded deviation (plan item 17): the plan draws the swap donor entity from "another eligible item's gold
# answers". Eligibility itself depends on the swap prepare (c7), so donors come from every other item of the
# seed pool instead (NER on its gold long answer, entities equal to one of that item's aliases), covered or
# not. The current item is excluded inside swap() by sample id. Listed in probe_deviations().
DONOR_POOL_RULE = "every_other_item_of_seed_pool"
SWAP_DONOR_PLAN = "another eligible item's gold answers"
# Plan item 20 (null_edit): slot t text + "\n", every other slot and slot t's title unchanged. No content is
# added or removed; the arm draws nothing at random and adds no eligibility criterion.
NULL_EDIT_PLAN = "slot t text + '\\n'; all other slots and slot t title unchanged (plan item 20)"
# Null And Positive Control Arms contract (the null_random_drop prefix): after all arms ran on a seed, null rows
# are dropped at random so the null arm keeps items at the treatment's observed kept rate. The document edit
# stays NULL_EDIT_PLAN; the drop acts on the seed's rows, never on a passage, payload, or eligibility.
NULL_DROP_TREATMENT = SELF_END
# Must be a key of data.PURPOSE_CODE (item_rng raises KeyError otherwise).
NULL_DROP_PURPOSE = "null_drop"
NULL_DROP_RULE = (f"null row kept iff item_rng(seed, sample_id, '{NULL_DROP_PURPOSE}').random() < "
                  f"share of {NULL_DROP_TREATMENT} rows with kept == True in the same seed")
# Recorded deviation: the plan draws the positive-control donor from "an unrelated eligible item". Eligibility
# itself depends on the positive-control prepare (c8), so the donor is drawn from the covered items of the
# seed pool instead (a pre-treatment population). Written into every positive-control payload and row.
POSCONTROL_DONOR_POPULATION = "covered_items_of_seed_pool"
POSCONTROL_DONOR_PLAN = "top-1 passage of an unrelated eligible item"
Docs = List[Dict[str, str]]


def probe_deviations() -> List[Dict[str, str]]:
    """Plan deviations made inside the probes, for the results deviations block."""
    return [{"component": POSITIVE_ARM, "key": "donor_population", "planned": POSCONTROL_DONOR_PLAN,
             "used": POSCONTROL_DONOR_POPULATION,
             "reason": "eligibility (c8) depends on the donor, so eligible items cannot define the donor pool"},
            {"component": SWAP_END, "key": "donor_population", "planned": SWAP_DONOR_PLAN,
             "used": DONOR_POOL_RULE,
             "reason": "eligibility (c7) depends on the donor pool, so eligible items cannot define the donor "
                       "pool; donors come from every other seed-pool item, covered or not"}]


class Deps:
    """Holds every scorer; main.py injects real models, tests inject fakes."""

    def __init__(self, hp: Dict[str, Any], *, generate: Callable, tf_slots: Callable, answer_lp: Callable,
                 sent_nll: Callable, token_len: Callable, embed: Callable, nli: Callable,
                 splitter: Callable, span_splitter: Callable, ner: Callable, stop_words: frozenset):
        self.hp = hp
        self._generate, self._tf_slots, self._answer_lp = generate, tf_slots, answer_lp
        self._sent_nll, self._token_len, self._embed, self._nli = sent_nll, token_len, embed, nli
        self._splitter, self._span_splitter, self._ner = splitter, span_splitter, ner
        self.stop_words = frozenset(stop_words)
        self.counters: Counter = new_counters()
        self.donors: Dict[str, List[Tuple[str, str]]] = {}
        self.poscontrol_pool: List[Dict[str, Any]] = []

    def reset_counters(self) -> None:
        self.counters = new_counters()

    def set_seed_pool(self, pool: Sequence[Dict[str, Any]], donors: Dict[str, List[Tuple[str, str]]]) -> None:
        """Positive-control donors: covered items of the seed pool (POSCONTROL_DONOR_POPULATION)."""
        self.donors = donors
        self.poscontrol_pool = sorted((it for it in pool if it.get("covered")), key=lambda it: str(it["sample_id"]))

    def generate(self, question: str, docs: Docs) -> str:
        return self._generate(question, docs)

    def tf_slots(self, question: str, docs: Docs, answer: str) -> List[Dict[str, Any]]:
        return self._tf_slots(question, docs, answer, self.counters)

    def answer_lp(self, question: str, docs: Docs, answer: str) -> float:
        return float(self._answer_lp(question, docs, answer))

    def sent_nll(self, sentence: str) -> Optional[float]:
        return self._sent_nll(sentence, self.counters)

    def token_len(self, text: str) -> int:
        return int(self._token_len(text))

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        return np.asarray(self._embed(texts, is_query), dtype=np.float64)

    def nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        return self._nli(premise, hypothesis, self.counters)

    def splitter(self, text: str) -> List[str]:
        return list(self._splitter(text))

    def span_splitter(self, text: str) -> List[Tuple[int, int]]:
        return list(self._span_splitter(text))

    def ner(self, text: str) -> List[Tuple[int, int, str, str]]:
        return list(self._ner(text))


def content_tokens(text: str, stop_words: Iterable[str]) -> set:
    sw = set(stop_words)
    return {w for w in normalize_answer(text).split() if w not in sw}


# ITEM 14
def select_target(baseline_cited: Iterable[int], docs: Docs, stratum_coin_rng: np.random.Generator,
                  target_rng: np.random.Generator, splitter: Optional[Callable] = None,
                  strata: Optional[Dict[str, List[int]]] = None, min_sentences: int = 3
                  ) -> Tuple[Optional[int], str]:
    strata = strata or DEFAULT_STRATA
    stratum = "early" if stratum_coin_rng.random() < 0.5 else "late"
    cited = set(baseline_cited)
    pool = [k for k in strata[stratum]
            if k not in cited and len(split_sentences(docs[k - 1]["text"], splitter)) >= min_sentences]
    if not pool:
        return None, stratum
    return int(pool[int(target_rng.integers(len(pool)))]), stratum


# ITEM 20
def null_edit(docs: Docs, t: int) -> Docs:
    new_docs = copy.deepcopy(list(docs))
    new_docs[t - 1]["text"] = new_docs[t - 1]["text"] + "\n"
    return new_docs


def null_random_drop(rows: Dict[Tuple[str, str], Dict[str, Any]], seed: int
                     ) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Seed-level random drop of the null arm at the treatment's observed kept rate (NULL_DROP_RULE).

    `rows` holds one seed's rows keyed by (condition, sample_id). Returns a new dict without the dropped null
    rows (every other row, and every kept null row, is the same object) plus a record for results.json. The
    draw per item is item_rng(seed, sample_id, NULL_DROP_PURPOSE), so it does not depend on row order."""
    treat = [r for (cond, _sid), r in rows.items() if cond == NULL_DROP_TREATMENT]
    if not treat:
        raise ValueError(f"null random drop needs {NULL_DROP_TREATMENT} rows; the seed has none")
    no_kept = sorted(str(r.get("sample_id")) for r in treat if "kept" not in r)
    if no_kept:
        raise ValueError(f"{NULL_DROP_TREATMENT} rows without a 'kept' field: {no_kept}")
    rate = float(sum(1 for r in treat if r["kept"])) / len(treat)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dropped: List[str] = []
    n_null = 0
    for key, row in rows.items():
        if key[0] == NULL_ARM:
            n_null += 1
            if float(item_rng(seed, str(key[1]), NULL_DROP_PURPOSE).random()) >= rate:
                dropped.append(str(key[1]))
                continue
        out[key] = row
    record = {"condition": NULL_ARM, "treatment_condition": NULL_DROP_TREATMENT, "rule": NULL_DROP_RULE,
              "treatment_kept_rate": rate, "n_treatment_rows": len(treat), "n_null_before": n_null,
              "n_null_after": n_null - len(dropped), "dropped_sample_ids": sorted(dropped)}
    return out, record


# ITEM 21
def relocate_evidence(docs: Docs, t: int, o: int, donor_passage: Dict[str, str]) -> Docs:
    new_docs = copy.deepcopy(list(docs))
    new_docs[t - 1] = {"title": docs[o - 1]["title"], "text": docs[o - 1]["text"]}
    new_docs[o - 1] = {"title": donor_passage["title"], "text": donor_passage["text"]}
    return new_docs


def assert_only_slots_changed(orig: Docs, new: Docs, changed: Sequence[int]) -> None:
    for k in range(1, len(orig) + 1):
        if k not in changed and orig[k - 1] != new[k - 1]:
            raise ValueError(f"edit touched slot {k} outside the declared slots {list(changed)}")


# ITEM 23
def delta_logp_target(base_slots: Sequence[Dict[str, Any]], edited_slots: Sequence[Dict[str, Any]], t: int,
                      counters: Optional[Counter] = None) -> Optional[float]:
    if len(base_slots) == 0 or len(base_slots) != len(edited_slots):
        if counters is not None:
            counters["slot_mismatch"] += 1
        return None
    diffs = [float(e["logp"][t - 1]) - float(b["logp"][t - 1]) for b, e in zip(base_slots, edited_slots)]
    return float(sum(diffs) / len(diffs))


class ArmProbe:
    condition = ""
    reliance_applicable = True
    is_random_position = False
    position = "end"

    def __init__(self, hp: Dict[str, Any]):
        self.hp = hp
        self.n = int(hp["n_passages"])

    def prepare(self, item: Dict[str, Any], baseline: Dict[str, Any], payload: Dict[str, Any], seed: int,
                deps: Deps) -> Optional[str]:
        raise NotImplementedError

    def edit_context(self, item: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[Docs, Dict[str, Any]]:
        raise NotImplementedError

    def score_outcome(self, baseline: Dict[str, Any], edited: str, t: int, aliases: Sequence[str]) -> Dict[str, Any]:
        out = score_outcome(baseline, edited, t, aliases, self.n)
        out["position"] = self.position
        return out

    def _append_at_end(self, item: Dict[str, Any], payload: Dict[str, Any], plant: str) -> Tuple[Docs, Dict[str, Any]]:
        docs = copy.deepcopy(item["context_docs"])
        t = int(payload["t"])
        docs[t - 1]["text"] = docs[t - 1]["text"].rstrip() + self.hp["separator"] + plant
        return docs, {"changed_slots": [t], "plant": plant}

    def _row_meta(self, item: Dict[str, Any], baseline: Dict[str, Any], payload: Dict[str, Any],
                  meta: Dict[str, Any], edited: str) -> Dict[str, Any]:
        return {"condition": self.condition, "sample_id": str(item["sample_id"]), "t": int(payload["t"]),
                "o": int(payload["o"]), "stratum": payload.get("stratum"), "redundancy": item["redundancy"],
                "margin": baseline.get("margin"), "edited_answer": edited, "plant": meta.get("plant"),
                "offset": meta.get("offset"), "offset_bin": meta.get("offset_bin")}

    def run(self, item: Dict[str, Any], baseline: Dict[str, Any], payload: Dict[str, Any], generate_fn: Callable,
            tf_fn: Callable, counters: Counter) -> Dict[str, Any]:
        t = int(payload["t"])
        docs, meta = self.edit_context(item, payload)
        assert_only_slots_changed(item["context_docs"], docs, meta["changed_slots"])
        edited = generate_fn(item["question"], docs)
        row = self.score_outcome(baseline, edited, t, item["aliases"])
        ed_slots = tf_fn(item["question"], docs, baseline["answer"])
        row["delta_logp_target_id"] = delta_logp_target(baseline["citation_slots"], ed_slots, t, counters)
        row.update(self._row_meta(item, baseline, payload, meta, edited))
        row["_docs"] = docs
        return row


class BaselineUneditedArm(ArmProbe):
    condition = BASELINE
    reliance_applicable = False
    position = "none"

    @staticmethod
    def generate_baseline(item: Dict[str, Any], generate_fn: Callable, tf_fn: Callable,
                          n_passages: int = 5) -> Dict[str, Any]:
        docs = item["context_docs"]
        answer = generate_fn(item["question"], docs)
        cited, ok = parse_citations(answer, n_passages)
        slots = tf_fn(item["question"], docs, answer) if ok else []
        from baselines import answer_alias_set
        return {"answer": answer, "cited": sorted(cited), "parse_ok": ok,
                "alias_set": sorted(answer_alias_set(answer, item["aliases"])), "citation_slots": slots,
                "margin": min(s["margin"] for s in slots) if slots else None}

    def prepare(self, item, baseline, payload, seed, deps):
        if not baseline.get("citation_slots") or baseline.get("margin") is None:
            return "c9_margin"
        payload["base_margin"] = float(baseline["margin"])
        return None

    def edit_context(self, item, payload):
        return copy.deepcopy(item["context_docs"]), {"changed_slots": [], "plant": None}

    def run(self, item, baseline, payload, generate_fn, tf_fn, counters):
        """No regeneration: the unedited answer is the paired reference, ITT = 0 by construction."""
        docs, meta = self.edit_context(item, payload)
        row = self.score_outcome(baseline, baseline["answer"], int(payload["t"]), item["aliases"])
        row["delta_logp_target_id"] = 0.0
        row.update(self._row_meta(item, baseline, payload, meta, baseline["answer"]))
        row["_docs"] = docs
        return row


class SelfSpanEndInjectionProbe(ArmProbe):
    """Published probe: the answer's own alias sentence is appended to an uncited passage."""
    condition = SELF_END

    def _draw_target(self, item: Dict[str, Any], baseline: Dict[str, Any], seed: int,
                     deps: Deps) -> Tuple[Optional[int], str]:
        sid = str(item["sample_id"])
        return select_target(baseline["cited"], item["context_docs"], item_rng(seed, sid, "coin"),
                             item_rng(seed, sid, "target"), deps.splitter, self.hp["slot_strata"],
                             int(self.hp["min_target_sentences"]))

    def prepare(self, item, baseline, payload, seed, deps):
        span, ids = extract_self_span(baseline["answer"], item["aliases"], deps.splitter)
        if span is None or not ids:
            return "c4_self_span"
        t, stratum = self._draw_target(item, baseline, seed, deps)
        payload["stratum"] = stratum
        if t is None:
            return "c5_target"
        payload.update(span=span, o=int(min(ids)), t=int(t))
        return None

    def edit_context(self, item, payload):
        return self._append_at_end(item, payload, payload["span"])


class MatchedTopicalFoilInjectionProbe(ArmProbe):
    condition = TOPICAL_END

    # ITEM 15
    def mine_candidate_sentences(self, item: Dict[str, Any], splitter: Optional[Callable] = None
                                 ) -> List[Tuple[str, int]]:
        lo, hi = int(self.hp["foil_mining_ranks"][0]), int(self.hp["foil_mining_ranks"][1])
        out = []
        for rank, doc in enumerate(item["foil_docs"][: hi - lo + 1], start=lo):
            for s in split_sentences(doc["text"], splitter):
                out.append((s, rank))
        return out

    def _filter_alias(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        return [c for c in cands if not alias_in_text(item["aliases"], c)]

    def _filter_overlap(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        return [c for c in cands if not content_tokens(c, deps.stop_words) & ref["ans_only"]]

    def _filter_length(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        length_tol = float(self.hp["foil_length_tol"]) * ref["L0"]
        return [c for c in cands if abs(deps.token_len(c) - ref["L0"]) <= length_tol]

    def _filter_cosine(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        sims = deps.embed(list(cands), False) @ ref["q"]
        cos_tol = float(self.hp["foil_cosine_tol"])
        return [c for c, v in zip(cands, sims) if abs(float(v) - ref["c0"]) <= cos_tol]

    def _filter_ppl(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        ppl_tol = float(self.hp["foil_ppl_tol_nats"])
        passed: List[str] = []
        for c in cands:
            v = deps.sent_nll(c)
            if v is not None and abs(v - ref["nll0"]) <= ppl_tol:
                passed.append(c)
        return passed

    def _filter_nli(self, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any], deps: Deps) -> List[str]:
        contra_max = float(self.hp["topical_nli_contradiction_max"])
        return [c for c in cands if deps.nli(ref["span"], c)["contradiction"] < contra_max]

    def _apply_filter(self, name: str, cands: List[str], item: Dict[str, Any], ref: Dict[str, Any],
                      deps: Deps) -> List[str]:
        filters = {"alias": self._filter_alias, "overlap": self._filter_overlap, "length": self._filter_length,
                   "cosine": self._filter_cosine, "ppl": self._filter_ppl, "nli": self._filter_nli}
        if name not in filters:
            raise ValueError(f"unknown foil filter {name}")
        if not cands:
            return []
        return filters[name](cands, item, ref, deps)

    # ITEM 16
    def match_foil(self, item: Dict[str, Any], span: str, question: str, seed: int, deps: Deps,
                   filter_order: Sequence[str] = FOIL_FILTER_ORDER) -> Tuple[Optional[str], Dict[str, int]]:
        drops = {name: 0 for name in FOIL_FILTER_ORDER}
        seen, cands = set(), []
        for s, _rank in self.mine_candidate_sentences(item, deps.splitter):
            if s not in seen:
                seen.add(s)
                cands.append(s)
        nll0 = deps.sent_nll(span)
        if nll0 is None:
            drops["span_unscorable"] = len(cands)
            return None, drops
        q = deps.embed([question], True)[0]
        ref = {"span": span, "L0": deps.token_len(span), "nll0": nll0, "q": q,
               "c0": float(np.dot(q, deps.embed([span], False)[0])),
               "ans_only": content_tokens(span, deps.stop_words) - content_tokens(question, deps.stop_words)}
        survivors = cands
        for name in filter_order:
            next_survivors = self._apply_filter(name, survivors, item, ref, deps)
            drops[name] += len(survivors) - len(next_survivors)
            survivors = next_survivors
        if not survivors:
            return None, drops
        j = int(item_rng(seed, str(item["sample_id"]), "foil").integers(len(survivors)))
        return survivors[j], drops

    def prepare(self, item, baseline, payload, seed, deps):
        foil, drops = self.match_foil(item, payload["span"], item["question"], seed, deps)
        payload["foil_drops"] = drops
        if foil is None:
            return "c6_foil"
        payload["foil"] = foil
        return None

    def edit_context(self, item, payload):
        return self._append_at_end(item, payload, payload["foil"])


class EntitySwapFoilProbe(ArmProbe):
    condition = SWAP_END

    # ITEM 17
    @staticmethod
    def build_donor_pool(pool_items: Sequence[Dict[str, Any]], ner: Callable, entity_types: Sequence[str]
                         ) -> Dict[str, List[Tuple[str, str]]]:
        """Donor entities from every pool item (covered or not; DONOR_POOL_RULE, a recorded deviation from
        SWAP_DONOR_PLAN): NER on its gold long answer, keeping entities whose normalized text equals one of
        that item's aliases. swap() excludes the current item."""
        donors: Dict[str, List[Tuple[str, str]]] = {}
        types = set(entity_types)
        for it in sorted(pool_items, key=lambda x: str(x["sample_id"])):
            aliases = set(it["aliases"])
            for _s, _e, label, text in ner(it["gold_long_answer"]):
                if label in types and normalize_answer(text) in aliases:
                    donors.setdefault(label, []).append((str(it["sample_id"]), text))
        return donors

    def swap(self, span: str, aliases: Sequence[str], donors: Dict[str, List[Tuple[str, str]]], seed: int,
             sid: str, nli: Callable, ner: Callable, counters: Optional[Counter] = None
             ) -> Tuple[Optional[str], Dict[str, Any]]:
        """`nli(premise_placeholder, hypothesis)`: the first argument is the self-span; callers that follow
        the plan (prepare) pass an nli bound to the baseline answer as premise (SWAP_NLI_PREMISE)."""
        counters = counters if counters is not None else Counter()
        types = set(self.hp["entity_types"])
        alias_set = set(aliases)
        ents = [e for e in ner(span) if e[2] in types
                and any((" " + a + " ") in (" " + normalize_answer(e[3]) + " ") for a in aliases)]
        if not ents:
            return None, {"reason": "no_alias_entity"}
        start, end, label, _text = ents[0]
        cand = [d for d in donors.get(label, []) if d[0] != sid and normalize_answer(d[1]) not in alias_set]
        if not cand:
            return None, {"reason": "no_donor", "label": label}
        order = item_rng(seed, sid, "swap").permutation(len(cand))[: int(self.hp["swap_max_tries"])]
        for j in order:
            donor_sid, donor_text = cand[int(j)]
            s2 = span[:start] + donor_text + span[end:]
            if nli(span, s2)["contradiction"] >= float(self.hp["swap_nli_contradiction_min"]):
                return s2, {"label": label, "donor": donor_sid, "entity_span": [int(start), int(end)]}
            counters["swap_nli_rejected"] += 1
        return None, {"reason": "nli_rejected_all", "label": label}

    def prepare(self, item, baseline, payload, seed, deps):
        premise = strip_citations(baseline["answer"]).strip()
        if not premise:
            deps.counters["swap_empty_premise"] += 1
            payload["swap_meta"] = {"reason": "empty_baseline_premise", "nli_premise": SWAP_NLI_PREMISE,
                                    "donor_pool_rule": DONOR_POOL_RULE, "donor_pool_planned": SWAP_DONOR_PLAN}
            return "c7_swap"

        def nli_vs_baseline(_span: str, hypothesis: str) -> Dict[str, float]:
            # Plan: contradiction of the swapped sentence against the baseline answer, not the span.
            return deps.nli(premise, hypothesis)

        swapped, meta = self.swap(payload["span"], item["aliases"], deps.donors, seed, str(item["sample_id"]),
                                  nli_vs_baseline, deps.ner, deps.counters)
        meta["nli_premise"] = SWAP_NLI_PREMISE
        meta["donor_pool_rule"] = DONOR_POOL_RULE
        meta["donor_pool_planned"] = SWAP_DONOR_PLAN
        payload["swap_meta"] = meta
        if swapped is None:
            return "c7_swap"
        payload["swap_span"] = swapped
        return None

    def edit_context(self, item, payload):
        return self._append_at_end(item, payload, payload["swap_span"])


class RandomBoundarySelfSpanProbe(ArmProbe):
    condition = SELF_RANDOM
    is_random_position = True
    position = "random"

    # ITEM 18
    @staticmethod
    def internal_boundaries(text: str, span_splitter: Optional[Callable] = None) -> List[int]:
        return [s for s, _e in sentence_spans(text, span_splitter)[1:]]

    @staticmethod
    def inject_at(text: str, b: int, plant: str) -> str:
        return text[:b].rstrip() + " " + plant + " " + text[b:].lstrip()

    @staticmethod
    def offset_bin(b: int, text_len: int, bins: int = 5) -> Tuple[float, int]:
        off = b / text_len
        return off, min(bins - 1, int(bins * off))

    @staticmethod
    def neighbor_excess(rows_c: Sequence[Dict[str, Any]], rows_null: Sequence[Dict[str, Any]], side: str
                        ) -> Optional[float]:
        return neighbor_excess(rows_c, rows_null, side)

    def prepare(self, item, baseline, payload, seed, deps):
        text = item["context_docs"][int(payload["t"]) - 1]["text"]
        bs = self.internal_boundaries(text, deps.span_splitter)
        if not bs:
            return "c5_boundary"
        b = int(bs[int(item_rng(seed, str(item["sample_id"]), "boundary").integers(len(bs)))])
        off, obin = self.offset_bin(b, len(text), int(self.hp["offset_bins"]))
        payload.update(boundary=b, offset=off, offset_bin=obin)
        return None

    def _insert(self, item: Dict[str, Any], payload: Dict[str, Any], plant: str) -> Tuple[Docs, Dict[str, Any]]:
        docs = copy.deepcopy(item["context_docs"])
        t = int(payload["t"])
        docs[t - 1]["text"] = self.inject_at(docs[t - 1]["text"], int(payload["boundary"]), plant)
        return docs, {"changed_slots": [t], "plant": plant, "offset": payload["offset"],
                      "offset_bin": payload["offset_bin"]}

    def edit_context(self, item, payload):
        return self._insert(item, payload, payload["span"])


class RandomBoundaryTopicalFoilProbe(RandomBoundarySelfSpanProbe):
    """Matched foil at the self-random arm's boundary: isolates span content from position."""
    condition = TOPICAL_RANDOM

    def prepare(self, item, baseline, payload, seed, deps):
        """Draws nothing: reuses the end arm's foil and the self-random arm's boundary."""
        if "foil" not in payload or "boundary" not in payload:
            return "c6_foil"
        target_text = item["context_docs"][int(payload["t"]) - 1]["text"]
        if payload["foil"] in target_text:
            return "c6_foil"  # inserting a sentence already present would duplicate, not plant
        return None

    def edit_context(self, item, payload):
        return self._insert(item, payload, payload["foil"])


class NullTrailingWhitespaceProbe(ArmProbe):
    """Null arm (plan item 20): null_edit, i.e. slot t text + "\\n" with everything else unchanged. No content
    is added or removed, so any migration toward t is edit-perturbation noise. prepare() only checks that the
    target slot exists; it writes nothing to the payload and adds no eligibility criterion. The random drop the
    null_random_drop prefix names is applied to the seed's rows by null_random_drop(), after every arm ran:
    null rows are kept at the treatment's observed kept rate (NULL_DROP_RULE)."""
    condition = NULL_ARM
    plant = "\n"

    def prepare(self, item, baseline, payload, seed, deps):
        t = payload.get("t")
        if t is None or not 1 <= int(t) <= self.n:
            # Unreachable: the self-span arm is prepared first and returns c5_target without a valid t.
            raise ValueError(f"null arm prepared without a target slot (t={t!r}); prepare order is broken")
        return None

    def edit_context(self, item, payload):
        t = int(payload["t"])
        orig = item["context_docs"]
        docs = null_edit(orig, t)
        if docs[t - 1]["text"] != orig[t - 1]["text"] + "\n" or docs[t - 1]["title"] != orig[t - 1]["title"]:
            raise ValueError(f"null edit on slot {t} is not the plan's text + '\\n'")
        return docs, {"changed_slots": [t], "plant": self.plant}

    def _row_meta(self, item, baseline, payload, meta, edited):
        row = super()._row_meta(item, baseline, payload, meta, edited)
        row["null_edit"] = NULL_EDIT_PLAN
        return row


class PositiveControlEvidenceRelocatedProbe(ArmProbe):
    """Moves the cited evidence passage into slot t; a gold-free donor fills its old slot.
    Donor population: POSCONTROL_DONOR_POPULATION (recorded deviation from POSCONTROL_DONOR_PLAN)."""
    condition = POSITIVE_ARM
    reliance_applicable = False
    position = "none"

    @staticmethod
    def donor_candidates(item: Dict[str, Any], pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Other items whose top passage lacks this item's gold aliases (keeps slot o evidence-free)."""
        sid, aliases = str(item["sample_id"]), item["aliases"]
        return [it for it in pool if str(it["sample_id"]) != sid
                and not alias_in_text(aliases, it["context_docs"][0]["text"])]

    def prepare(self, item, baseline, payload, seed, deps):
        payload["donor_population"] = POSCONTROL_DONOR_POPULATION
        if int(payload["t"]) == int(payload["o"]):
            return "c8_poscontrol"
        others = self.donor_candidates(item, deps.poscontrol_pool)
        if not others:
            return "c8_poscontrol"
        donor = others[int(item_rng(seed, str(item["sample_id"]), "poscontrol").integers(len(others)))]
        d = donor["context_docs"][0]
        payload["donor_passage"] = {"title": d["title"], "text": d["text"]}
        payload["donor_id"] = str(donor["sample_id"])
        return None

    def edit_context(self, item, payload):
        t, o = int(payload["t"]), int(payload["o"])
        docs = relocate_evidence(item["context_docs"], t, o, payload["donor_passage"])
        return docs, {"changed_slots": sorted([t, o]), "plant": "relocated_evidence"}

    def _row_meta(self, item, baseline, payload, meta, edited):
        row = super()._row_meta(item, baseline, payload, meta, edited)
        row.update(donor_id=payload.get("donor_id"),
                   donor_population=payload.get("donor_population", POSCONTROL_DONOR_POPULATION))
        return row


ARM_CLASSES = {
    BASELINE: BaselineUneditedArm, SELF_END: SelfSpanEndInjectionProbe, TOPICAL_END: MatchedTopicalFoilInjectionProbe,
    SWAP_END: EntitySwapFoilProbe, SELF_RANDOM: RandomBoundarySelfSpanProbe,
    TOPICAL_RANDOM: RandomBoundaryTopicalFoilProbe, NULL_ARM: NullTrailingWhitespaceProbe,
    POSITIVE_ARM: PositiveControlEvidenceRelocatedProbe}
# Order follows ELIGIBILITY_LABELS: c4/c5_target (self-span), c5_boundary, c6, c7, c8, c9. The null arm
# follows the self-span arm (it needs t) and never excludes an item.
PREPARE_ORDER = (SELF_END, NULL_ARM, SELF_RANDOM, TOPICAL_END, TOPICAL_RANDOM, SWAP_END, POSITIVE_ARM, BASELINE)


def build_arms(hp: Dict[str, Any]) -> Dict[str, ArmProbe]:
    return {c: ARM_CLASSES[c](hp) for c in CONDITION_ORDER}


# ITEM 13
def check_eligibility(item: Dict[str, Any], baseline: Optional[Dict[str, Any]], seed: int, deps: Deps,
                      quota_state: Optional[Dict[str, List[int]]] = None,
                      arms: Optional[Dict[str, ArmProbe]] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """c1..c9 in order; reads the baseline generation only (pre-treatment)."""
    payload: Dict[str, Any] = {"sample_id": str(item["sample_id"]), "redundancy": item.get("redundancy")}
    if not item.get("covered"):
        return False, "c1_uncovered", payload
    if quota_state is not None:
        cnt, cap = quota_state.get(item["redundancy"], [0, 0])
        if cnt >= cap:
            return False, "c2_quota_full", payload
    if baseline is None or not baseline.get("parse_ok"):
        return False, "c3_parse", payload
    arms = arms if arms is not None else build_arms(deps.hp)
    for cond in PREPARE_ORDER:
        label = arms[cond].prepare(item, baseline, payload, seed, deps)
        if label is not None:
            return False, label, payload
    return True, None, payload


# ITEM 24
def reliance_audit(row: Dict[str, Any], lp_fn: Callable, ctx_c: Docs, ctx_base: Docs, o: int, *, question: str,
                   hp: Dict[str, Any], counters: Optional[Counter] = None, applicable: bool = True,
                   force: bool = False) -> Dict[str, Any]:
    counters = counters if counters is not None else Counter()
    out: Dict[str, Any] = {"rel_class": None, "dp": None, "red": None, "loo_e": None, "loo_b": None}
    if not applicable or not (row.get("move") or force):
        return out
    a = row["edited_answer"]
    lp_e = float(lp_fn(question, ctx_c, a))
    lp_b = float(lp_fn(question, ctx_base, a))
    lp_eo = float(lp_fn(question, blank_passage(ctx_c, o), a))
    lp_bo = float(lp_fn(question, blank_passage(ctx_base, o), a))
    dp, loo_e, loo_b = lp_e - lp_b, lp_e - lp_eo, lp_b - lp_bo
    out.update(dp=dp, loo_e=loo_e, loo_b=loo_b)
    if loo_b <= 0:
        counters["loo_base_nonpositive"] += 1
        out["rel_class"] = "undefined"
        return out
    red = 1.0 - loo_e / loo_b
    out["red"] = red
    if dp >= float(hp["reliance_delta_plant_nats_per_token"]) and red >= float(hp["reliance_orig_drop_resourcing"]):
        out["rel_class"] = "re_sourcing"
    elif dp < float(hp["reliance_delta_plant_nats_per_token"]) and red < float(hp["reliance_orig_drop_postrat"]):
        out["rel_class"] = "post_rationalization"
    else:
        out["rel_class"] = "mixed"
    return out


def _premise(docs: Docs, ids: Sequence[int]) -> str:
    return "\n".join(docs[k - 1]["title"] + " " + docs[k - 1]["text"] for k in ids)


# ITEM 25
def alce_citation_proxy(answer: str, docs: Docs, nli_fn: Callable, splitter: Optional[Callable] = None, *,
                        threshold: float = 0.5, n_passages: int = 5) -> Dict[str, Optional[float]]:
    recalls: List[int] = []
    total, imprecise = 0, 0
    for sent in split_sentences(answer, splitter):
        claim = strip_citations(sent)
        if not claim:
            continue
        ids = sorted(parse_citations(sent, n_passages)[0])
        joint = bool(ids) and nli_fn(_premise(docs, ids), claim)["entailment"] >= threshold
        recalls.append(1 if joint else 0)
        for k in ids:
            total += 1
            if not joint:
                imprecise += 1
                continue
            if len(ids) > 1 and nli_fn(_premise(docs, [k]), claim)["entailment"] < threshold:
                rest = [j for j in ids if j != k]
                if nli_fn(_premise(docs, rest), claim)["entailment"] >= threshold:
                    imprecise += 1
    recall = None if not recalls else float(sum(recalls)) / len(recalls)
    precision = None if total == 0 else 1.0 - imprecise / total
    return {"alce_recall": recall, "alce_precision": precision}


# ITEM 22
def run_item(item: Dict[str, Any], baseline: Dict[str, Any], payload: Dict[str, Any], generate_fn: Callable,
             tf_fn: Callable, lp_fn: Callable, nli_fn: Callable, *, hp: Dict[str, Any],
             splitter: Optional[Callable] = None, counters: Optional[Counter] = None,
             arms: Optional[Dict[str, ArmProbe]] = None, force_audit: bool = False
             ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """All 8 arms on one item; rows keyed by (condition, sample_id). The null arm's seed-level random drop
    (null_random_drop) runs on the collected rows of the whole seed, not here."""
    counters = counters if counters is not None else new_counters()
    arms = arms if arms is not None else build_arms(hp)
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cond in CONDITION_ORDER:
        arm = arms[cond]
        row = arm.run(item, baseline, payload, generate_fn, tf_fn, counters)
        docs = row.pop("_docs")
        row.update(reliance_audit(row, lp_fn, docs, item["context_docs"], int(payload["o"]),
                                  question=item["question"], hp=hp, counters=counters,
                                  applicable=arm.reliance_applicable, force=force_audit))
        row.update(alce_citation_proxy(row["edited_answer"], docs, nli_fn, splitter,
                                       threshold=float(hp["alce_entail_threshold"]), n_passages=int(hp["n_passages"])))
        rows[(cond, str(item["sample_id"]))] = row
    return rows