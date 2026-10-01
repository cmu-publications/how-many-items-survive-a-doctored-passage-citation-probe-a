"""
Reader wrappers and pure text utilities for the doctored-passage citation probe.

Baselines named by the plan: ALCE vanilla citation prompting (Gao et al., EMNLP 2023) and
the published self-span end-injection probe (Wallat et al., arXiv:2412.18004). The listed
"LSTM LM" and "DANN" are not citation-measurement methods and are not instantiated.
Nothing is trained; every forward pass runs under torch.inference_mode().

spaCy models load lazily through get_sent_nlp / get_ner_nlp (cached per model name), so
importing this module loads no model. Production sentence spans come from
spacy_sentence_spans; the regex path of split_sentences / sentence_spans serves tests only.
"""
import copy
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from data import normalize_answer

READER_ID = "Qwen/Qwen2.5-3B-Instruct"
SPACY_SENT_MODEL = "en_core_web_sm"
N_DIGIT_SLOTS = 5  # the reader scores citations over the single-token digits "1".."5"
_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_CITE_NUMBER = re.compile(r"\d+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_SPACE_PUNCT = re.compile(r"\s+([.,;:!?])")
_NLP_CACHE: Dict[str, Any] = {}

_DEFAULT_INSTRUCTION = (
    "Instruction: Write an accurate, engaging, and concise answer for the given question "
    "using only the provided search results (some of which might be irrelevant) and cite "
    "them properly. Use an unbiased and journalistic tone. Always cite for any factual "
    "claim. When citing several search results, use [1][2][3]. Cite at least one document "
    "and at most three documents in each sentence. If multiple documents support the "
    "sentence, only cite a minimum sufficient subset of the documents."
)


class NonFiniteValueError(FloatingPointError):
    """A model score (log-prob or probability) came back NaN or infinite."""


# ITEM 7
def get_sent_nlp(model_name: str = SPACY_SENT_MODEL) -> Any:
    key = "sent:" + model_name
    if key not in _NLP_CACHE:
        import spacy
        _NLP_CACHE[key] = spacy.load(model_name, disable=["ner", "lemmatizer"])
    return _NLP_CACHE[key]


def get_ner_nlp(model_name: str = SPACY_SENT_MODEL) -> Any:
    key = "ner:" + model_name
    if key not in _NLP_CACHE:
        import spacy
        _NLP_CACHE[key] = spacy.load(model_name, disable=["lemmatizer"])
    return _NLP_CACHE[key]


def get_stop_words() -> frozenset:
    from spacy.lang.en.stop_words import STOP_WORDS
    return frozenset(STOP_WORDS)


def spacy_sentence_spans(text: str, model_name: str = SPACY_SENT_MODEL) -> List[Tuple[int, int]]:
    """Character spans of the spaCy sentences of text (the production span splitter)."""
    if not text.strip():
        return []
    doc = get_sent_nlp(model_name)(text)
    spans = [(int(s.start_char), int(s.end_char)) for s in doc.sents]
    return [(s, e) for s, e in spans if text[s:e].strip()]


def split_sentences(text: str, splitter: Optional[Callable[[str], Iterable[str]]] = None) -> List[str]:
    """Sentence split. main.py always passes the spaCy splitter; the regex path serves tests."""
    text = text.strip()
    if not text:
        return []
    if splitter is not None:
        return [s.strip() for s in splitter(text) if s.strip()]
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def sentence_spans(text: str, span_splitter: Optional[Callable[[str], Iterable[Tuple[int, int]]]] = None
                   ) -> List[Tuple[int, int]]:
    """Sentence spans. Production passes spacy_sentence_spans; the regex path serves tests only."""
    if span_splitter is not None:
        return [(int(s), int(e)) for s, e in span_splitter(text)]
    spans, start = [], 0
    for m in _SENT_RE.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def strip_citations(text: str) -> str:
    t = re.sub(r"\s+", " ", _CITE_GROUP.sub("", text)).strip()
    return _SPACE_PUNCT.sub(r"\1", t)


# ITEM 11
def parse_citations(answer: str, n_passages: int = 5) -> Tuple[Set[int], bool]:
    cited: Set[int] = set()
    groups = _CITE_GROUP.findall(answer)
    ok = len(groups) > 0
    for g in groups:
        for tok in g.split(","):
            k = int(tok.strip())
            if 1 <= k <= n_passages:
                cited.add(k)
            else:
                ok = False
    return cited, ok


def answer_alias_set(answer: str, aliases: Sequence[str]) -> frozenset:
    t = " " + normalize_answer(strip_citations(answer)) + " "
    return frozenset(a for a in aliases if (" " + a + " ") in t)


# ITEM 12
def extract_self_span(answer: str, aliases: Sequence[str],
                      splitter: Optional[Callable[[str], Iterable[str]]] = None) -> Tuple[Optional[str], List[int]]:
    """First answer sentence holding a gold alias -> (markers stripped, ids cited in it)."""
    for sent in split_sentences(answer, splitter):
        clean = strip_citations(sent)
        if clean and answer_alias_set(clean, aliases):
            ids, _ok = parse_citations(sent)
            return clean, sorted(ids)
    return None, []


def score_outcome(baseline: Dict[str, Any], edited_answer: str, t: int, aliases: Sequence[str],
                  n_passages: int = 5) -> Dict[str, Any]:
    cited, ok = parse_citations(edited_answer, n_passages)
    base = set(baseline["cited"])
    kept = answer_alias_set(edited_answer, aliases) == frozenset(baseline["alias_set"])
    t_new = bool(ok and t in cited and t not in base)
    nxt, prv = t + 1, t - 1
    return {
        "parse_ok": bool(ok), "target_cited": t_new, "answer_kept": bool(kept),
        "move": bool(t_new and kept), "answer_flip": not kept,
        "next_exists": nxt <= n_passages, "prev_exists": prv >= 1,
        "neighbor_next_new": bool(ok and nxt <= n_passages and nxt in cited and nxt not in base),
        "neighbor_prev_new": bool(ok and prv >= 1 and prv in cited and prv not in base),
        "cited": sorted(cited),
    }


def digit_logprobs_from_logits(logit_rows: Any, digit_ids: Sequence[int]) -> List[List[float]]:
    """fp32 log_softmax restricted to the digit ids: the other vocabulary entries never enter the normaliser."""
    arr = np.asarray(logit_rows, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim != 2:
        raise ValueError(f"expected a [slots, vocab] logit matrix, got shape {arr.shape}")
    sel = arr[:, [int(i) for i in digit_ids]]
    if not bool(np.isfinite(sel).all()):
        raise NonFiniteValueError(f"non-finite citation digit logit {sel.tolist()}")
    shifted = sel - sel.max(axis=1, keepdims=True)
    lp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True, dtype=np.float32))
    if not bool(np.isfinite(lp).all()):
        raise NonFiniteValueError(f"non-finite citation log-prob {lp.tolist()}")
    return lp.astype(np.float32).tolist()


def slots_from_logprobs(digit_logp: List[List[float]], cited_digits: List[int]) -> List[Dict[str, Any]]:
    slots = []
    for row, d in zip(digit_logp, cited_digits):
        vals = [float(v) for v in row]
        if not all(math.isfinite(v) for v in vals):
            raise NonFiniteValueError(f"non-finite citation log-prob {vals}")
        srt = sorted(vals, reverse=True)
        slots.append({"cited": int(d), "logp": vals, "margin": srt[0] - srt[1]})
    return slots


def citation_positions(answer: str, prompt_len: int, offsets: List[Tuple[int, int]],
                       counters: Counter) -> Tuple[List[int], List[int]]:
    """Positions preceding each cited digit, in answer order, for [d] and comma groups [d, e].

    Digits merged into a neighbouring token increment digit_merged_slot_skipped; ids that are
    not a single digit in 1..N_DIGIT_SLOTS have no digit-token slot and increment
    citation_out_of_range_slot_skipped.
    """
    start_of = {int(s): i for i, (s, e) in enumerate(offsets) if e > s}
    positions, digits = [], []
    for g in _CITE_GROUP.finditer(answer):
        group_start = g.start(1)
        for n in _CITE_NUMBER.finditer(g.group(1)):
            text = n.group(0)
            k = int(text)
            if len(text) != 1 or not 1 <= k <= N_DIGIT_SLOTS:
                counters["citation_out_of_range_slot_skipped"] += 1
                continue
            cp = prompt_len + group_start + n.start()
            ti = start_of.get(cp)
            if ti is None or ti == 0 or int(offsets[ti][1]) != cp + 1:
                counters["digit_merged_slot_skipped"] += 1
                continue
            positions.append(ti - 1)
            digits.append(k)
    return positions, digits


def mean_nll_or_none(token_logprobs: List[float], counters: Counter) -> Optional[float]:
    """Mean NLL of tokens 2..n; fewer than 2 tokens (no scored token) gives None."""
    if len(token_logprobs) < 1:
        counters["ppl_too_short"] += 1
        return None
    vals = [float(v) for v in token_logprobs]
    if not all(math.isfinite(v) for v in vals):
        raise NonFiniteValueError("non-finite sentence log-prob")
    return -sum(vals) / len(vals)


def count_truncation(n_tokens: int, max_len: int, counters: Counter) -> bool:
    if n_tokens > max_len:
        counters["nli_truncated"] += 1
        return True
    return False


# ITEM 8
def select_dtype(capability: Optional[Tuple[int, int]]) -> str:
    if capability is None:
        return "float32"
    return "bfloat16" if int(capability[0]) >= 8 else "float16"


# ITEM 10
def blank_passage(docs: Sequence[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    new_docs = copy.deepcopy(list(docs))
    new_docs[k - 1] = {"title": "", "text": ""}
    return new_docs


def format_docs(docs: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"Document [{i + 1}](Title: {d['title']}): {d['text']}" for i, d in enumerate(docs))


def build_alce_user_prompt(question: str, docs: Sequence[Dict[str, str]], demos: Sequence[Dict[str, Any]],
                           instruction: str = "", n_demos: int = 2) -> str:
    parts = [instruction or _DEFAULT_INSTRUCTION, ""]
    for demo in list(demos)[:n_demos]:
        parts.append(format_docs(demo.get("docs", [])[:5]))
        parts.append(f"\nQuestion: {demo['question']}\n\nAnswer: {demo['answer']}\n\n")
    parts.append(format_docs(docs))
    parts.append(f"\nQuestion: {question}\n\nAnswer:")
    return "\n".join(parts)


def _dtype_kwargs(dtype: Any) -> Dict[str, Any]:
    import transformers
    major = int(str(transformers.__version__).split(".", maxsplit=1)[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


class SharedReader:
    """Loads the reader once, offline; fp16 on Turing because bf16 needs capability >= 8."""

    def __init__(self, model_id: str = READER_ID, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        cap = torch.cuda.get_device_capability(0) if use_cuda else None
        self.dtype_name = select_dtype(cap)
        dtype = getattr(torch, self.dtype_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if not self._tokenizer.is_fast:
            raise RuntimeError(f"{model_id}: a fast tokenizer is required for offset mapping")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, **_dtype_kwargs(dtype))
        self.model.to(self.device)
        self.model.train(False)  # inference mode for dropout/norm layers
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.digit_ids: List[int] = []
        for d in "12345"[:N_DIGIT_SLOTS]:
            ids = self._tokenizer.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Digit '{d}' is not a single token for {model_id}")
            self.digit_ids.append(int(ids[0]))

    def tokenizer(self) -> Any:
        """The fast tokenizer loaded with the reader."""
        return self._tokenizer

    def hidden_states(self, input_ids: Any) -> Any:
        import torch
        with torch.inference_mode():
            return self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state

    def logits_at(self, hidden: Any, positions: Any) -> Any:
        import torch
        with torch.inference_mode():
            return self.model.get_output_embeddings()(hidden[0, positions]).float()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class AlceVanillaCitationRunner:
    """ALCE prompting, greedy generation, memory-safe teacher-forced log-probs."""

    def __init__(self, reader: SharedReader, demos: Sequence[Dict[str, Any]] = (), instruction: str = "",
                 max_new_tokens: int = 300, n_passages: int = 5, n_demos: int = 2, decoding: str = "greedy"):
        if decoding != "greedy":
            raise ValueError(f"only greedy decoding is implemented, got {decoding}")
        self.reader = reader
        self.demos = list(demos)[:n_demos]
        self.instruction = instruction
        self.max_new_tokens = int(max_new_tokens)
        self.n_passages = int(n_passages)
        self.n_demos = int(n_demos)

    def build_prompt(self, question: str, docs: Sequence[Dict[str, str]]) -> str:
        user = build_alce_user_prompt(question, docs, self.demos, self.instruction, self.n_demos)
        return self.reader.tokenizer().apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, docs: Sequence[Dict[str, str]]) -> str:
        import torch
        from transformers import GenerationConfig
        tok = self.reader.tokenizer()
        enc = tok(self.build_prompt(question, docs), return_tensors="pt", add_special_tokens=False).to(self.reader.device)
        gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=self.max_new_tokens,
                                   pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                   eos_token_id=self.reader.model.generation_config.eos_token_id)
        with torch.inference_mode():
            out = self.reader.model.generate(**enc, generation_config=gen_cfg)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ITEM 9
    def teacher_forced_citation_logprobs(self, question: str, docs: Sequence[Dict[str, str]], answer: str,
                                         counters: Optional[Counter] = None) -> List[Dict[str, Any]]:
        import torch
        counters = counters if counters is not None else Counter()
        tok = self.reader.tokenizer()
        prompt = self.build_prompt(question, docs)
        enc = tok(prompt + answer, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        positions, digits = citation_positions(answer, len(prompt), offsets, counters)
        if not positions:
            return []
        ids = enc["input_ids"].to(self.reader.device)
        h = self.reader.hidden_states(ids)                                   # [1, L+A, H]
        pos = torch.tensor(positions, device=self.reader.device)
        logits = self.reader.logits_at(h, pos)                               # lm_head at S slots only, fp32
        lp = digit_logprobs_from_logits(logits.cpu().numpy(), self.reader.digit_ids)   # [S, 5]
        return slots_from_logprobs(lp, digits)

    def answer_logprob_per_token(self, question: str, docs: Sequence[Dict[str, str]], answer: str) -> float:
        import torch
        tok = self.reader.tokenizer()
        p_ids = tok(self.build_prompt(question, docs), add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        if not a_ids:
            raise ValueError("empty answer passed to answer_logprob_per_token")
        ids = torch.tensor([p_ids + a_ids], device=self.reader.device)
        h = self.reader.hidden_states(ids)
        pos = torch.arange(len(p_ids) - 1, len(p_ids) + len(a_ids) - 1, device=self.reader.device)
        with torch.inference_mode():
            lp = torch.log_softmax(self.reader.logits_at(h, pos), dim=-1)                # [A, V] fp32
            tgt = torch.tensor(a_ids, device=self.reader.device).unsqueeze(1)
            tok_lp = lp.gather(1, tgt).squeeze(1)
        if not bool(torch.isfinite(tok_lp).all()):
            raise NonFiniteValueError("non-finite answer log-prob")
        return float(tok_lp.mean())

    def sentence_nll(self, sentence: str, counters: Optional[Counter] = None) -> Optional[float]:
        import torch
        counters = counters if counters is not None else Counter()
        ids = self.reader.tokenizer()(sentence, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            return mean_nll_or_none([], counters)
        t_ids = torch.tensor([ids], device=self.reader.device)
        h = self.reader.hidden_states(t_ids)
        pos = torch.arange(0, len(ids) - 1, device=self.reader.device)
        with torch.inference_mode():
            lp = torch.log_softmax(self.reader.logits_at(h, pos), dim=-1)
            tgt = torch.tensor(ids[1:], device=self.reader.device).unsqueeze(1)
            tok_lp = lp.gather(1, tgt).squeeze(1).cpu().tolist()
        return mean_nll_or_none(tok_lp, counters)

    def token_len(self, text: str) -> int:
        return len(self.reader.tokenizer()(text, add_special_tokens=False)["input_ids"])