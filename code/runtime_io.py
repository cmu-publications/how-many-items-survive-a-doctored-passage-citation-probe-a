"""
Runtime I/O for main.py: CPU scorers (bge embedder, NLI cross-encoder), a counter-replaying memo, and the
results sanitiser used before results go to the mandated experiment_harness.

Caches replay the counter increments of the original call on every hit, so fallback counters such as
ppl_too_short and nli_truncated count every call, including calls after the per-seed counter reset.
The bge embedder never truncates: an input longer than its 512-token limit raises EmbeddingTooLongError
(there is no silent fallback and no truncated embedding is ever produced).
Every object is sanitised before it is handed to the harness: non-finite floats become null and each
replaced path is listed in results["non_finite_values_replaced"] and printed.

This module writes no files. results.json is written only by experiment_harness.write_results, and the
PRIMARY line is printed only by the harness.
"""
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from baselines import NonFiniteValueError, count_truncation


class EmbeddingTooLongError(ValueError):
    """Raised instead of truncating an embedder input that exceeds the model's token limit."""


class ReplayCache:
    """Memo that stores (value, counter delta) and adds the delta to the caller's counters on every hit."""

    def __init__(self) -> None:
        self._store: Dict[Any, Tuple[Any, Counter]] = {}

    def get(self, key: Any, compute: Callable[[Counter], Any], counters: Counter) -> Any:
        if key not in self._store:
            local: Counter = Counter()
            value = compute(local)  # an exception leaves nothing cached
            self._store[key] = (value, local)
        value, delta = self._store[key]
        counters.update(delta)
        return value

    def __len__(self) -> int:
        return len(self._store)


def _memo(fn: Callable[[str, Counter], Optional[float]]) -> Callable[[str, Counter], Optional[float]]:
    cache = ReplayCache()

    def wrapped(sentence: str, counters: Counter) -> Optional[float]:
        return cache.get(sentence, lambda local: fn(sentence, local), counters)
    return wrapped


class BgeEmbedder:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    MAX_LENGTH = 512

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=True).to(torch.float32)
        self.model.train(False)  # inference mode for dropout/norm layers
        self.batch_size = int(batch_size)
        self._cache: Dict[Tuple[str, bool], np.ndarray] = {}

    def _check_lengths(self, inp: List[str]) -> None:
        """Raises EmbeddingTooLongError if any input exceeds MAX_LENGTH tokens; the embedder never truncates."""
        ids = self.tok(inp, truncation=False)["input_ids"]
        too_long = [(t, len(i)) for t, i in zip(inp, ids) if len(i) > self.MAX_LENGTH]
        if too_long:
            text, n_tok = too_long[0]
            raise EmbeddingTooLongError(
                f"{len(too_long)} embedder input(s) exceed {self.MAX_LENGTH} tokens (first: {n_tok} tokens, "
                f"{text[:80]!r}); refusing to truncate")

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        import torch
        todo = [t for t in dict.fromkeys(texts) if (t, is_query) not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            inp = [self.QUERY_PREFIX + t if is_query else t for t in batch]
            self._check_lengths(inp)
            enc = self.tok(inp, padding=True, truncation=False, return_tensors="pt")
            with torch.inference_mode():
                cls = self.model(**enc).last_hidden_state[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            for t, v in zip(batch, cls.numpy().astype(np.float64)):
                if not np.all(np.isfinite(v)):
                    raise NonFiniteValueError("non-finite embedding")
                self._cache[(t, is_query)] = v
        return np.stack([self._cache[(t, is_query)] for t in texts]) if texts else np.zeros((0, 384))


class NliScorer:
    def __init__(self, model_id: str = "cross-encoder/nli-deberta-v3-base", max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
        self.model.to(torch.float32)
        self.model.train(False)  # inference mode for dropout/norm layers
        self.labels = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        if not {"contradiction", "entailment", "neutral"} <= set(self.labels.values()):
            raise RuntimeError(f"{model_id}: unexpected NLI labels {self.labels}")
        self.max_length = int(max_length)
        self._cache = ReplayCache()

    def _score(self, premise: str, hypothesis: str, counters: Counter) -> Dict[str, float]:
        import torch
        n_tok = len(self.tok(premise, hypothesis)["input_ids"])
        count_truncation(n_tok, self.max_length, counters)
        enc = self.tok(premise, hypothesis, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits[0].float(), dim=-1).tolist()
        out = {self.labels[i]: float(p) for i, p in enumerate(probs)}
        if not all(math.isfinite(v) for v in out.values()):
            raise NonFiniteValueError("non-finite NLI probability")
        return out

    def score(self, premise: str, hypothesis: str, counters: Counter) -> Dict[str, float]:
        return self._cache.get((premise, hypothesis), lambda local: self._score(premise, hypothesis, local),
                               counters)


# ---------------------------------------------------------------------------------------
# Results sanitising (the harness writes; nothing here touches the file system)
# ---------------------------------------------------------------------------------------
def sanitize_for_json(obj: Any, root: str = "$") -> Tuple[Any, List[str], List[str]]:
    """JSON-safe copy; non-finite floats -> None (paths returned), unknown objects -> str (paths returned)."""
    non_finite: List[str] = []
    stringified: List[str] = []

    def walk(o: Any, p: str) -> Any:
        if isinstance(o, dict):
            return {str(k): walk(v, f"{p}.{k}") for k, v in o.items()}
        if isinstance(o, (set, frozenset)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(sorted(o, key=str))]
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(o)]
        if isinstance(o, np.ndarray):
            return walk(o.tolist(), p)
        if o is None or isinstance(o, str):
            return o
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            f = float(o)
            if math.isfinite(f):
                return f
            non_finite.append(p)
            return None
        stringified.append(p)
        return str(o)

    return walk(obj, root), non_finite, stringified


def finalize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitises results and records every replacement. Prints FLAG lines for non-finite values only;
    it never prints a PRIMARY or primary_metric line (the harness prints PRIMARY at write_results)."""
    clean, non_finite, stringified = sanitize_for_json(results)
    clean["non_finite_values_replaced"] = non_finite
    clean["stringified_values"] = stringified
    for p in non_finite:
        print(f"FLAG: non-finite value at {p} written as null", flush=True)
    for p in stringified:
        print(f"FLAG: non-JSON value at {p} written as its string form", flush=True)
    return clean