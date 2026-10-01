"""BgeEmbedder must never truncate silently: an over-long input raises, and the tokenizer is never asked to
truncate. Uses a fake tokenizer and model (no model load)."""
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from runtime_io import BgeEmbedder, EmbeddingTooLongError


class _FakeTok:
    """Whitespace tokenizer that records every call's keyword arguments."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, texts: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        ids = [[101] + [7] * len(t.split()) + [102] for t in texts]
        if kwargs.get("truncation") and kwargs.get("max_length"):
            ids = [i[: kwargs["max_length"]] for i in ids]
        if kwargs.get("return_tensors") == "pt":
            width = max(len(i) for i in ids)
            padded = [i + [0] * (width - len(i)) for i in ids]
            mask = [[1] * len(i) + [0] * (width - len(i)) for i in ids]
            return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids}


class _FakeModel:
    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        self.n_calls += 1
        b, length = input_ids.shape
        hidden = torch.ones(b, length, 384) * (input_ids[:, :1, None].float() + 1.0)
        return SimpleNamespace(last_hidden_state=hidden)


def _embedder() -> BgeEmbedder:
    emb = BgeEmbedder.__new__(BgeEmbedder)
    emb.tok = _FakeTok()
    emb.model = _FakeModel()
    emb.batch_size = 4
    emb._cache = {}
    return emb


def test_overlong_input_raises_instead_of_truncating():
    emb = _embedder()
    long_text = " ".join(["word"] * (BgeEmbedder.MAX_LENGTH + 10))
    with pytest.raises(EmbeddingTooLongError, match="refusing to truncate"):
        emb.embed(["short sentence.", long_text])
    assert emb.model.n_calls == 0
    assert emb._cache == {}


def test_tokenizer_is_never_asked_to_truncate():
    emb = _embedder()
    emb.embed(["a short sentence.", "another one here."])
    emb.embed(["what is it?"], is_query=True)
    assert emb.tok.calls
    assert all(not c.get("truncation", False) for c in emb.tok.calls)


def test_input_at_the_limit_is_embedded_unit_norm():
    emb = _embedder()
    exact = " ".join(["w"] * (BgeEmbedder.MAX_LENGTH - 2))  # plus two special tokens = MAX_LENGTH
    out = emb.embed([exact, "short."])
    assert out.shape == (2, 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_query_prefix_counts_toward_the_limit():
    emb = _embedder()
    n_prefix = len(BgeEmbedder.QUERY_PREFIX.split())
    text = " ".join(["w"] * (BgeEmbedder.MAX_LENGTH - 2 - n_prefix + 1))
    emb.embed([text], is_query=False)
    with pytest.raises(EmbeddingTooLongError):
        emb.embed([text], is_query=True)


def test_too_long_error_is_a_value_error():
    assert issubclass(EmbeddingTooLongError, ValueError)