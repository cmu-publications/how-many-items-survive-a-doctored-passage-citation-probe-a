"""
Real model-backed dependencies (reader, embedder, NLI, spaCy) and the backend/provenance record.
Imported by main.py after the CUDA/HF-offline environment switches are set.
"""
import os
import sys
from importlib import metadata
from typing import Any, Dict, List, Tuple

from data import DATA_CONFIG
from probes import Deps
from runtime_io import BgeEmbedder, NliScorer, _memo


def build_real_deps(hp: Dict[str, Any], datasets: Dict[str, Any]) -> Tuple[Deps, Any]:
    from baselines import AlceVanillaCitationRunner, SharedReader, get_ner_nlp, get_sent_nlp, get_stop_words
    if hp["decoding"] != "greedy" or int(hp["batch_size"]) != 1:
        raise ValueError(f"the runner decodes greedily with batch 1; got decoding={hp['decoding']} "
                         f"batch_size={hp['batch_size']}")
    reader = SharedReader(hp["reader_model_id"])
    runner = AlceVanillaCitationRunner(reader, demos=datasets["train"].items,
                                       instruction=DATA_CONFIG.get("instruction", ""),
                                       max_new_tokens=int(hp["max_new_tokens"]), n_passages=int(hp["n_passages"]),
                                       n_demos=int(hp["n_demos"]))
    embedder = BgeEmbedder(hp["embed_model_id"])
    nli = NliScorer(hp["nli_model_id"], max_length=int(hp["nli_max_length"]))
    sent_nlp, ner_nlp = get_sent_nlp(hp["spacy_model"]), get_ner_nlp(hp["spacy_model"])

    def splitter(text: str) -> List[str]:
        return [s.text for s in sent_nlp(text).sents]

    def span_splitter(text: str) -> List[Tuple[int, int]]:
        return [(s.start_char, s.end_char) for s in sent_nlp(text).sents]

    def ner(text: str) -> List[Tuple[int, int, str, str]]:
        return [(e.start_char, e.end_char, e.label_, e.text) for e in ner_nlp(text).ents]

    deps = Deps(hp, generate=runner.generate, tf_slots=runner.teacher_forced_citation_logprobs,
                answer_lp=runner.answer_logprob_per_token, sent_nll=_memo(runner.sentence_nll),
                token_len=runner.token_len, embed=embedder.embed, nli=nli.score, splitter=splitter,
                span_splitter=span_splitter, ner=ner, stop_words=get_stop_words())
    return deps, reader


def spacy_ner_labels(model_name: str) -> List[str]:
    from baselines import get_ner_nlp
    return list(get_ner_nlp(model_name).get_pipe("ner").labels)


def backend_info(hp: Dict[str, Any], reader_dtype: str, cuda_ok: bool, device_name: str) -> Dict[str, Any]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "numpy", "spacy", "scipy", "statsmodels", "en_core_web_sm"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {"cuda_available": cuda_ok, "device": device_name, "reader_dtype": reader_dtype,
            "models": {k: hp[k] for k in ("reader_model_id", "embed_model_id", "nli_model_id", "spacy_model")},
            "decoding": hp["decoding"], "batch_size": hp["batch_size"], "nli_max_length": hp["nli_max_length"],
            "versions": versions, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "hf_offline": os.environ.get("HF_HUB_OFFLINE")}