"""
One-time setup, the only step allowed to use the network: installs dependencies, downloads the ALCE ASQA eval
file and the model weights, and checks the spaCy pipeline.

Why this version exists: spaCy 3.7 (and its blis/thinc stack) has no wheels for Python 3.14, so pip fell back
to building spaCy -> thinc -> blis from source and the Cython build of blis failed; the smoke run never started.
This script therefore
  * pins the spaCy 3.8 series and its matching pipeline en_core_web_sm 3.8.0 (3.7.1 requires spaCy < 3.8), and
  * installs every compiled dependency from prebuilt wheels only (--only-binary), so a missing wheel stops setup
    within seconds with pip's own message instead of attempting a source build.
Model weights go to the default Hugging Face cache (no cache_dir / local_dir is ever passed). Resolved model
revisions, the dataset revision, file sha256 digests and library versions are written to the data manifest.

Usage:  python setup.py
"""
import hashlib
import importlib
import importlib.util
import json
import os
import site
import sys
import tarfile
import urllib.request
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

DATASET_ID = "princeton-nlp/ALCE-data"
EVAL_FILE = "asqa_eval_gtr_top100.json"
DATA_ROOT = os.path.join(".", "data")
EXPECTED_ITEMS = 948
SPACY_SERIES = "3.8."
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.8.0"
SPACY_MODEL_URL = ("https://github.com/explosion/spacy-models/releases/download/"
                   f"{SPACY_MODEL}-{SPACY_MODEL_VERSION}/{SPACY_MODEL}-{SPACY_MODEL_VERSION}-py3-none-any.whl")
MANIFEST_NAME = "alce_asqa_manifest.json"
DOWNLOAD_TIMEOUT_SEC = 600
# Compiled packages are taken from wheels only: a source build of this stack is what failed on Python 3.14.
BINARY_ONLY = ("spacy", "thinc", "blis", "cymem", "preshed", "murmurhash", "srsly", "numpy", "torch",
               "tokenizers", "safetensors")
# import name -> (distribution name, pip requirement)
DEPENDENCIES: Dict[str, Tuple[str, str]] = {
    "numpy": ("numpy", "numpy"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
    "huggingface_hub": ("huggingface_hub", "huggingface_hub"),
    "sentencepiece": ("sentencepiece", "sentencepiece"),
    "spacy": ("spacy", "spacy>=3.8,<3.9"),
}
# Same ids as main.HYPERPARAMETERS (main is not imported here: it needs torch, which may not be installed yet).
MODEL_IDS = ("Qwen/Qwen2.5-3B-Instruct", "BAAI/bge-small-en-v1.5", "cross-encoder/nli-deberta-v3-base")


class DependencyMissingError(ModuleNotFoundError):
    """A required package is not importable after installation."""


class DependencyVersionError(RuntimeError):
    """A required package is installed at a version outside the pinned series."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: str, data: bytes) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _refresh_import_paths() -> None:
    """Makes packages pip just installed (possibly into the user site) importable in this process."""
    user_site = site.getusersitepackages()
    if isinstance(user_site, str) and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.append(user_site)
    importlib.invalidate_caches()


def _pip_install(requirements: List[str]) -> None:
    """Runs pip's command-line entry point in process (no shell). Compiled packages are wheel-only."""
    if not requirements:
        return
    from pip._internal.cli.main import main as pip_main

    args = ["install", "--disable-pip-version-check", f"--only-binary={','.join(BINARY_ONLY)}", *requirements]
    print(f"setup: pip {' '.join(args)}", flush=True)
    code = pip_main(args)
    if code != 0:
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise DependencyMissingError(
            f"pip install failed (exit {code}) for {requirements}; {', '.join(BINARY_ONLY)} are installed from "
            f"wheels only, so the likely cause is that no wheel exists for Python {py} on this platform")
    _refresh_import_paths()


def installed_version(dist: str) -> Optional[str]:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def version_ok(dist: str, version: Optional[str]) -> bool:
    if version is None:
        return False
    if dist == "spacy":
        return version.startswith(SPACY_SERIES)
    if dist == SPACY_MODEL:
        return version == SPACY_MODEL_VERSION
    return True


def _needs_install(mod: str) -> bool:
    return importlib.util.find_spec(mod) is None


def ensure_dependencies() -> Dict[str, str]:
    """Installs missing or off-series packages, then checks every one; returns dist -> installed version."""
    todo = [req for mod, (dist, req) in DEPENDENCIES.items()
            if _needs_install(mod) or not version_ok(dist, installed_version(dist))]
    _pip_install(todo)
    versions: Dict[str, str] = {}
    for mod, (dist, req) in DEPENDENCIES.items():
        if _needs_install(mod):
            raise DependencyMissingError(f"{mod} is not importable after installing {req}")
        version = installed_version(dist)
        if not version_ok(dist, version):
            raise DependencyVersionError(f"{dist}=={version} is outside the pinned requirement {req}")
        versions[dist] = str(version)
    print(f"setup: dependencies {versions}", flush=True)
    return versions


def ensure_spacy_model() -> Dict[str, Any]:
    """Installs en_core_web_sm 3.8.0 from its release wheel when absent or at another version."""
    version = installed_version(SPACY_MODEL)
    if version_ok(SPACY_MODEL, version):
        return {"model": SPACY_MODEL, "version": str(version), "url": None, "sha256": None,
                "note": "already installed"}
    print(f"setup: installing {SPACY_MODEL} {SPACY_MODEL_VERSION} (found {version})", flush=True)
    with urllib.request.urlopen(SPACY_MODEL_URL, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:  # nosec B310 fixed https
        data = resp.read()
    wheel_path = os.path.join(DATA_ROOT, "_wheels", os.path.basename(SPACY_MODEL_URL))
    _write_bytes(wheel_path, data)
    _pip_install(["--no-deps", "--force-reinstall", wheel_path])
    version = installed_version(SPACY_MODEL)
    if not version_ok(SPACY_MODEL, version):
        raise DependencyVersionError(f"{SPACY_MODEL}=={version} after install; expected {SPACY_MODEL_VERSION}")
    return {"model": SPACY_MODEL, "version": str(version), "url": SPACY_MODEL_URL, "sha256": _sha256_bytes(data)}


def download_models() -> Dict[str, str]:
    """Fetches each model into the default Hugging Face cache; returns model id -> resolved commit hash."""
    from huggingface_hub import snapshot_download

    revisions: Dict[str, str] = {}
    for model_id in MODEL_IDS:
        path = snapshot_download(repo_id=model_id)
        revisions[model_id] = os.path.basename(os.path.normpath(path))  # snapshots/<commit>
        print(f"setup: model {model_id} revision {revisions[model_id]}", flush=True)
    return revisions


def _extract_eval(snapshot_dir: str, out_path: str) -> str:
    """Copies EVAL_FILE out of the dataset snapshot (plain file or inside a tar archive) to out_path.
    Archive members are read by name and written to out_path only (no extractall, so no path traversal)."""
    for root, _dirs, files in os.walk(snapshot_dir):
        if EVAL_FILE in files:
            with open(os.path.join(root, EVAL_FILE), "rb") as fh:
                data = fh.read()
            _write_bytes(out_path, data)
            return _sha256_bytes(data)
    for root, _dirs, files in os.walk(snapshot_dir):
        for name in sorted(files):
            if not (name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz")):
                continue
            with tarfile.open(os.path.join(root, name)) as archive:
                for member in archive.getmembers():
                    if member.isfile() and os.path.basename(member.name) == EVAL_FILE:
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise FileNotFoundError(f"{member.name} in {name} could not be read")
                        data = handle.read()
                        _write_bytes(out_path, data)
                        return _sha256_bytes(data)
    raise FileNotFoundError(f"{EVAL_FILE} not found in dataset snapshot {snapshot_dir}")


def download_data(data_root: str = DATA_ROOT) -> Dict[str, Any]:
    """Places EVAL_FILE under data_root, checks it holds EXPECTED_ITEMS items, and returns its provenance."""
    from huggingface_hub import snapshot_download

    out_path = os.path.join(data_root, EVAL_FILE)
    snap = snapshot_download(repo_id=DATASET_ID, repo_type="dataset",
                             allow_patterns=[EVAL_FILE, f"*/{EVAL_FILE}", "*.tar", "*.tar.gz", "*.tgz"])
    revision = os.path.basename(os.path.normpath(snap))
    digest = _extract_eval(snap, out_path)
    with open(out_path, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list) or len(items) != EXPECTED_ITEMS:
        n = len(items) if isinstance(items, list) else type(items).__name__
        raise ValueError(f"{out_path} holds {n} items; expected a list of {EXPECTED_ITEMS}")
    print(f"setup: data {out_path} items={len(items)} sha256={digest} revision={revision}", flush=True)
    return {"dataset_id": DATASET_ID, "revision": revision, "file": EVAL_FILE, "path": out_path,
            "sha256": digest, "n_items": len(items), "expected_items": EXPECTED_ITEMS}


def verify_spacy() -> str:
    """Loads the pipeline and checks it yields sentences and entities; returns the spaCy version."""
    spacy = importlib.import_module("spacy")
    nlp = spacy.load(SPACY_MODEL)
    doc = nlp("Dr. Smith visited Paris in 1999. He stayed for two weeks.")
    if len(list(doc.sents)) < 1 or len(doc.ents) < 1:
        raise RuntimeError(f"{SPACY_MODEL} produced no sentences or entities on the check sentence")
    version = str(spacy.__version__)
    if not version.startswith(SPACY_SERIES):
        raise DependencyVersionError(f"spaCy {version} loaded; expected the {SPACY_SERIES}x series")
    print(f"setup: spaCy {version} with {SPACY_MODEL} ok", flush=True)
    return version


def main() -> Dict[str, Any]:
    versions = ensure_dependencies()
    spacy_model = ensure_spacy_model()
    spacy_version = verify_spacy()
    models = download_models()
    data = download_data(DATA_ROOT)
    manifest = {**data, "python": sys.version.split()[0], "library_versions": versions,
                "spacy_version": spacy_version, "spacy_model": spacy_model, "model_revisions": models}
    _write_bytes(os.path.join(DATA_ROOT, MANIFEST_NAME), json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"setup: manifest written to {os.path.join(DATA_ROOT, MANIFEST_NAME)}", flush=True)
    return manifest


if __name__ == "__main__":
    main()