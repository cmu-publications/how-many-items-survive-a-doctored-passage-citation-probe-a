"""
Numpy-only binomial GLM and GLMM for analysis.py (F4: deviance drop and the span x position GLMM).

statsmodels, pandas and scipy are not dependencies of this project, so the models are fitted here:
- design_matrix builds treatment-coded columns with patsy-style names (first sorted level = reference).
- drop_aliased removes columns that are linear combinations of earlier ones; the dropped names are returned
  so the caller records them (nothing is dropped silently).
- fit_logit: binomial GLM (logit link) by IRLS.
- fit_binomial_glmm: binomial GLMM with random intercepts per grouping factor (item, seed). For each pair of
  random-effect SDs on the fixed grid SD_GRID (pre-registered here, not tuned on data) the fixed and random
  effects are fitted by penalized IRLS; the pair with the highest Laplace-approximate marginal log-likelihood is
  kept. Fixed-effect SDs come from the inverse penalized Hessian at that fit.
The linear predictor is clipped to +-ETA_CLIP so separated data give finite (recorded, possibly non-converged)
estimates rather than NaN. Every fit reports converged and n_iter.
"""
import itertools
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

TERMS_FULL = ("span", "position", "span:position", "slot", "redundancy", "tercile")
TERMS_NO_SLOT = ("span", "position", "span:position", "redundancy", "tercile")
TERMS_NO_SPAN = ("position", "slot", "redundancy", "tercile")
ETA_CLIP = 30.0
MIN_WEIGHT = 1e-12
MAX_ITER = 100
TOL = 1e-10
SD_GRID = (0.1, 0.3, 0.7, 1.5, 3.0)
GLM_METHOD = "binomial GLM (logit link), IRLS in numpy; aliased columns dropped and recorded"
GLMM_METHOD = ("binomial GLMM (logit link), random intercepts for item and seed, penalized IRLS in numpy; "
               f"random-effect SDs chosen on the fixed grid {SD_GRID} by Laplace-approximate marginal likelihood")
_LABELS = {"span": "span", "position": "position", "slot": "C(slot)", "redundancy": "redundancy",
           "tercile": "C(tercile)"}


class GlmFitError(ValueError):
    """A model fit produced a non-finite or non-positive-definite quantity."""


def _dummies(recs: Sequence[Dict[str, Any]], key: str) -> List[Tuple[str, np.ndarray]]:
    vals = [r[key] for r in recs]
    levels = sorted(set(vals))
    label = _LABELS[key]
    return [(f"{label}[T.{lv}]", np.array([1.0 if v == lv else 0.0 for v in vals])) for lv in levels[1:]]


def design_matrix(recs: Sequence[Dict[str, Any]], terms: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    cols: List[Tuple[str, np.ndarray]] = [("Intercept", np.ones(len(recs)))]
    main: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for term in terms:
        if ":" in term:
            a, b = term.split(":", 1)
            if a not in main or b not in main:
                raise ValueError(f"interaction {term} listed before its main effects")
            cols.extend((f"{na}:{nb}", ca * cb) for na, ca in main[a] for nb, cb in main[b])
        else:
            main[term] = _dummies(recs, term)
            cols.extend(main[term])
    return np.column_stack([c for _n, c in cols]), [nm for nm, _c in cols]


def drop_aliased(X: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Keeps each column only if it raises the rank of the kept set; returns (X, kept names, dropped names)."""
    keep: List[int] = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        if int(np.linalg.matrix_rank(X[:, cand])) == len(cand):
            keep.append(j)
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    return X[:, keep], [names[j] for j in keep], dropped


def _expit(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _loglik(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logit(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    beta = np.zeros(X.shape[1])
    eta = np.zeros(len(y))
    dev_old, dev = float("inf"), float("inf")
    converged, n_iter = False, 0
    for n_iter in range(1, MAX_ITER + 1):
        mu = _expit(eta)
        w = np.maximum(mu * (1.0 - mu), MIN_WEIGHT)
        z = eta + (y - mu) / w
        xtw = X.T * w
        beta = np.linalg.solve(xtw @ X, xtw @ z)
        eta = np.clip(X @ beta, -ETA_CLIP, ETA_CLIP)
        dev = -2.0 * _loglik(y, eta)
        if abs(dev_old - dev) <= TOL * (abs(dev) + 1.0):
            converged = True
            break
        dev_old = dev
    if not np.isfinite(dev) or not bool(np.all(np.isfinite(beta))):
        raise GlmFitError("non-finite GLM estimate or deviance")
    return {"beta": beta, "deviance": float(dev), "converged": converged, "n_iter": n_iter}


def _group_matrix(groups: Dict[str, Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
    blocks, owner = [], []
    for gname, labels in groups.items():
        levels = sorted(set(labels))
        idx = {lv: i for i, lv in enumerate(levels)}
        Zg = np.zeros((len(labels), len(levels)))
        for r, lab in enumerate(labels):
            Zg[r, idx[lab]] = 1.0
        blocks.append(Zg)
        owner.extend([gname] * len(levels))
    return np.hstack(blocks), owner


def _pirls(M: np.ndarray, y: np.ndarray, pen: np.ndarray, theta0: np.ndarray
           ) -> Tuple[np.ndarray, float, np.ndarray, bool, int]:
    theta = theta0.copy()
    eta = np.clip(M @ theta, -ETA_CLIP, ETA_CLIP)
    obj_old, obj = float("inf"), float("inf")
    converged, n_iter = False, 0
    P = np.diag(pen)
    for n_iter in range(1, MAX_ITER + 1):
        mu = _expit(eta)
        w = np.maximum(mu * (1.0 - mu), MIN_WEIGHT)
        z = eta + (y - mu) / w
        mtw = M.T * w
        theta = np.linalg.solve(mtw @ M + P, mtw @ z)
        eta = np.clip(M @ theta, -ETA_CLIP, ETA_CLIP)
        obj = -2.0 * _loglik(y, eta) + float(theta @ (pen * theta))
        if abs(obj_old - obj) <= TOL * (abs(obj) + 1.0):
            converged = True
            break
        obj_old = obj
    if not np.isfinite(obj) or not bool(np.all(np.isfinite(theta))):
        raise GlmFitError("non-finite GLMM estimate")
    mu = _expit(eta)
    w = np.maximum(mu * (1.0 - mu), MIN_WEIGHT)
    H = (M.T * w) @ M + P
    return theta, _loglik(y, eta), H, converged, n_iter


def fit_binomial_glmm(X: np.ndarray, y: np.ndarray, groups: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    Z, owner = _group_matrix(groups)
    M = np.hstack([X, Z])
    p = X.shape[1]
    gnames = list(groups)
    theta = np.zeros(M.shape[1])
    best: Dict[str, Any] = {}
    grid: List[Dict[str, Any]] = []
    for sds in itertools.product(SD_GRID, repeat=len(gnames)):
        sd_of = dict(zip(gnames, sds))
        ginv = np.array([1.0 / sd_of[g] ** 2 for g in owner])
        pen = np.concatenate([np.zeros(p), ginv])
        theta, ll, H, conv, n_iter = _pirls(M, y, pen, theta)
        sign, logdet = np.linalg.slogdet(H[p:, p:])
        if sign <= 0 or not np.isfinite(logdet):
            raise GlmFitError("random-effect Hessian is not positive definite")
        u = theta[p:]
        lap = ll - 0.5 * float(u @ (ginv * u)) + 0.5 * float(np.sum(np.log(ginv))) - 0.5 * float(logdet)
        grid.append({"sd": dict(sd_of), "laplace_loglik": lap, "converged": conv})
        if not best or lap > best["laplace_loglik"]:
            best = {"theta": theta.copy(), "H": H, "sd": dict(sd_of), "laplace_loglik": lap,
                    "converged": conv, "n_iter": n_iter}
    cov = np.linalg.inv(best["H"])
    var = np.diag(cov)[:p]
    if not bool(np.all(np.isfinite(var))) or bool(np.any(var <= 0)):
        raise GlmFitError("non-positive or non-finite fixed-effect variance")
    return {"fe_mean": best["theta"][:p], "fe_sd": np.sqrt(var), "random_effect_sd": best["sd"],
            "laplace_loglik": float(best["laplace_loglik"]), "converged": bool(best["converged"]),
            "n_iter": int(best["n_iter"]), "grid": grid}