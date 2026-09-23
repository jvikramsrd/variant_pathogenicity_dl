"""Baselines, scored exactly like the neural arms: same splits, same metrics, same calibration.

    majority        predicts the training class distribution for everything — the floor every
                    other number must clear
    tfidf_lr        TF-IDF (word 1-2 grams) + multinomial logistic regression, L2
    tfidf_svm       TF-IDF + linear SVM (one-vs-rest, squared hinge, L2); its "probabilities"
                    are a softmax over margins and are calibrated on validation like the rest
    structured_lr   the structured genomic fields ONLY, no text — the other half of the
                    clinical-text ablation (EXP-011 / ablation A)

The two linear models are fitted here with L-BFGS on the sparse matrix rather
than through scikit-learn's estimators, for one practical reason: on the
Windows machine this repository is developed on, an Application Control policy
blocks scikit-learn's compiled ``_loss`` module, so ``sklearn.linear_model``
cannot even be imported (docs/dl/CODEBASE_AUDIT.md records the same four test
failures). The objectives are the standard ones and ``backend="sklearn"``
runs the reference implementation where it is available, so the choice is
checkable rather than assumed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from vpdl.slm.schema import CLASSES

__all__ = ["BASELINES", "BaselineModel", "MajorityBaseline", "LinearTextBaseline",
           "StructuredBaseline", "run_baselines"]

BASELINES = ("majority", "tfidf_lr", "tfidf_svm", "structured_lr")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _fit_linear(X, y_onehot: np.ndarray, kind: str, l2: float, max_iter: int = 300) -> np.ndarray:
    """Weights [features+1, classes] for multinomial LR or one-vs-rest squared-hinge SVM."""
    from scipy.optimize import minimize
    from scipy import sparse

    n, d = X.shape
    k = y_onehot.shape[1]
    X = sparse.hstack([X, sparse.csr_matrix(np.ones((n, 1)))]).tocsr()

    def objective(flat):
        W = flat.reshape(d + 1, k)
        scores = X @ W
        if kind == "lr":
            shifted = scores - scores.max(1, keepdims=True)
            log_sum = np.log(np.exp(shifted).sum(1, keepdims=True))
            loss = float((-(y_onehot * (shifted - log_sum)).sum()) / n)
            grad = X.T @ ((np.exp(shifted - log_sum) - y_onehot) / n)
        else:                                   # one-vs-rest squared hinge
            signs = 2 * y_onehot - 1
            margins = np.maximum(0.0, 1 - signs * scores)
            loss = float((margins ** 2).sum() / n)
            grad = X.T @ (-2 * signs * margins / n)
        loss += l2 * float((W[:-1] ** 2).sum())
        grad = np.asarray(grad)
        grad[:-1] += 2 * l2 * W[:-1]
        return loss, grad.ravel()

    result = minimize(objective, np.zeros((d + 1) * k), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter})
    return result.x.reshape(d + 1, k)


@dataclass
class BaselineModel:
    name: str
    classes: tuple[str, ...] = CLASSES
    meta: dict[str, Any] = field(default_factory=dict)

    def fit(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> "BaselineModel":
        raise NotImplementedError

    def predict_proba(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> np.ndarray:
        raise NotImplementedError


def _targets(examples: pd.DataFrame) -> np.ndarray:
    """[N, 5] probability targets (soft for ClinVar's paired terms), rows without a label are zero."""
    target = np.zeros((len(examples), len(CLASSES)), dtype=float)
    lookup = {name: i for i, name in enumerate(CLASSES)}
    for row, (label, soft) in enumerate(zip(examples.get("target_label", pd.Series([None] * len(examples))),
                                            examples.get("target_soft", pd.Series([None] * len(examples))))):
        if isinstance(label, str) and label in lookup:
            target[row, lookup[label]] = 1.0
        elif soft is not None and not (isinstance(soft, float) and np.isnan(soft)):
            target[row] = np.asarray(soft, dtype=float)
    return target


@dataclass
class MajorityBaseline(BaselineModel):
    prior: np.ndarray = field(default_factory=lambda: np.zeros(len(CLASSES)))

    def fit(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> "MajorityBaseline":
        target = _targets(examples)
        total = target.sum()
        self.prior = target.sum(0) / total if total else np.full(len(CLASSES), 1 / len(CLASSES))
        self.meta = {"prior": dict(zip(CLASSES, self.prior.round(4).tolist()))}
        return self

    def predict_proba(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> np.ndarray:
        return np.tile(self.prior, (len(examples), 1))


@dataclass
class LinearTextBaseline(BaselineModel):
    kind: str = "lr"
    l2: float = 1e-4
    max_features: int = 200_000
    backend: str = "scipy"
    vectorizer: Any = None
    weights: np.ndarray | None = None

    def fit(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> "LinearTextBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                          max_features=self.max_features, lowercase=True)
        X = self.vectorizer.fit_transform(examples["input_text"].fillna(""))
        y = _targets(examples)
        keep = y.sum(1) > 0
        if self.backend == "sklearn":
            self._fit_sklearn(X[keep], y[keep])
        else:
            self.weights = _fit_linear(X[keep], y[keep], self.kind, self.l2)
        self.meta = {"vocabulary": int(len(self.vectorizer.vocabulary_)), "backend": self.backend,
                     "kind": self.kind, "l2": self.l2, "rows": int(keep.sum())}
        return self

    def _fit_sklearn(self, X, y) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        labels = y.argmax(1)
        model = (LogisticRegression(max_iter=1000, C=1 / (2 * self.l2 * len(labels)))
                 if self.kind == "lr" else LinearSVC(C=1 / (2 * self.l2 * len(labels))))
        model.fit(X, labels)
        self._sklearn = model

    def predict_proba(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> np.ndarray:
        X = self.vectorizer.transform(examples["input_text"].fillna(""))
        if self.backend == "sklearn":
            model = self._sklearn
            scores = (model.predict_proba(X) if hasattr(model, "predict_proba")
                      else _softmax(np.atleast_2d(model.decision_function(X))))
            out = np.zeros((len(examples), len(CLASSES)))
            for column, label in enumerate(model.classes_):
                out[:, int(label)] = scores[:, column]
            return out / np.maximum(out.sum(1, keepdims=True), 1e-12)
        from scipy import sparse
        X = sparse.hstack([X, sparse.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
        return _softmax(np.asarray(X @ self.weights))


@dataclass
class StructuredBaseline(BaselineModel):
    l2: float = 1e-3
    weights: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)

    @staticmethod
    def design(features: pd.DataFrame, columns: Sequence[str] | None = None):
        encoded = pd.get_dummies(features.astype({c: str for c in features.columns
                                                  if features[c].dtype == object}), dummy_na=True)
        encoded = encoded.astype(float).fillna(0.0)
        if columns is not None:
            encoded = encoded.reindex(columns=list(columns), fill_value=0.0)
        return encoded

    def fit(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> "StructuredBaseline":
        from scipy import sparse
        if features is None:
            raise ValueError("structured_lr needs structured features")
        design = self.design(features)
        self.columns = list(design.columns)
        y = _targets(examples)
        keep = y.sum(1) > 0
        self.weights = _fit_linear(sparse.csr_matrix(design.to_numpy()[keep]), y[keep], "lr", self.l2)
        self.meta = {"columns": len(self.columns), "rows": int(keep.sum())}
        return self

    def predict_proba(self, examples: pd.DataFrame, features: pd.DataFrame | None = None) -> np.ndarray:
        from scipy import sparse
        design = self.design(features, self.columns).to_numpy()
        X = sparse.hstack([sparse.csr_matrix(design), sparse.csr_matrix(np.ones((len(design), 1)))]).tocsr()
        return _softmax(np.asarray(X @ self.weights))


def _make(kind: str, backend: str) -> BaselineModel:
    if kind == "majority":
        return MajorityBaseline(kind)
    if kind == "tfidf_lr":
        return LinearTextBaseline(kind, kind="lr", backend=backend)
    if kind == "tfidf_svm":
        return LinearTextBaseline(kind, kind="svm", backend=backend)
    if kind == "structured_lr":
        return StructuredBaseline(kind)
    raise ValueError(f"unknown baseline {kind!r}; known {BASELINES}")


def run_baselines(examples: pd.DataFrame, out_dir: Path | str, kinds: Sequence[str] = BASELINES,
                  features: pd.DataFrame | None = None, backend: str = "scipy",
                  n_bootstrap: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Fit on train, choose the threshold on validation, report on every evaluated split."""
    from vpdl.slm.evaluation.calibration import calibration_summary
    from vpdl.slm.evaluation.metrics import (binary_metrics, five_class_metrics, pathogenic_score,
                                             validation_threshold)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = examples["split"].isin(["train", "mmr_train"]).to_numpy()
    results: dict[str, Any] = {"n_train": int(train.sum()), "backend": backend, "kinds": list(kinds)}
    lookup = {name: i for i, name in enumerate(CLASSES)}
    hard = examples.get("target_label", pd.Series([None] * len(examples))).map(
        lambda v: lookup.get(v, -1) if isinstance(v, str) else -1).to_numpy()
    binary = pd.to_numeric(examples.get("target_binary", pd.Series([np.nan] * len(examples))),
                           errors="coerce").to_numpy(dtype=float)
    for kind in kinds:
        started = time.time()
        model = _make(kind, backend)
        model.fit(examples.loc[train], None if features is None else features.loc[train])
        entry: dict[str, Any] = {"meta": model.meta, "seconds": round(time.time() - started, 2)}
        threshold = None
        for split in ("val", "mmr_val", "test", "broad_test"):
            rows = (examples["split"] == split).to_numpy()
            if not rows.any():
                continue
            probs = model.predict_proba(examples.loc[rows],
                                        None if features is None else features.loc[rows])
            usable = np.isfinite(binary[rows])
            if split.endswith("val") and threshold is None and usable.sum() and \
                    len(np.unique(binary[rows][usable])) == 2:
                threshold = validation_threshold(binary[rows][usable], pathogenic_score(probs)[usable])
            entry[split] = {"n": int(rows.sum()), "five_class": five_class_metrics(hard[rows], probs),
                            "calibration": calibration_summary(probs, hard[rows])}
            if usable.sum() and len(np.unique(binary[rows][usable])) == 2:
                entry[split]["binary"] = binary_metrics(
                    binary[rows][usable], pathogenic_score(probs)[usable], threshold,
                    n_bootstrap=n_bootstrap, seed=seed, is_validation=split.endswith("val"))
        results[kind] = entry
    (out / "baselines.json").write_text(json.dumps(results, indent=2, default=str))
    return results
