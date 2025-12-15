"""
Ensemble Methods (NumPy-only).

This module provides lightweight ensemble models suitable for teaching:
- BaggingClassifier: bootstrap aggregation over any base estimator
- RandomForestClassifier: bagging of decision trees with feature subsampling

The API is intentionally sklearn-like: fit / predict / predict_proba / score.

Notes
-----
- Everything is implemented with NumPy only (no scikit-learn dependency).
- Base estimators are expected to follow a simple protocol:
    - fit(X, y) -> self
    - predict(X) -> 1D array
    - (optional) predict_proba(X) -> 2D array
- RandomForestClassifier uses DecisionTreeClassifier from this package by default.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.ensemble_methods import BaggingClassifier
>>> from rice_ml.supervised_learning.knn import KNNClassifier
>>> X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
>>> y = np.array([0,0,1,1])
>>> base = KNNClassifier(n_neighbors=1)
>>> bag = BaggingClassifier(base_estimator=base, n_estimators=5, random_state=0).fit(X, y)
>>> bag.predict([[0.1, 0.1]]).tolist()
[0]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Try to import your DecisionTreeClassifier for RandomForest; if not available, RF will error nicely.
try:
    from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier
except Exception:  # pragma: no cover
    DecisionTreeClassifier = None  # type: ignore


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


# --------------------------- helpers ---------------------------

def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    Xarr = np.asarray(X)
    if Xarr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if Xarr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(Xarr.dtype, np.number):
        try:
            Xarr = Xarr.astype(float)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must contain numeric values.") from e
    else:
        Xarr = Xarr.astype(float, copy=False)
    return Xarr


def _ensure_1d(y: Any, name: str = "y") -> np.ndarray:
    yarr = np.asarray(y)
    if yarr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if yarr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return yarr


def _rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("random_state must be an int or None.")
    return np.random.default_rng(int(seed))


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)


def _clone_estimator(estimator: Any) -> Any:
    """
    Best-effort clone.

    - If estimator has get_params/set_params like sklearn, clone via params.
    - Otherwise, try to reconstruct via __class__(**vars) for common patterns.
    - Fallback: use the same instance (not ideal but better than crashing).
    """
    if hasattr(estimator, "get_params"):
        params = estimator.get_params(deep=True)  # type: ignore
        return estimator.__class__(**params)
    # Try constructor kwargs from __dict__ (works for your KNN/DecisionTree styles)
    try:
        return estimator.__class__(**{k: v for k, v in vars(estimator).items() if not k.startswith("_")})
    except Exception:
        return estimator


def _majority_vote(preds: np.ndarray) -> np.ndarray:
    """
    preds: shape (n_estimators, n_samples) of labels (any dtype).
    Returns shape (n_samples,) labels via majority vote (ties broken by sorted order).
    """
    n_estimators, n_samples = preds.shape
    out = np.empty(n_samples, dtype=object)
    for j in range(n_samples):
        col = preds[:, j]
        vals, counts = np.unique(col, return_counts=True)
        # tie-break by sorted unique order via np.unique
        out[j] = vals[np.argmax(counts)]
    return out


def _safe_predict_proba(estimator: Any, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    If estimator has predict_proba, use it and align columns to `classes`.
    Otherwise, approximate probabilities from hard predictions.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)  # type: ignore
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError("predict_proba must return a 2D array.")
        # If estimator exposes classes_, align columns
        if hasattr(estimator, "classes_"):
            est_classes = np.asarray(getattr(estimator, "classes_"))
            aligned = np.zeros((X.shape[0], len(classes)), dtype=float)
            for i, c in enumerate(est_classes):
                idx = np.where(classes == c)[0]
                if idx.size:
                    aligned[:, idx[0]] = proba[:, i]
            # re-normalize (just in case)
            row_sums = aligned.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return aligned / row_sums
        # assume already aligned if no classes_
        return proba
    # fallback: one-hot from predict
    yhat = np.asarray(estimator.predict(X))
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for i, c in enumerate(classes):
        out[:, i] = (yhat == c).astype(float)
    return out


# --------------------------- BaggingClassifier ---------------------------

class BaggingClassifier:
    """
    Bootstrap aggregating (bagging) for classification.

    Parameters
    ----------
    base_estimator : object
        Any estimator supporting fit(X,y) and predict(X). Optional predict_proba(X).
    n_estimators : int, default=10
        Number of bootstrap models.
    max_samples : float | int, default=1.0
        If float, fraction of training set size to sample (with replacement).
        If int, exact number of samples to draw (with replacement).
    random_state : int | None
        RNG seed.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.random_state = random_state

        self.estimators_: List[Any] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Any) -> "BaggingClassifier":
        Xtr = _ensure_2d_float(X, "X")
        ytr = _ensure_1d(y, "y")
        if Xtr.shape[0] != ytr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.classes_ = np.unique(ytr)
        rng = _rng(self.random_state)

        n = Xtr.shape[0]
        if isinstance(self.max_samples, (int, np.integer)):
            m = int(self.max_samples)
            if m <= 0 or m > n:
                raise ValueError("max_samples (int) must be in [1, n_samples].")
        else:
            frac = float(self.max_samples)
            if frac <= 0.0 or frac > 1.0:
                raise ValueError("max_samples (float) must be in (0, 1].")
            m = max(1, int(round(frac * n)))

        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=m, endpoint=False)
            est = _clone_estimator(self.base_estimator)
            est.fit(Xtr[idx], ytr[idx])
            self.estimators_.append(est)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted.")
        Xq = _ensure_2d_float(X, "X")
        probs = np.zeros((Xq.shape[0], len(self.classes_)), dtype=float)
        for est in self.estimators_:
            probs += _safe_predict_proba(est, Xq, self.classes_)
        probs /= float(len(self.estimators_))
        # normalize
        row = probs.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return probs / row

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted.")
        Xq = _ensure_2d_float(X, "X")
        # If all have predict_proba, use averaged proba; else majority vote
        use_proba = all(hasattr(est, "predict_proba") for est in self.estimators_)
        if use_proba:
            proba = self.predict_proba(Xq)
            return self.classes_[np.argmax(proba, axis=1)]
        preds = np.vstack([np.asarray(est.predict(Xq), dtype=object) for est in self.estimators_])
        return _majority_vote(preds)

    def score(self, X: ArrayLike, y: Any) -> float:
        ytrue = _ensure_1d(y, "y")
        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(ytrue == ypred))


# --------------------------- RandomForestClassifier ---------------------------

class RandomForestClassifier:
    """
    Random Forest classifier (bagging of decision trees + feature subsampling).

    Parameters
    ----------
    n_estimators : int, default=50
    max_depth : int | None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int | float | str | None, default="sqrt"
        - int: number of features per split
        - float in (0,1]: fraction of features per split
        - "sqrt": sqrt(n_features)
        - "log2": log2(n_features)
        - None: all features
    random_state : int | None
    """

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        random_state: Optional[int] = None,
    ) -> None:
        if DecisionTreeClassifier is None:
            raise ImportError("DecisionTreeClassifier could not be imported; RandomForestClassifier is unavailable.")

        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")

        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_: List[Any] = []
        self.classes_: Optional[np.ndarray] = None

    def _resolve_max_features(self, n_features: int) -> Optional[Union[int, float]]:
        mf = self.max_features
        if mf is None:
            return None
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features))) if n_features > 1 else 1
            raise ValueError("max_features string must be one of {'sqrt','log2'} or None.")
        if isinstance(mf, (int, np.integer)):
            if mf <= 0 or mf > n_features:
                raise ValueError("max_features int must be in [1, n_features].")
            return int(mf)
        if isinstance(mf, (float, np.floating)):
            f = float(mf)
            if not (0.0 < f <= 1.0):
                raise ValueError("max_features float must be in (0,1].")
            return float(f)
        raise ValueError("max_features must be int, float, str, or None.")

    def fit(self, X: ArrayLike, y: Any) -> "RandomForestClassifier":
        Xtr = _ensure_2d_float(X, "X")
        ytr = _ensure_1d(y, "y")
        if Xtr.shape[0] != ytr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.classes_ = np.unique(ytr)
        rng = _rng(self.random_state)

        n_samples, n_features = Xtr.shape
        mf = self._resolve_max_features(n_features)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = _bootstrap_indices(n_samples, rng)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(Xtr[idx], ytr[idx].astype(int) if np.issubdtype(ytr.dtype, np.integer) else ytr[idx])
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("Model is not fitted.")
        Xq = _ensure_2d_float(X, "X")

        probs = np.zeros((Xq.shape[0], len(self.classes_)), dtype=float)
        for est in self.estimators_:
            probs += _safe_predict_proba(est, Xq, self.classes_)
        probs /= float(len(self.estimators_))

        row = probs.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return probs / row

    def predict(self, X: ArrayLike) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: Any) -> float:
        ytrue = _ensure_1d(y, "y")
        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(ytrue == ypred))
