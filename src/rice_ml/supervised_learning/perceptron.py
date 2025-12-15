"""
Perceptron (NumPy-only).

Binary linear classifier trained with the classic perceptron update rule.

API mirrors a subset of scikit-learn:
- fit(X, y)
- predict(X)
- score(X, y)  -> accuracy

Notes
-----
- This implementation is binary only.
- y may be any two distinct labels (e.g., {0,1} or {"A","B"}).
  Internally we map the "positive" class to +1 and the other to -1.
- Uses a simple online (sample-by-sample) update per epoch.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.perceptron import Perceptron
>>> X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
>>> y = np.array([0, 0, 0, 1])  # AND
>>> clf = Perceptron(lr=1.0, max_iter=20, random_state=0).fit(X, y)
>>> clf.predict(X).tolist()
[0, 0, 0, 1]
"""

from __future__ import annotations

from typing import Optional, Sequence, Union, Tuple

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    Xarr = np.asarray(X)
    if Xarr.ndim == 1:
        Xarr = Xarr.reshape(-1, 1)
    if Xarr.ndim != 2:
        raise ValueError(f"{name} must be 2D (or 1D convertible to 2D).")
    if Xarr.size == 0 or Xarr.shape[0] == 0 or Xarr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(Xarr.dtype, np.number):
        try:
            Xarr = Xarr.astype(float)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must contain numeric values.") from e
    else:
        Xarr = Xarr.astype(float, copy=False)
    return Xarr


def _ensure_1d(y: ArrayLike, name: str = "y") -> np.ndarray:
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
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(seed))


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1), dtype=float), X]


class Perceptron:
    """
    Binary perceptron classifier.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lr : float, default=1.0
        Learning rate for updates.
    max_iter : int, default=1000
        Number of epochs (passes over the dataset).
    shuffle : bool, default=True
        Whether to shuffle training data each epoch.
    random_state : int | None
        RNG seed (used for init and shuffling).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector.
    intercept_ : float
        Bias term.
    classes_ : ndarray of shape (2,)
        Sorted class labels seen during fit.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        lr: float = 1.0,
        max_iter: int = 1000,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a bool.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool.")

        try:
            lr_f = float(lr)
        except (TypeError, ValueError) as e:
            raise TypeError("lr must be a float.") from e
        if lr_f <= 0:
            raise ValueError("lr must be > 0.")

        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")

        self.fit_intercept = fit_intercept
        self.lr = lr_f
        self.max_iter = int(max_iter)
        self.shuffle = shuffle
        self.random_state = random_state

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def _encode_binary(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("Perceptron supports binary classification only (exactly 2 unique labels).")
        # map classes[1] -> +1, classes[0] -> -1
        y_pm = np.where(y == classes[1], 1.0, -1.0)
        return y_pm, classes

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        Xarr = _ensure_2d_float(X, "X")
        yarr = _ensure_1d(y, "y")
        if Xarr.shape[0] != yarr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        y_pm, classes = self._encode_binary(yarr)
        self.classes_ = classes
        self.n_features_in_ = Xarr.shape[1]

        A = _add_intercept(Xarr) if self.fit_intercept else Xarr
        n, d = A.shape

        rng = _rng(self.random_state)
        w = rng.normal(loc=0.0, scale=0.01, size=d).astype(float)

        indices = np.arange(n)

        for _ in range(self.max_iter):
            if self.shuffle:
                rng.shuffle(indices)

            mistakes = 0
            for i in indices:
                xi = A[i]
                yi = y_pm[i]
                score = float(xi @ w)

                # If misclassified or on boundary, update
                if yi * score <= 0.0:
                    w = w + self.lr * yi * xi
                    mistakes += 1

            # Early stop if perfectly classified
            if mistakes == 0:
                break

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].astype(float, copy=False)
        else:
            self.intercept_ = 0.0
            self.coef_ = w.astype(float, copy=False)

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        Xarr = _ensure_2d_float(X, "X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")

        return (Xarr @ self.coef_) + float(self.intercept_)

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        scores = self.decision_function(X)
        # score >= 0 -> classes_[1] (positive), else classes_[0]
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        ytrue = _ensure_1d(y, "y")
        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(ytrue == ypred))
from .perceptron import Perceptron