"""
Logistic Regression (NumPy-only).

Binary logistic regression trained with (batch) gradient descent.

API mirrors a subset of scikit-learn:
- fit(X, y)
- predict_proba(X)  -> P(y=1)
- predict(X)
- score(X, y)       -> accuracy

Notes
-----
- This implementation is binary only.
- y must contain exactly two unique values; they will be mapped to {0,1}
  internally, and predictions are returned using the original label set.
- Uses a numerically stable sigmoid.

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.logistic_regression import LogisticRegression
>>> X = np.array([[0.0],[1.0],[2.0],[3.0]])
>>> y = np.array([0, 0, 1, 1])
>>> clf = LogisticRegression(lr=0.5, max_iter=2000, random_state=0).fit(X, y)
>>> clf.predict([[0.2],[2.8]]).tolist()
[0, 1]
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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """
    Binary logistic regression classifier trained by gradient descent.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    lr : float, default=0.1
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Stop if change in loss is smaller than tol.
    alpha : float, default=0.0
        L2 regularization strength. Intercept is NOT regularized.
    random_state : int | None
        RNG seed used for weight initialization.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a bool.")
        self.fit_intercept = fit_intercept

        try:
            self.lr = float(lr)
        except (TypeError, ValueError) as e:
            raise TypeError("lr must be a float.") from e
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")

        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")
        self.max_iter = int(max_iter)

        try:
            self.tol = float(tol)
        except (TypeError, ValueError) as e:
            raise TypeError("tol must be a float.") from e
        if self.tol < 0:
            raise ValueError("tol must be >= 0.")

        try:
            self.alpha = float(alpha)
        except (TypeError, ValueError) as e:
            raise TypeError("alpha must be a float.") from e
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")

        self.random_state = random_state

        # learned
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def _encode_binary(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map labels -> {0,1}. Store class order for decoding.
        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("This LogisticRegression supports binary classification only (exactly 2 classes).")
        # sorted order from np.unique; treat classes_[1] as positive class
        y01 = (y == classes[1]).astype(float)
        return y01, classes

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        Xarr = _ensure_2d_float(X, "X")
        yarr = _ensure_1d(y, "y")
        if Xarr.shape[0] != yarr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        y01, classes = self._encode_binary(yarr)
        self.classes_ = classes
        self.n_features_in_ = Xarr.shape[1]

        A = _add_intercept(Xarr) if self.fit_intercept else Xarr
        n, d = A.shape

        rng = _rng(self.random_state)
        w = rng.normal(loc=0.0, scale=0.01, size=d).astype(float)

        prev_loss = np.inf

        for _ in range(self.max_iter):
            z = A @ w
            p = _sigmoid(z)

            # loss = -mean(y log p + (1-y) log(1-p)) + alpha/2 * ||w||^2 (no intercept)
            eps = 1e-12
            p_clip = np.clip(p, eps, 1.0 - eps)
            data_loss = -np.mean(y01 * np.log(p_clip) + (1.0 - y01) * np.log(1.0 - p_clip))

            if self.alpha > 0.0:
                if self.fit_intercept:
                    reg_loss = 0.5 * self.alpha * float(np.sum(w[1:] ** 2))
                else:
                    reg_loss = 0.5 * self.alpha * float(np.sum(w ** 2))
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            # gradient: A^T(p - y)/n + alpha*w (no intercept)
            grad = (A.T @ (p - y01)) / n
            if self.alpha > 0.0:
                if self.fit_intercept:
                    grad[1:] += self.alpha * w[1:]
                else:
                    grad += self.alpha * w

            w -= self.lr * grad

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].astype(float, copy=False)
        else:
            self.intercept_ = 0.0
            self.coef_ = w.astype(float, copy=False)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return probabilities for both classes: shape (n_samples, 2).
        Column order is self.classes_ (sorted).
        """
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        Xarr = _ensure_2d_float(X, "X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")

        z = (Xarr @ self.coef_) + float(self.intercept_)
        p1 = _sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels (original label dtype).
        """
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        return self.classes_[idx]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Accuracy score.
        """
        ytrue = _ensure_1d(y, "y")
        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(ytrue == ypred))
