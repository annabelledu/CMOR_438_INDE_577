"""
Linear Regression (NumPy-only).

Implements ordinary least squares (OLS) linear regression with an optional
L2 (ridge) regularization term.

API mirrors a subset of scikit-learn:
- fit(X, y)
- predict(X)
- score(X, y)  -> R^2

Notes
-----
- Uses a stable least-squares solve rather than explicit matrix inverse.
- If fit_intercept=True, an intercept column of ones is added internally.
- If alpha > 0, ridge regularization is applied to weights (intercept is NOT
  regularized by default).

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.linear_regression import LinearRegression
>>> X = np.array([[1.0], [2.0], [3.0]])
>>> y = np.array([2.0, 4.0, 6.0])
>>> model = LinearRegression().fit(X, y)
>>> model.coef_.round(6).tolist()
[2.0]
>>> float(model.intercept_)
0.0
>>> model.predict([[4.0]]).round(6).tolist()
[8.0]
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

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


def _ensure_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    yarr = np.asarray(y)
    if yarr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if yarr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(yarr.dtype, np.number):
        try:
            yarr = yarr.astype(float)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must contain numeric values.") from e
    else:
        yarr = yarr.astype(float, copy=False)
    return yarr


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1), dtype=float), X]


class LinearRegression:
    """
    Ordinary least squares linear regression (optionally ridge-regularized).

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.
    alpha : float, default=0.0
        L2 regularization strength. alpha=0.0 gives OLS.
    """

    def __init__(self, *, fit_intercept: bool = True, alpha: float = 0.0) -> None:
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a bool.")
        try:
            alpha_f = float(alpha)
        except (TypeError, ValueError) as e:
            raise TypeError("alpha must be a float.") from e
        if alpha_f < 0.0:
            raise ValueError("alpha must be >= 0.")
        self.fit_intercept = fit_intercept
        self.alpha = alpha_f

        # learned params
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        Xarr = _ensure_2d_float(X, "X")
        yarr = _ensure_1d_float(y, "y")
        if Xarr.shape[0] != yarr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.n_features_in_ = Xarr.shape[1]

        if self.fit_intercept:
            A = _add_intercept(Xarr)
        else:
            A = Xarr

        # Solve (A^T A + alpha*I) w = A^T y
        # Do NOT regularize intercept by default
        if self.alpha == 0.0:
            w, *_ = np.linalg.lstsq(A, yarr, rcond=None)
        else:
            AtA = A.T @ A
            Aty = A.T @ yarr

            reg = np.eye(AtA.shape[0], dtype=float) * self.alpha
            if self.fit_intercept:
                reg[0, 0] = 0.0  # no penalty on intercept

            w = np.linalg.solve(AtA + reg, Aty)

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].astype(float, copy=False)
        else:
            self.intercept_ = 0.0
            self.coef_ = w.astype(float, copy=False)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        Xarr = _ensure_2d_float(X, "X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")

        return (Xarr @ self.coef_) + float(self.intercept_)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = _ensure_1d_float(y, "y")
        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))

        if ss_tot == 0.0:
            # constant y_true
            if ss_res == 0.0:
                return 1.0
            raise ValueError("R^2 is undefined when y_true is constant and fit is not perfect.")

        return float(1.0 - ss_res / ss_tot)
from .linear_regression import LinearRegression
__all__ = ["LinearRegression"]
