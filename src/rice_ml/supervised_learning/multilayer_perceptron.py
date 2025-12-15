"""
Multilayer Perceptron (NumPy-only).

Provides small, teaching-focused neural networks:
- MLPClassifier: softmax output, cross-entropy loss
- MLPRegressor: linear output, MSE loss

Both are trained with batch gradient descent (full-batch) and support:
- 1 hidden layer (for simplicity and readability)
- ReLU or tanh activation
- L2 regularization

API mirrors a subset of scikit-learn:
- fit(X, y)
- predict(X)
- (classifier only) predict_proba(X)
- score(X, y)

Notes
-----
- This is intentionally compact and educational rather than optimized.
- Uses NumPy only (no autograd).

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
>>> X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
>>> y = np.array([0, 1, 1, 0])  # XOR
>>> clf = MLPClassifier(hidden_layer_sizes=4, lr=0.1, max_iter=5000, random_state=0).fit(X, y)
>>> clf.predict(X).shape
(4,)
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union, Tuple

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ------------------------ Helpers / validation ------------------------

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


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t


def _softmax(logits: np.ndarray) -> np.ndarray:
    # stable softmax
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((y.size, n_classes), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


# ------------------------ Base: 1-hidden-layer MLP ------------------------

class _MLPBase:
    def __init__(
        self,
        *,
        hidden_layer_sizes: int = 32,
        activation: Literal["relu", "tanh"] = "relu",
        lr: float = 0.1,
        max_iter: int = 1000,
        alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        if not isinstance(hidden_layer_sizes, (int, np.integer)) or int(hidden_layer_sizes) < 1:
            raise ValueError("hidden_layer_sizes must be a positive integer.")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'.")
        try:
            lr_f = float(lr)
        except (TypeError, ValueError) as e:
            raise TypeError("lr must be a float.") from e
        if lr_f <= 0:
            raise ValueError("lr must be > 0.")
        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")
        try:
            alpha_f = float(alpha)
        except (TypeError, ValueError) as e:
            raise TypeError("alpha must be a float.") from e
        if alpha_f < 0:
            raise ValueError("alpha must be >= 0.")

        self.hidden_layer_sizes = int(hidden_layer_sizes)
        self.activation = activation
        self.lr = lr_f
        self.max_iter = int(max_iter)
        self.alpha = alpha_f
        self.random_state = random_state

        # learned params
        self.n_features_in_: Optional[int] = None
        self._W1: Optional[np.ndarray] = None
        self._b1: Optional[np.ndarray] = None
        self._W2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None

    def _act(self, z: np.ndarray) -> np.ndarray:
        return _relu(z) if self.activation == "relu" else _tanh(z)

    def _act_grad(self, z: np.ndarray) -> np.ndarray:
        return _relu_grad(z) if self.activation == "relu" else _tanh_grad(z)

    def _init_weights(self, n_in: int, n_out: int) -> None:
        rng = _rng(self.random_state)
        h = self.hidden_layer_sizes

        # He init for relu, Xavier for tanh (simple variant)
        if self.activation == "relu":
            scale1 = np.sqrt(2.0 / n_in)
            scale2 = np.sqrt(2.0 / h)
        else:
            scale1 = np.sqrt(1.0 / n_in)
            scale2 = np.sqrt(1.0 / h)

        self._W1 = rng.normal(0.0, scale1, size=(n_in, h)).astype(float)
        self._b1 = np.zeros((h,), dtype=float)
        self._W2 = rng.normal(0.0, scale2, size=(h, n_out)).astype(float)
        self._b2 = np.zeros((n_out,), dtype=float)

    def _check_fitted(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._W1 is None or self._b1 is None or self._W2 is None or self._b2 is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self._W1, self._b1, self._W2, self._b2


# ------------------------ Classifier ------------------------

class MLPClassifier(_MLPBase):
    """
    1-hidden-layer MLP classifier (softmax).

    y must be integer labels 0..K-1 for simplicity.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: int = 32,
        activation: Literal["relu", "tanh"] = "relu",
        lr: float = 0.1,
        max_iter: int = 1000,
        alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            lr=lr,
            max_iter=max_iter,
            alpha=alpha,
            random_state=random_state,
        )
        self.n_classes_: Optional[int] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPClassifier":
        Xarr = _ensure_2d_float(X, "X")
        yarr = _ensure_1d(y, "y")

        if Xarr.shape[0] != yarr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if not np.issubdtype(yarr.dtype, np.integer):
            raise ValueError("y must be integer-encoded class labels (0,1,2,...).")

        classes = np.unique(yarr)
        self.n_classes_ = int(classes.max() + 1)
        self.n_features_in_ = Xarr.shape[1]

        self._init_weights(self.n_features_in_, self.n_classes_)
        W1, b1, W2, b2 = self._check_fitted()

        Y = _one_hot(yarr, self.n_classes_)
        n = Xarr.shape[0]

        for _ in range(self.max_iter):
            # forward
            Z1 = Xarr @ W1 + b1
            A1 = self._act(Z1)
            logits = A1 @ W2 + b2
            P = _softmax(logits)

            # gradients (cross-entropy with softmax)
            dlogits = (P - Y) / n  # (n, K)
            dW2 = A1.T @ dlogits + self.alpha * W2
            db2 = np.sum(dlogits, axis=0)

            dA1 = dlogits @ W2.T
            dZ1 = dA1 * self._act_grad(Z1)
            dW1 = Xarr.T @ dZ1 + self.alpha * W1
            db1 = np.sum(dZ1, axis=0)

            # update
            W1 -= self.lr * dW1
            b1 -= self.lr * db1
            W2 -= self.lr * dW2
            b2 -= self.lr * db2

        # store back
        self._W1, self._b1, self._W2, self._b2 = W1, b1, W2, b2
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        W1, b1, W2, b2 = self._check_fitted()
        if self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted.")
        Xarr = _ensure_2d_float(X, "X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")

        Z1 = Xarr @ W1 + b1
        A1 = self._act(Z1)
        logits = A1 @ W2 + b2
        return _softmax(logits)

    def predict(self, X: ArrayLike) -> np.ndarray:
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        ytrue = _ensure_1d(y, "y")
        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(ytrue == ypred))


# ------------------------ Regressor ------------------------

class MLPRegressor(_MLPBase):
    """
    1-hidden-layer MLP regressor (linear output, MSE loss).
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressor":
        Xarr = _ensure_2d_float(X, "X")
        yarr = _ensure_1d(y, "y")

        # require numeric y
        if not np.issubdtype(yarr.dtype, np.number):
            try:
                yarr = yarr.astype(float)
            except (TypeError, ValueError) as e:
                raise TypeError("y must be numeric for regression.") from e
        else:
            yarr = yarr.astype(float, copy=False)

        if Xarr.shape[0] != yarr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.n_features_in_ = Xarr.shape[1]
        self._init_weights(self.n_features_in_, 1)
        W1, b1, W2, b2 = self._check_fitted()

        n = Xarr.shape[0]
        ycol = yarr.reshape(-1, 1)

        for _ in range(self.max_iter):
            # forward
            Z1 = Xarr @ W1 + b1
            A1 = self._act(Z1)
            yhat = A1 @ W2 + b2  # (n,1)

            # gradients (MSE)
            dy = (yhat - ycol) / n  # (n,1)
            dW2 = A1.T @ dy + self.alpha * W2
            db2 = np.sum(dy, axis=0)

            dA1 = dy @ W2.T
            dZ1 = dA1 * self._act_grad(Z1)
            dW1 = Xarr.T @ dZ1 + self.alpha * W1
            db1 = np.sum(dZ1, axis=0)

            # update
            W1 -= self.lr * dW1
            b1 -= self.lr * db1
            W2 -= self.lr * dW2
            b2 -= self.lr * db2

        self._W1, self._b1, self._W2, self._b2 = W1, b1, W2, b2
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        W1, b1, W2, b2 = self._check_fitted()
        if self.n_features_in_ is None:
            raise RuntimeError("Model is not fitted.")
        Xarr = _ensure_2d_float(X, "X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")

        Z1 = Xarr @ W1 + b1
        A1 = self._act(Z1)
        yhat = A1 @ W2 + b2
        return yhat.ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        # R^2, consistent with your r2_score behavior
        ytrue = _ensure_1d(y, "y")
        if not np.issubdtype(np.asarray(ytrue).dtype, np.number):
            raise TypeError("y must be numeric for regression.")
        ytrue = np.asarray(ytrue, dtype=float)

        ypred = self.predict(X)
        if ytrue.shape[0] != ypred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        ss_res = float(np.sum((ytrue - ypred) ** 2))
        y_mean = float(np.mean(ytrue))
        ss_tot = float(np.sum((ytrue - y_mean) ** 2))

        if ss_tot == 0.0:
            if ss_res == 0.0:
                return 1.0
            raise ValueError("R^2 is undefined when y_true is constant and fit is not perfect.")
        return float(1.0 - ss_res / ss_tot)
from .multilayer_perceptron import MLPClassifier, MLPRegressor
