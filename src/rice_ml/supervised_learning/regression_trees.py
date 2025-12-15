"""
Decision tree regressor implementation for the rice_ml package.

This module provides a simple CART-style regression tree using mean-squared
error (variance) reduction for splits. Implemented from scratch with NumPy only.

API mirrors a subset of scikit-learn:
- fit(X, y)
- predict(X)
- score(X, y) -> R^2

Notes
-----
- Leaves store a constant prediction (mean of y in that leaf).
- Splits are of the form x[feature_index] <= threshold.

Example
-------
>>> import numpy as np
>>> from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor
>>> X = np.array([[0.0],[1.0],[2.0],[3.0]])
>>> y = np.array([0.0, 1.0, 1.5, 3.0])
>>> reg = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)
>>> reg.predict([[1.5]]).shape
(1,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class _RegNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_RegNode"] = None
    right: Optional["_RegNode"] = None
    value: Optional[float] = None  # prediction at node (mean of y)

    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeRegressor:
    """
    CART-style regression tree (MSE criterion).

    Parameters
    ----------
    max_depth : int | None, optional
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split.
    min_samples_leaf : int, default=1
        Minimum number of samples in a leaf.
    max_features : int or float or None, optional
        Number of features to consider at each split.
        - int: use exactly that many
        - float in (0,1]: use that fraction of total features
        - None: use all
    random_state : int | None, optional
        RNG seed (used for feature subsampling).

    Attributes
    ----------
    n_features_ : int
        Number of features seen during fit.
    tree_ : _RegNode
        Root of the fitted tree.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[float | int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_: Optional[int] = None
        self.tree_: Optional[_RegNode] = None
        self._rng: Optional[np.random.Generator] = None

    # ---------------- Public API ----------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.size == 0:
            raise ValueError("X must be non-empty.")
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(float)
            except (TypeError, ValueError) as e:
                raise TypeError("X must contain numeric values.") from e
        else:
            X = X.astype(float, copy=False)

        if not np.issubdtype(y.dtype, np.number):
            try:
                y = y.astype(float)
            except (TypeError, ValueError) as e:
                raise TypeError("y must contain numeric values for regression.") from e
        else:
            y = y.astype(float, copy=False)

        if not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")
        if self.max_depth is not None and (not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0):
            raise ValueError("max_depth must be None or an integer >= 0.")

        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None or self.n_features_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_}.")
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(float)
            except (TypeError, ValueError) as e:
                raise TypeError("X must contain numeric values.") from e
        else:
            X = X.astype(float, copy=False)

        out = np.empty((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            out[i] = float(self._traverse(X[i], self.tree_).value)
        return out

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = np.asarray(y)
        if y_true.ndim != 1:
            raise ValueError("y must be 1D.")
        if not np.issubdtype(y_true.dtype, np.number):
            raise TypeError("y must be numeric.")
        y_true = y_true.astype(float, copy=False)

        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))

        if ss_tot == 0.0:
            if ss_res == 0.0:
                return 1.0
            raise ValueError("R^2 is undefined when y_true is constant and fit is not perfect.")
        return float(1.0 - ss_res / ss_tot)

    # ---------------- Tree building ----------------

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _RegNode:
        n_samples, n_features = X.shape
        value = float(np.mean(y)) if y.size else 0.0

        # stopping
        if (
            n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _RegNode(value=value)

        feat_idx, thresh, (left_mask, right_mask) = self._best_split(X, y)
        if feat_idx is None:
            return _RegNode(value=value)

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _RegNode(
            feature_index=feat_idx,
            threshold=thresh,
            left=left,
            right=right,
            value=value,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], Tuple[np.ndarray, np.ndarray]]:
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, (np.array([]), np.array([]))

        # choose candidate features
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError("max_features int must be in [1, n_features].")
            feature_indices = self._rng.choice(n_features, self.max_features, replace=False)
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
            feature_indices = self._rng.choice(n_features, k, replace=False)
        else:
            raise ValueError("max_features must be None, int, or float.")

        best_score = np.inf
        best_feat = None
        best_thresh = None
        best_left = np.array([], dtype=bool)
        best_right = np.array([], dtype=bool)

        for feat in feature_indices:
            xcol = X[:, feat]
            uniq = np.unique(xcol)
            if uniq.size <= 1:
                continue

            # Using unique values as thresholds (simple + readable)
            for t in uniq:
                left_mask = xcol <= t
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # weighted MSE (variance) within children
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                score = (n_left * mse_left + n_right * mse_right) / n_samples

                if score < best_score:
                    best_score = score
                    best_feat = int(feat)
                    best_thresh = float(t)
                    best_left = left_mask
                    best_right = right_mask

        if best_feat is None:
            return None, None, (np.array([]), np.array([]))
        return best_feat, best_thresh, (best_left, best_right)

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        mu = float(np.mean(y))
        return float(np.mean((y - mu) ** 2))

    def _traverse(self, x: np.ndarray, node: _RegNode) -> _RegNode:
        while not node.is_leaf():
            assert node.feature_index is not None and node.threshold is not None
            if x[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node
