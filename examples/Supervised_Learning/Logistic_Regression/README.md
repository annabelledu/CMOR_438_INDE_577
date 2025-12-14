# Logistic Regression

This directory contains example code and notes for the Logistic Regression
algorithm in supervised learning.

## Algorithm

Logistic Regression is a linear model for **classification** (often binary, but
can be extended to multiclass). It models the probability of a class using the
logistic (sigmoid) function:

`p(y=1 | x) = sigmoid(w·x + b)`

Training typically maximizes the likelihood (equivalently minimizes log loss /
cross-entropy). Predictions are made by thresholding probabilities (binary) or
choosing the most probable class (multiclass extensions).

Key hyperparameters/settings include:
- **Regularization strength** (e.g., L2 penalty) to control overfitting
- **Optimization settings** (learning rate, iterations) if using gradient descent
- **Decision threshold** for binary classification (default often 0.5)
- **Multiclass strategy** (one-vs-rest or softmax-based) if applicable

Logistic regression is valued for simplicity, interpretability, and strong
baseline performance.

## Data

Logistic regression uses a numeric feature matrix `X` and discrete class labels `y`.
Preprocessing commonly includes:
- feature scaling/standardization (important for stable optimization)
- encoding categorical features (one-hot encoding)
- handling missing values

Data is split into training/testing sets, and evaluation often uses accuracy,
precision/recall, F1 score, ROC-AUC (binary), and confusion matrices.
