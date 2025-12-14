# Linear Regression

This directory contains example code and notes for the Linear Regression
algorithm in supervised learning.

## Algorithm

Linear Regression models a continuous target variable as a linear function of
input features. Given features `X`, the model predicts:

`y_hat = Xw + b`

where `w` is a vector of coefficients and `b` is an intercept. Training typically
minimizes a loss function such as mean squared error (MSE), either by solving the
normal equations (closed form) or by iterative optimization (e.g., gradient descent).

Key hyperparameters/settings include:
- **Regularization** (if used): L2 (ridge) or L1 (lasso) penalties to reduce
  overfitting and improve stability
- **Learning rate / number of iterations** (for gradient descent training)
- **Fit intercept** behavior
- **Feature selection / polynomial features** (if extending beyond linear)

Interpretability is a major advantage: coefficients represent the estimated
effect of each feature on the target (holding others constant).

## Data

Linear regression uses a numeric feature matrix `X` and a continuous target `y`.
Common preprocessing includes:
- scaling/standardizing features (especially for gradient descent and regularization)
- handling missing values
- encoding categorical variables (one-hot encoding)

Data is typically split into training/testing sets, and evaluation commonly uses
MSE/RMSE and R². Outliers and multicollinearity can affect coefficient stability.
