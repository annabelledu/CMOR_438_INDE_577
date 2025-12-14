# Perceptron

This directory contains example code and notes for the Perceptron algorithm
in supervised learning.

## Algorithm

The Perceptron is a linear binary classifier that learns a decision boundary of
the form:

`sign(w·x + b)`

It is trained iteratively by updating weights when a sample is misclassified.
For a labeled example `(x, y)` with `y ∈ {−1, +1}` (or {0,1} depending on setup),
the update typically moves the weights in the direction that would correct the
mistake.

Key hyperparameters/settings include:
- **Learning rate** (step size for updates)
- **Number of epochs** (passes over the training data)
- **Shuffle** behavior (can affect convergence)
- **Feature scaling** (helps with stable updates)

The perceptron converges for linearly separable data, but cannot represent
nonlinear decision boundaries without feature transformations.

## Data

The perceptron uses a numeric feature matrix `X` and binary labels `y`. Common
preprocessing steps include:
- scaling/standardizing features
- encoding labels into a consistent binary format
- adding a bias/intercept term (or learning `b` separately)

Data is split into training/testing sets to evaluate classification accuracy.
