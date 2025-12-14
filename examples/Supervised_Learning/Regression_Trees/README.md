# Regression Trees

This directory contains example code and notes for the Regression Trees
algorithm in supervised learning.

## Algorithm

A Regression Tree is a decision tree used for predicting continuous targets.
It recursively partitions the feature space into regions and assigns each region
a predicted value, typically the mean of the training targets in that leaf.

Splits are chosen to reduce a regression loss such as:
- **Mean Squared Error (MSE)** / variance reduction
- **Mean Absolute Error (MAE)** (less common but more robust to outliers)

Key hyperparameters include:
- **Maximum depth**
- **Minimum samples per split / leaf**
- **Splitting criterion** (MSE/variance reduction, etc.)
- **Pruning or regularization** settings to prevent overfitting

Regression trees can model nonlinear relationships and interactions, but single
trees can overfit; ensembles (Random Forests, Gradient Boosting) are common
improvements.

## Data

Regression trees use a numeric feature matrix `X` and a continuous target vector `y`.
Preprocessing typically includes:
- handling missing values
- encoding categorical variables

Feature scaling is usually not required for tree-based models, but consistent
preprocessing is still important for evaluation. Data is split into training and
testing sets, with performance measured using metrics like MSE/RMSE and R².
