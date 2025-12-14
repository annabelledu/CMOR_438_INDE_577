# K Nearest Neighbors

This directory contains example code and notes for the K Nearest Neighbors
algorithm in supervised learning.

## Algorithm

k-Nearest Neighbors (kNN) is a non-parametric, instance-based learning method.
Rather than fitting a single explicit set of model parameters, kNN stores the
training data and makes predictions by comparing a new sample to the training
samples.

For **classification**, kNN predicts the most common label among the `k` nearest
neighbors (majority vote). For **regression**, it predicts the average (or weighted
average) of neighbor target values.

Key hyperparameters include:
- **k (n_neighbors):** number of neighbors used to make a prediction
- **Distance metric:** how “closeness” is measured (e.g., Euclidean, Manhattan)
- **Weighting scheme:** uniform weighting vs distance-based weighting
- **Tie-breaking** behavior (classification) when votes are equal

The main tradeoff is that small `k` can overfit (high variance) while large `k`
can underfit (high bias).

## Data

kNN expects a numeric feature matrix `X` (shape `(n_samples, n_features)`) and a
label/target vector `y`. Because kNN relies on distance computations, feature
scaling can strongly affect results; common preprocessing steps include:
- standardization or normalization of features
- encoding categorical variables into numeric form
- handling missing values (imputation)

Data is typically split into training and testing sets. kNN is sensitive to
irrelevant features and differing feature scales, so careful preprocessing and
feature selection can be important.
