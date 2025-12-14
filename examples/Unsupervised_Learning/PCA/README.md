# PCA

This directory contains example code and notes for the PCA algorithm
in unsupervised learning.

## Algorithm

Principal Component Analysis (PCA) is a dimensionality reduction technique that
projects data onto a lower-dimensional space while preserving as much variance
as possible. PCA finds orthogonal directions (principal components) that maximize
variance in the data.

The algorithm typically involves:
- centering the data
- computing the covariance matrix (or using SVD)
- extracting eigenvectors corresponding to the largest eigenvalues

Key hyperparameters/settings include:
- **Number of components** to retain
- **Explained variance threshold**
- **Whitening** (optional)

PCA is commonly used for visualization, noise reduction, and preprocessing
before applying other learning algorithms.

## Data

PCA operates on a numeric feature matrix `X` with no labels. Preprocessing usually
includes:
- centering features (mean subtraction)
- scaling features if variances differ significantly

Output is a transformed feature matrix with reduced dimensionality, along with
information about explained variance per component.
