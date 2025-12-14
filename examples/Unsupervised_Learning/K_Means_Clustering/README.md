# K Means Clustering

This directory contains example code and notes for the K Means Clustering
algorithm in unsupervised learning.

## Algorithm

k-Means clustering partitions data into `k` clusters by minimizing within-cluster
variance. The algorithm alternates between:
1. **Assignment step:** assign each point to the nearest cluster centroid
2. **Update step:** recompute centroids as the mean of assigned points

This process continues until convergence (no change in assignments or centroids).

Key hyperparameters include:
- **k:** number of clusters
- **Initialization method:** random or k-means++
- **Maximum number of iterations**
- **Convergence tolerance**

k-Means is efficient and simple, but assumes roughly spherical, equally sized
clusters and can be sensitive to initialization and outliers.

## Data

k-Means uses a numeric feature matrix `X` with no labels. Since distance is central
to the algorithm, preprocessing often includes:
- feature scaling or normalization
- handling outliers
- optional dimensionality reduction

Cluster quality may be evaluated using metrics like inertia, silhouette score,
or by visual inspection.
