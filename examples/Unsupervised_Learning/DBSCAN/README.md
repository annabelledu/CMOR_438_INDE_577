# DBSCAN

This directory contains example code and notes for the DBSCAN algorithm
in unsupervised learning.

## Algorithm

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups
points based on density rather than distance to a centroid. It identifies
clusters as dense regions of points separated by regions of lower density.

Key concepts include:
- **Core points:** points with at least `min_samples` neighbors within distance `eps`
- **Border points:** points within `eps` of a core point but with fewer neighbors
- **Noise points:** points that are neither core nor border points

DBSCAN can find arbitrarily shaped clusters and automatically identifies noise.

Key hyperparameters include:
- **eps:** neighborhood radius
- **min_samples:** minimum number of points required to form a dense region
- **Distance metric:** how distances are computed between points

Unlike k-means, DBSCAN does not require specifying the number of clusters in advance.

## Data

DBSCAN uses a numeric feature matrix `X` with no labels. Because clustering is
based on distances, feature scaling is often critical for meaningful results.

Typical preprocessing steps include:
- normalization or standardization of features
- removing duplicate points (optional)
- dimensionality reduction (sometimes helpful in high dimensions)

Output labels indicate cluster membership, with a special label (often `-1`)
used for noise points.
