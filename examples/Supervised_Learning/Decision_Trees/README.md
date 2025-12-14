# Decision Trees

This directory contains example code and notes for the Decision Trees algorithm in supervised learning.

## Algorithm

Decision Trees are supervised learning models used for classification and regression.
They work by recursively partitioning the feature space into regions that are increasingly
homogeneous with respect to the target variable. At each internal node, the algorithm
selects a feature and a split threshold that best separates the data according to an
impurity measure.

The objective of a Decision Tree is to minimize prediction error by choosing splits
that reduce impurity at each step. Common impurity measures include entropy and
Gini impurity for classification, and variance or mean squared error for regression.

Key hyperparameters include:
- **Maximum depth**: limits how deep the tree can grow
- **Minimum samples per split / leaf**: controls how many data points are required
  to split a node or remain in a leaf
- **Splitting criterion**: such as entropy, Gini impurity, or variance reduction
- **Maximum number of features** considered at each split (optional)

These hyperparameters help control overfitting and model complexity.

## Data

Decision Trees operate on structured tabular data consisting of input features and,
for supervised learning, corresponding labels. Features may be numerical or categorical,
although categorical features are often encoded numerically prior to training.

In these examples, datasets are loaded into feature matrices and label vectors.
Basic preprocessing may include:
- handling missing values
- encoding categorical variables
- normalizing or standardizing features (optional, though not strictly required
  for Decision Trees)

The data is typically split into training and testing sets to evaluate model
performance and generalization.
