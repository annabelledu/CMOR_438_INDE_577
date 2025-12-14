# Ensemble Methods

This directory contains example code and notes for the Ensemble Methods algorithm
in supervised learning.

## Algorithm

Ensemble methods combine predictions from multiple “base” models to improve
generalization, stability, and performance compared to a single model.
The core idea is that a group of diverse models can reduce error by averaging
(or voting), especially when individual models have different mistakes.

Common ensemble strategies include:
- **Bagging (Bootstrap Aggregating):** train many models on bootstrap-resampled
  datasets and average/vote their predictions. This primarily reduces variance
  (e.g., Random Forests).
- **Boosting:** train models sequentially, where each new model focuses more on
  correcting errors made by earlier models. This can reduce bias and improve
  accuracy (e.g., AdaBoost, Gradient Boosting).
- **Stacking:** train multiple different models and then train a “meta-model”
  that learns how to best combine their outputs.

Key hyperparameters depend on the ensemble type, but often include:
- **Number of estimators** (how many base learners)
- **Base learner complexity** (e.g., tree depth for tree-based ensembles)
- **Sampling/subsampling settings** (bootstrap, feature subsampling)
- **Learning rate** (boosting)
- **Regularization** controls to prevent overfitting

## Data

Ensemble methods typically operate on the same kinds of input data as their base
learners (often tabular feature matrices with labels). For classification, labels
are categorical; for regression, labels are continuous.

Datasets are usually loaded into a feature matrix `X` and target vector `y`,
then split into training/testing sets. Preprocessing may include:
- encoding categorical variables
- handling missing values
- basic scaling (sometimes helpful, depending on base learner)

For tree-based ensembles, feature scaling is often not required, but clean,
consistent preprocessing still matters for fair evaluation.
