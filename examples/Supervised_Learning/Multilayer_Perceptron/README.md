# Multilayer Perceptron

This directory contains example code and notes for the Multilayer Perceptron
algorithm in supervised learning.

## Algorithm

A Multilayer Perceptron (MLP) is a feedforward neural network composed of layers
of neurons: an input layer, one or more hidden layers, and an output layer.
Each layer applies an affine transformation followed by a nonlinear activation
function (e.g., ReLU, sigmoid, tanh). The network is trained using backpropagation
to compute gradients and an optimizer (often gradient descent variants) to update
weights.

Key hyperparameters include:
- **Network architecture:** number of hidden layers and hidden units per layer
- **Activation functions:** ReLU, tanh, sigmoid, etc.
- **Learning rate** and **number of epochs**
- **Batch size** (stochastic vs mini-batch training)
- **Regularization:** L2 weight decay, dropout, early stopping
- **Initialization** strategy

MLPs can model nonlinear relationships and handle complex patterns, but may
require careful tuning to avoid overfitting.

## Data

MLPs typically use numeric feature matrices `X` and labels/targets `y` (for
classification or regression). Common preprocessing includes:
- normalization/standardization of features
- encoding categorical variables
- train/validation/test splits (validation is important for tuning)
- shuffling data and batching for training stability

For classification, labels may be integer-encoded and optionally one-hot encoded,
depending on the output layer and loss function.
