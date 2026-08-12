# Chapter 9: Neural Networks - From Perceptron to Deep MLP

## Chapter Overview
This chapter builds the foundation of deep learning. We trace the evolution of neural architectures from the biological neuron and Perceptron to multi-layer perceptrons (MLP). We rigorously derive the backpropagation algorithm using computation graphs and the multivariate chain rule.

## Learning Objectives
* Understand why non-linear activation functions are required to break linear composability.
* Derive and implement the backpropagation algorithm from scratch using pure NumPy.
* Implement modern activation functions (ReLU, GELU, Swish) and analyze their gradient flow properties.
* Apply momentum, RMSProp, and Adam optimizers with bias correction.
* Implement weight initialization techniques (Xavier/Glorot, He/Kaiming) to prevent vanishing/exploding gradients.

## Concepts Covered
* The Perceptron & Universal Approximation Theorem
* Computation Graphs & Automatic Differentiation
* Activations: Sigmoid, Tanh, ReLU, GELU, Swish
* Optimization: SGD, Momentum, RMSProp, Adam, AdamW
* Regularization: Inverted Dropout, Batch Normalization, Layer Normalization
* Weight Initialization Strategies

## Connection to Textbook
This module directly mirrors Chapter 9 of the textbook. The mathematical derivations of the error signals $\delta^{[l]}$ across hidden layers match the matrix-calculus backpropagation proofs presented in the text.
