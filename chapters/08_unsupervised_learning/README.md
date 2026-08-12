# Chapter 8: Unsupervised Learning - Clustering and Dimensionality Reduction

## Chapter Overview
In unsupervised learning, we are given only inputs without labels. This chapter covers the discovery of intrinsic structure in data through clustering algorithms and dimensionality reduction techniques.

## Learning Objectives
* Implement the k-Means clustering algorithm (Lloyd's Algorithm) from scratch using coordinate descent[cite: 1].
* Apply Principal Component Analysis (PCA) to extract eigenvectors from covariance matrices for dimensionality reduction[cite: 1].
* Understand non-linear dimensionality reduction through the mechanics of autoencoders[cite: 1].

## Concepts Covered
* k-Means and k-Means++ initialization
* DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* Principal Component Analysis (PCA) & Singular Value Decomposition (SVD)
* Autoencoders & Bottleneck Architectures

## Connection to Textbook
The code provides direct implementation of Lloyd's algorithm as described in the text, highlighting its connection to the Expectation-Maximization (EM) algorithm with hard assignments[cite: 1]. We also build a PyTorch Autoencoder to map the theoretical compression of high-dimensional inputs to latent bottleneck codes[cite: 1].
