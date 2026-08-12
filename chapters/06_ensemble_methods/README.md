# Chapter 6: Ensemble Methods - Bagging, Boosting, and Stacking

## Chapter Overview
This chapter demonstrates that combining multiple machine learning models yields superior performance compared to any single model. By rigorously proving the bias-variance decomposition of Mean Squared Error (MSE)[cite: 1], we show how different ensemble techniques target either variance reduction or bias reduction.

## Learning Objectives
* Prove and apply the bias-variance decomposition framework[cite: 1].
* Implement Bootstrap Aggregating (Bagging) and Random Forests to reduce variance[cite: 1].
* Construct Boosting algorithms (AdaBoost, Gradient Boosting, XGBoost) to reduce bias sequentially[cite: 1].
* Apply Stacking properly to avoid data leakage using out-of-fold (OOF) predictions[cite: 1].

## Concepts Covered
* Bias-Variance Tradeoff
* Out-of-Bag (OOB) Error Estimation
* Feature Subsampling
* Exponential Loss & Pseudo-Residuals
* Second-Order Taylor Expansion in XGBoost
* Meta-Learners & Cross-Validated Stacking

## Connection to Textbook
The code explicitly highlights the mathematical divergence between bagging and boosting. For instance, the gradient boosting scripts map to the textbook's definition of functional gradient descent, where trees are fitted sequentially to the pseudo-residuals of the loss function[cite: 1]. Additionally, we explore XGBoost's specific system innovations[cite: 1].
