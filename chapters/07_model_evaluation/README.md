# Chapter 7: Model Evaluation, Selection, and Hyperparameter Optimisation

## Chapter Overview
Evaluating a model incorrectly is the most common failure mode in applied machine learning. This chapter covers the rigorous frameworks required to estimate true risk from empirical risk, emphasizing the importance of preventing data leakage. We implement comprehensive metrics, cross-validation strategies, and search algorithms for hyperparameter tuning.

## Learning Objectives
* Compute classification and regression metrics from scratch using confusion matrices[cite: 1].
* Implement and analyze k-Fold Cross-Validation[cite: 1].
* Execute Grid Search and Random Search for hyperparameter tuning, understanding why Random Search is often more sample-efficient in high dimensions[cite: 1].
* Diagnose learning curves to differentiate between high bias and high variance[cite: 1].

## Concepts Covered
* Precision, Recall, F1-Score, and ROC-AUC
* MAE, MSE, and $R^2$ Score
* K-Fold and Stratified Cross-Validation
* Grid Search vs. Random Search
* Bayesian Optimisation concepts

## Connection to Textbook
The code explicitly reflects the text's mathematical definitions of evaluation metrics. The from-scratch implementations demonstrate how every binary classification metric derives from the 4 elements of the confusion matrix: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)[cite: 1].
