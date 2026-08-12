# Chapter 5: Classification - From Logistic Regression to Support Vector Machines

## Chapter Overview
This chapter explores how to separate data into discrete classes. We begin by illustrating why linear regression fails for classification (due to outlier sensitivity and non-probabilistic outputs) and introduce the sigmoid function. We then advance to maximum margin classification using Support Vector Machines (SVM).

## Learning Objectives
* Understand the structural failure modes of using OLS for classification.
* Derive and implement the Logistic Regression gradient using the cross-entropy loss function.
* Implement k-Nearest Neighbors (k-NN) and analyze the curse of dimensionality.
* Optimize a Support Vector Machine via gradient descent on the hinge loss.

## Concepts Covered
* The Sigmoid Function
* Binary Cross-Entropy Loss
* Decision Boundaries
* k-Nearest Neighbors (k-NN)
* Maximum Margin Classification
* The Kernel Trick

## Connection to Textbook
The code explicitly maps to the textbook's derivations. For example, the logistic regression update rule mirrors the text's proof that the gradient elegantly simplifies to the prediction error multiplied by the feature vector[cite: 1].
