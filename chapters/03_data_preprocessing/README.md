# Chapter 3: Data Preprocessing and Feature Engineering

## Chapter Overview
Raw data from the real world is messy, inconsistent, and rarely in the form that ML algorithms require[cite: 1]. This chapter covers the preprocessing pipeline, handling missing data, scaling features, encoding categorical variables, and establishing rigorous train/validation/test splits.

## Learning Objectives
* Identify and handle missing values using principled imputation strategies (MCAR/MAR/MNAR)[cite: 1].
* Apply feature scaling methods (Standardisation, Min-Max) and understand when each is appropriate[cite: 1].
* Encode categorical variables using One-Hot and Target Encoding[cite: 1].
* Mitigate data leakage by strictly computing statistics on the training set only[cite: 1].

## Concepts Covered
* Missing Data Imputation
* Z-score Normalization vs. Min-Max Scaling
* Categorical Encoding
* Class Imbalance (SMOTE)

## Connection to Textbook
The code strictly adheres to the textbook's warning regarding data leakage: all preprocessing statistics (means, standard deviations, etc.) are computed solely on the training set before being applied to the validation and test sets[cite: 1].
