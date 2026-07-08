# Titanic Survival Prediction

A machine learning project predicting passenger survival on the Titanic using Logistic Regression.

## Overview
- Cleaned real-world data with missing values (Age, Cabin, Embarked)
- Encoded categorical features (Sex, Embarked) 
- Trained a Logistic Regression classifier
- Achieved 81% accuracy on held-out test data

## Tech stack
Python, pandas, scikit-learn

## Steps
1. Data exploration (.info(), .describe(), missing value analysis)
2. Data cleaning (dropped high-missing columns, filled Age/Embarked)
3. Feature encoding (label encoding for Sex, one-hot encoding for Embarked)
4. Train/test split (80/20)
5. Model training & evaluation (Logistic Regression, accuracy score)

## Results
81% accuracy on test set