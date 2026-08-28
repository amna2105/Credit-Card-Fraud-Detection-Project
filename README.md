# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset (fraud makes up only ~0.17% of transactions).

## Overview

This project walks through the full workflow of building a fraud-detection classifier:

- Exploratory data analysis on transaction patterns and class imbalance
- A naive baseline model, and why accuracy is the wrong metric for this problem
- Handling imbalance with **SMOTE** (oversampling) and **Random Undersampling**
- Comparing **Random Forest** vs **XGBoost**
- **Threshold tuning** to control the precision/recall trade-off

📓 **[View the full notebook](./credit_card_fraud_detection.ipynb)**

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) — 284,807 European cardholder transactions from September 2013. Features `V1`–`V28` are anonymized via PCA; `Time` and `Amount` are raw.

> The dataset (~150 MB) is **not included** in this repo due to size. Download it from Kaggle and place `creditcard.csv` in the project root before running the notebook.

## Results Summary

| Model | Resampling | Recall (Fraud) | Precision (Fraud) |
|---|---|---|---|
| Random Forest (baseline) | None | Moderate | Low |
| Random Forest | SMOTE | 0.88 | 0.60 |
| Random Forest | Random Undersampling | 0.92 | 0.04 |
| **XGBoost** | SMOTE | **0.89** | **0.53** |

XGBoost with SMOTE-resampled training data gave the strongest overall precision/recall balance. See the notebook for the full analysis and a threshold-tuning walkthrough.

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `XGBoost` · `matplotlib` · `seaborn`

## Setup

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into the project root, then open `credit_card_fraud_detection.ipynb` in Jupyter.

## Key Learnings

- Accuracy is misleading on imbalanced data — a model predicting "not fraud" every time scores 99.8% accuracy while catching zero fraud.
- Resampling strategy matters: SMOTE preserved more information than undersampling and produced a better precision/recall balance.
- Gradient boosting (XGBoost) outperformed Random Forest on this tabular data.
- Threshold tuning is a cheap way to adapt a trained model to different business cost trade-offs without retraining.

## Future Improvements

- Try `SMOTE + Tomek Links` or `ADASYN`
- Hyperparameter tuning with cross-validation
- Threshold tuning for the XGBoost model
- Precision-recall and ROC-AUC curves
- Wrap the best model behind a simple prediction API

## Acknowledgements

Base approach adapted from a [GeeksforGeeks](https://www.geeksforgeeks.org/) tutorial, extended with stratified resampling, XGBoost, and threshold tuning.
