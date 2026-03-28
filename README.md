# House Prices Prediction — Kaggle Competition

**Course:** UGST4088 — Introduction to Machine Learning and Data Mining 2025/26 Fall
**Institution:** Central European University, Vienna
**Author:** Zeteny Cseresznyes
**Kaggle Score:** 0.12205 (RMSLE)

## Overview

Predicts house sale prices from the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) using an ensemble of gradient boosting and linear models. The best submission uses a stacking regressor achieving **R² = 0.95** on the held-out test set.

## Contents

- `kaggle_house_prices_prediction_Cseresznyes.ipynb` — Full pipeline: EDA, feature engineering, model tuning, evaluation, and submission

## Pipeline

### 1. Data Preparation & Feature Engineering
- Log-transform skewed features (`SalePrice`, `LotArea`, `GrLivArea`, `1stFlrSF`)
- Create composite features: `TotalSF`, `TotalBaths`, `HouseAge`, `RemodAge`
- Drop highly correlated features (`TotRmsAbvGrd`, `GarageCars`)
- Ordinal encoding of categorical features; binary indicators for optional amenities
- Remove columns with >10% missing values; impute remaining with mean/0

### 2. Hyperparameter Tuning
- Optuna with 20 trials per model, 5-fold cross-validation

### 3. Models

| Model | Test R² |
|-------|---------|
| Random Forest | 0.87 |
| XGBoost | 0.90 |
| Gradient Boosting | 0.89 |
| Lasso | 0.92 |
| Averaged Ensemble | 0.90 |
| **Stacking Regressor** | **0.95** |

Final predictions rounded to the nearest thousand.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost optuna joblib
```

## Author

- LinkedIn: [zeteny-cseresznyes](https://www.linkedin.com/in/zeteny-cseresznyes-236ba8329/)
- GitHub: [cseresznyeszeteny](https://github.com/cseresznyeszeteny)
