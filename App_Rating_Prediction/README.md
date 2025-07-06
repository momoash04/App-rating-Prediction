# 📱 App Rating Prediction – Kaggle Competition

This repository presents my solution to the **App Rating Competition** competition on Kaggle. The task is to predict app ratings (on a scale of 1 to 5) based on a variety of app-related features such as size, type, install count, release date, and Android version.

Two machine learning approaches were used:
- 🐱 CatBoost Regressor
- 🌲 Random Forest Regressor

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `CatBoostRegression.ipynb` | Full pipeline using CatBoost, including preprocessing, modeling, feature analysis, and submission generation |
| `RandomForesetModel.ipynb` | Model pipeline using Random Forest with advanced imputation and feature scaling |
| `train.csv` | Training data containing app features and target rating `Y` |
| `test.csv` | Test data to be predicted |
| `SampleSubmission.csv` | Template for submission file |
| `submission.csv` | Example of model output for competition submission |

---

## 🧠 Problem Overview

Given a dataset of Android apps, your task is to predict the user rating (`Y`) for each app based on:

- `X3`: App size (e.g. '25M', '3.5K', or 'Varies with device')
- `X4`: Number of installs (e.g. '10,000+')
- `X5`: Type (Free or Paid)
- `X9`: Last updated date (e.g. 'June 1, 2021')
- `X11`: Minimum Android version required (e.g. '4.0 and up')
- And additional data columns that were not used
---

## 🧹 Preprocessing Pipeline

### 🔧 Shared Steps
- App size (`X3`) converted from 'M'/'K' strings to float MB
- Install count (`X4`) cleaned of commas and '+' signs
- Type (`X5`) mapped to binary: Free → 0, Paid → 1
- Date (`X9`) parsed into "days since May 9, 2025"
- Android version (`X11`) extracted as float from strings like `'4.0 and up'`
- Missing values imputed with:
  - CatBoost version: `SimpleImputer`
  - Random Forest version: **Linear Regression** imputation using correlated features

---

## 🐱 CatBoost Regressor (see `CatBoostRegression.ipynb`)

- Model: `CatBoostRegressor`
- Iterations: 500  
- Learning Rate: 0.000455  
- Depth: 5  
- Evaluation:
  - R²
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Visualizations:
  - Correlation heatmap
  - Feature importance
  - Partial dependence plots

---

## 🌲 Random Forest Regressor (see `RandomForesetModel.ipynb`)

- Preprocessing includes advanced date and Android version handling
- Missing value imputation using **Linear Regression models**
- Model: `RandomForestRegressor` with:
  - `n_estimators=200`
  - `max_depth=6`
  - `min_samples_leaf=10`
  - `max_features=3`
- 5-Fold Cross-Validation with:
  - R² score
  - MAE
  - RMSE

---

## 📊 Sample Submission Format

Your final `submission.csv` file should look like:

```csv
row_id,Y
0,4.31
1,3.82
...
