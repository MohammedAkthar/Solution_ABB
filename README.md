# Big Mart Sales Prediction

This repository contains an end-to-end machine learning pipeline to predict **Item_Outlet_Sales** using the Big Mart dataset. The project covers data cleaning, feature engineering, model training, evaluation, and final prediction generation.

---

## 📌 Project Overview

The objective of this project is to build a robust regression model that predicts product sales across different outlets. The pipeline is designed to:

* Prevent data leakage
* Handle missing values correctly
* Apply consistent feature transformations
* Train and evaluate multiple models

The final output is a CSV file with predicted sales values.

---

## 🗂 Project Structure

```
.
├── train.csv
├── test.csv
├── train_processed.csv
├── test_processed.csv
├── big-mart-submission.csv
├── main.py
└── README.md
```

---

## ⚙️ Key Components

### 1. DataPreparation Class

Handles all data preprocessing steps while avoiding data leakage.

**Includes:**

* Data cleaning and normalization
* Feature engineering
* Missing value imputation (fitted only on training data)
* Feature binning
* Label encoding
* Feature scaling

All transformations are fitted on the training dataset and applied consistently to both train and test data.

---

### 2. ModelTrainer Class

Responsible for training, evaluating, and comparing models.

**Models supported:**

* Random Forest Regressor
* XGBoost Regressor

**Features:**

* Optional hyperparameter tuning using RandomizedSearchCV
* Model evaluation using RMSE, MAE, R², and MAPE
* Feature importance visualization
* Prediction generation and submission file creation

---

### 3. Training Pipeline

The main pipeline performs the following steps:

1. Load train and test datasets
2. Apply data transformation and feature engineering
3. Perform imputation, binning, encoding, and scaling
4. Split training data into train/validation sets
5. Train models and evaluate performance
6. Generate predictions for test data

---

## 📊 Evaluation Metrics

The following metrics are used to evaluate model performance:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score
* MAPE (Mean Absolute Percentage Error)

Metrics are reported for both training and test sets.

---

## 📈 Feature Importance

For tree-based models, feature importance plots are generated and saved as PNG files to help interpret model behavior.

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

2. Place `train.csv` and `test.csv` in the project root.

3. Run the script:

```bash
python main.py
```

4. Output files generated:

* `train_processed.csv`
* `test_processed.csv`
* `big-mart-submission.csv`
* Feature importance plots

---

## 📝 Notes

* Negative sales predictions are clipped to zero.
* The pipeline is modular and easy to extend.
* Hyperparameter tuning can be enabled or disabled as needed.

---

## ✅ Future Improvements

* Add cross-validation reporting
* Integrate MLflow for experiment tracking
* Add model comparison summary
* Deploy as an API

---

Feel free to fork this repository and adapt the pipeline for similar regression problems.
