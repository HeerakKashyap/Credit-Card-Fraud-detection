# Credit Card Fraud Detection

## Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions. The goal is to accurately classify transactions as *fraudulent* or *genuine* using supervised learning, with a focus on handling severe class imbalance and evaluating with appropriate metrics.

---

## Dataset

- **Source:** [dataset.csv] (see repository)
- **Features:** 30 anonymized features (V1-V28, Amount, Time)
- **Target:** `Class` (1 = Fraudulent, 0 = Genuine)
- **Note:** The full dataset contains over 280,000 records. For computational efficiency and demonstration, we used a subset of approximately **70,000 entries**.

---

## Project Steps

### 1. Data Import & Preprocessing

- Imported the CSV dataset using `pandas`.
- Normalized the `Amount` and `Time` columns using `StandardScaler` to ensure all features are on a similar scale.
- Checked for and handled missing values (none found).

### 2. Feature Engineering

- Split the data into features (`X`) and target (`y`).

### 3. Train-Test Split

- Divided the subset into training and test sets (80:20 split) using stratification to preserve class distribution.

### 4. Handling Class Imbalance

- The dataset is highly imbalanced (fraud cases are rare).
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) only on the training data to generate synthetic samples of the minority class.
- Ensured the test set remains untouched for unbiased evaluation.

### 5. Model Selection & Training

- Chose **Random Forest Classifier** with `class_weight='balanced'` for robustness and interpretability.
- Trained the model on the resampled training data.

### 6. Evaluation

- Evaluated the model on the original (imbalanced) test set using:
  - **Precision**
  - **Recall**
  - **F1-score**
  - **ROC-AUC**
- Focused on recall and F1-score due to the business importance of minimizing false negatives (missed frauds).

---



