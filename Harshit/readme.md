# Machine Learning Model Summaries

This document provides concise, interview-ready summaries for the **Decision Tree**, **LightGBM**, and **Support Vector Machine (SVM)** models developed in this project. Each summary is based on the final notebooks and covers the model's purpose, tuning process, and key results.

---

## 1. Decision Tree (`04_DecisionTree`)

The Decision Tree model was implemented for both regression and classification tasks using the cleaned and engineered dataset. The process involved establishing baseline models, analysing complexity through depth variation, and finally tuning hyperparameters to optimize performance and reduce overfitting.

### Regression
- **Model**: `DecisionTreeRegressor`
- **Tuning Strategy**: `GridSearchCV` was used to optimize hyperparameters.
- **Optimization Metric**: Root Mean Squared Error (RMSE).
- **Search Space**: `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **Best Parameters**:
    - `criterion`: squared_error
    - `max_depth`: 12
    - `min_samples_split`: 10
    - `min_samples_leaf`: 5
- **Best Cross-Validation RMSE**: `0.2663`

### Classification
- **Model**: `DecisionTreeClassifier`
- **Tuning Strategy**: `GridSearchCV` was used to optimize hyperparameters.
- **Optimization Metric**: Macro Recall (to address class imbalance).
- **Search Space**: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight`.
- **Best Parameters**:
    - `criterion`: entropy
    - `max_depth`: 12
    - `min_samples_split`: 10
    - `min_samples_leaf`: 5
    - `class_weight`: balanced
- **Best Cross-Validation Macro Recall**: `0.8029`

#### Summary
The tuned Decision Tree models demonstrated improved generalization with controlled tree depth, significantly reducing overfitting. They provided meaningful feature importance and balanced performance across both regression and classification tasks.

---

## 2. LightGBM (`05_LightGBM`)

LightGBM was implemented as a modern gradient-boosting framework to efficiently capture non-linear relationships and handle large datasets. The process included training baseline models followed by hyperparameter tuning.

### Regression
- **Model**: `LGBMRegressor`
- **Tuning Strategy**: `RandomizedSearchCV` to manage computational cost.
- **Optimization Metric**: RMSE.
- **Search Space**: `learning_rate`, `n_estimators`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, and `colsample_bytree`.
- **Best Parameters**:
    - `learning_rate`: 0.2
    - `n_estimators`: 500
    - `num_leaves`: 90
    - `max_depth`: 12
    - `min_child_samples`: 30
    - `subsample`: 0.8
    - `colsample_bytree`: 0.6
- **Best Cross-Validation RMSE**: `0.1685`

### Classification
- **Model**: `LGBMClassifier`
- **Tuning Strategy**: `RandomizedSearchCV`.
- **Optimization Metric**: Macro Recall (due to class imbalance).
- **Search Space**: `learning_rate`, `n_estimators`, `num_leaves`, `max_depth`, `min_child_samples`, and `class_weight`.
- **Best Parameters**:
    - `learning_rate`: 0.01
    - `n_estimators`: 200
    - `num_leaves`: 70
    - `max_depth`: 14
    - `min_child_samples`: 30
    - `class_weight`: balanced
- **Best Cross-Validation Macro Recall**: `0.2514`

#### Summary
LightGBM proved to be the strongest regression model in the project, showcasing its ability to model complex feature interactions. However, classification performance was moderate, highlighting the challenge posed by the severe class imbalance in the target variable.

---

## 3. Support Vector Machine (`06_SVM`)

The SVM implementation used standardized features for both **Support Vector Regression (SVR)** and **Support Vector Classification (SVC)**. The process included a sensitivity analysis for key parameters before optimizing with `RandomizedSearchCV`.

### Regression (SVR)
- **Model**: `SVR`
- **Tuning Strategy**: `RandomizedSearchCV` with 3-fold cross-validation.
- **Optimization Metric**: RMSE.
- **Sensitivity Analysis**: Studied the influence of `C` (regularization) and `gamma` (RBF kernel parameter).
- **Best Parameters**:
    - `C`: 1
    - `gamma`: scale
    - `epsilon`: 0.1
- **Best Cross-Validation RMSE**: `0.3177`

### Classification (SVC)
- **Model**: `SVC`
- **Tuning Strategy**: `RandomizedSearchCV`.
- **Optimization Metric**: Matthews Correlation Coefficient (MCC) - used for a robust evaluation of imbalanced multiclass classification.
- **Search Space**: `C`, `gamma`, and `class_weight`.
- **Key Results**: The tuned SVC showed a significant improvement over the baseline, achieving higher MCC, Weighted F1-score, and overall accuracy with minimal overfitting.

#### Summary
While the tuned SVR generalized well, its accuracy was lower compared to the tree-based models, indicating that ensemble methods were more suitable for this regression dataset. For classification, the SVM delivered one of the most balanced performances across all rating categories, making it the strongest classifier in terms of MCC.

---

## Overall Performance Comparison

| Model           | Task         | Optimized Metric          | Best CV Score |
|-----------------|--------------|---------------------------|---------------|
| Decision Tree   | Regression   | RMSE                      | 0.2663        |
| LightGBM        | Regression   | RMSE                      | **0.1685**    |
| SVM (SVR)       | Regression   | RMSE                      | 0.3177        |
| Decision Tree   | Classification| Macro Recall              | **0.8029**    |
| LightGBM        | Classification| Macro Recall              | 0.2514        |
| SVM (SVC)       | Classification| MCC                       | *Significant Improvement* |

**Conclusion**: Tree-based models (LightGBM and Decision Tree) are highly effective for this regression problem. For classification, the Decision Tree performed best in terms of Recall, while SVM provided the most balanced performance across classes.