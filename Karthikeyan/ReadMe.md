
---

# 🍕 Zomato Restaurant Rating Prediction Pipeline

A end-to-end Machine Learning pipeline built using **scikit-learn**, **Pandas**, and **SQLAlchemy**. This repository extracts raw restaurant dataset from a PostgreSQL database, executes data cleaning/preprocessing, implements custom multi-label transformers, trains k-Nearest Neighbors (k-NN) classification models with group-aware cross-validation, and serializes production-ready artifacts.

---

## 📌 Project Overview

Predicting restaurant rating tiers and exact rating classes based on operational attributes, cuisine variations, cost estimates, and location indicators.

### Key Highlights

* **Automated DB Ingestion**: Pulls raw data directly from hosted PostgreSQL instance (`Aiven Cloud`).
* **Custom Scikit-Learn Estimator**: Features a `MultiLabelColumnBinarizer` wrapped inside a `Pipeline` to seamlessly handle list-like categorical attributes (e.g., `cuisines_list`, `rest_type_list`).
* **Leakage-Free Splitting**: Implements **`GroupShuffleSplit`** and **`GroupKFold`** based on restaurant names (`name`) ensuring branches/outlets of the same chain stay strictly within either the train or test set.
* **Controlled Model Experiments**:
* **Model A (Imputed)**: Uses median and most-frequent strategy imputation via `SimpleImputer` on all available records.
* **Model B (Raw / Complete-Case)**: Trained exclusively on rows with full records to evaluate the raw signal efficiency against imputed baseline data.


* **Fair Evaluation Protocol**: Compares imputed vs. complete-case models on the exact same shared complete-case evaluation test set.

---
---

## 🛠️ Data Pipeline & Preprocessing

1. **Database Extraction**: Reads 51,717 rows from the `zomato_raw_data` table.
2. **Column Dropping & Numeric Parsing**:
* Removed identifier/free-text fields: `url`, `address`, `phone`, `menu_item`, `location`.
* Stripped and parsed float values for `rate` (e.g., `"4.1/5"` $\rightarrow$ `4.1`).
* Cleaned formatting commas from `approx_cost(for two people)`.


3. **Filtering & Subsampling**:
* Rows with missing `cuisines` or target `rate` dropped.
* Downsampled zero-vote entries to **10%** of their population to reduce noise and imbalance.


4. **Target Formulation**:
* **5-Class Target (`rate_class_5`)**: Rounded numerical rating mapped to integer classes `[1, 2, 3, 4, 5]`.
* **Tier Target (`rate_class_tier`)**: Bucketized into broader categories:
* `Low`: Rating $< 3.0$
* `Average`: $3.0 \le \text{Rating} < 4.0$
* `High`: Rating $\ge 4.0$





---

## 🧪 Pipeline Architecture

Feature transformations are dynamically executed inside a unified `ColumnTransformer`:

* **Numerical Features** (`votes`, `approx_cost`): Imputed (Median) $\rightarrow$ Standardized (`StandardScaler`).
* **Categorical Features** (`online_order`, `book_table`, `listed_in(type)`, `listed_in(city)`): Imputed (Most Frequent) $\rightarrow$ One-Hot Encoded (`OneHotEncoder`).
* **Multi-Label Features** (`cuisines_list`, `rest_type_list`): Binarized using custom `MultiLabelColumnBinarizer`.

```text
Features
  ├── Numerical Features    --> [SimpleImputer] --> [StandardScaler] ----------┐
  ├── Categorical Features  --> [SimpleImputer] --> [OneHotEncoder] -----------┼--> [ColumnTransformer] --> [KNeighborsClassifier]
  └── Multi-Label Lists     --> [MultiLabelColumnBinarizer] -------------------┘

```

---

## 📊 Experimental Results

Both models were grid-searched across $k \in \{3, 5, 7, 9, 11, 15\}$ and weights $\in \{\text{'uniform'}, \text{'distance'}\}$ using 5-Fold GroupKFold CV.

### Fair Comparison on Shared Complete-Case Test Set ($N = 8,088$)

| Target Formulation | Model Variant | Best Hyperparameters | Test Accuracy | Macro F1-Score |
| --- | --- | --- | --- | --- |
| **5-Class Target** | Model A (Imputed) | `n_neighbors: 3, weights: 'uniform'` | **67.46%** | **0.2927** |
| **5-Class Target** | Model B (Raw) | `n_neighbors: 15, weights: 'uniform'` | **70.56%** | **0.2877** |
| **3-Tier Target** | Model A (Imputed) | `n_neighbors: 15, weights: 'distance'` | **75.53%** | **0.4896** |
| **3-Tier Target** | Model B (Raw) | `n_neighbors: 15, weights: 'distance'` | **75.54%** | **0.4898** |

*Note: Macro F1-Score was evaluated across all predefined class labels without class dropping, ensuring strict reporting on rare/unrepresented classes.*

---

## 💻 Environment Setup & Usage

### 1. Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib sqlalchemy psycopg2-binary

```

### 2. Environment Variables for DB Connection

Before running database extraction, set your PostgreSQL secret password (or configure Google Colab secrets):

```python
import os
os.environ["DB_PASSWORD"] = "your_aiven_db_password"

```

### 3. Loading Saved Models for Inference

When reloading saved pickle models, you **must** import or define the custom class transformer `MultiLabelColumnBinarizer` prior to calling `joblib.load()`.

```python
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# Define the custom transformer
class MultiLabelColumnBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0]
        self.mlb_ = MultiLabelBinarizer()
        self.mlb_.fit(col)
        return self
        
    def transform(self, X):
        col = X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0]
        return self.mlb_.transform(col)
        
    def get_feature_names_out(self, input_features=None):
        return np.array([f"label_{c}" for c in self.mlb_.classes_])

# Load trained model
model = joblib.load("zomato_knn_A_imputed_tier.pkl")

# Make predictions on preprocessed DataFrame input
# predictions = model.predict(X_new)

```

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
