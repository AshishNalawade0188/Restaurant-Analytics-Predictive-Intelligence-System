"""
Loading and inference helpers for the Zomato Rate Predictors.

The pickles in artifacts/ are full sklearn Pipelines:
- Ridge Regression: artifacts/dropna_ridge_regression_model.pickle
- Classification: artifacts/dt_classifier_without_imputation.pkl
"""
import json
import pickle
import joblib
from functools import lru_cache
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "dropna_ridge_regression_model.pickle"
CLASSIFICATION_MODEL_PATH = ARTIFACTS_DIR / "dt_classifier_without_imputation.pkl"
CATEGORIES_PATH = ARTIFACTS_DIR / "categories.json"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "feature_schema.json"

# Feature columns expected by Regression Model
REGRESSION_FEATURE_COLUMNS = [
    "online_order",
    "book_table",
    "votes",
    "rest_type",
    "dish_liked",
    "cuisines",
    "approx_cost(for two people)",
    "listed_in(type)",
]

# Feature columns expected by Classification Model (matching its fit time features)
CLASSIFICATION_FEATURE_COLUMNS = [
    "online_order",
    "book_table",
    "votes",
    "location",
    "approx_cost(for two people)",
]


@lru_cache(maxsize=1)
def load_model():
    """Returns the fitted sklearn Ridge Regression Pipeline, or None if missing."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_classification_model():
    """Returns the fitted sklearn Classification Pipeline, or None if missing."""
    if not CLASSIFICATION_MODEL_PATH.exists():
        return None
    with open(CLASSIFICATION_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_schema():
    if not SCHEMA_PATH.exists():
        return {}
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_categories():
    """Returns full category lists for dropdowns including location."""
    categories = {}

    if CATEGORIES_PATH.exists():
        with open(CATEGORIES_PATH) as f:
            categories = json.load(f)

    # Fallback to schema if categories.json is not present
    if not categories:
        schema = load_schema()
        if schema:
            fmap = {f["name"]: f for f in schema.get("features_in_order", [])}
            categories = {
                "rest_type": fmap.get("rest_type", {}).get("valid_values", []),
                "dish_liked": fmap.get("dish_liked", {}).get("valid_values_top_10_by_frequency", []),
                "cuisines": fmap.get("cuisines", {}).get("valid_values_top_10_by_frequency", []),
                "listed_in(type)": fmap.get("listed_in(type)", {}).get("valid_values", []),
                "location": fmap.get("location", {}).get("valid_values", []),
            }

    # Ensure location dropdown list exists and is populated
    if "location" not in categories or not categories["location"]:
        categories["location"] = [
            "BTM",
            "Koramangala 5th Block",
            "HSR",
            "Indiranagar",
            "Jayanagar",
            "Whitefield",
            "JP Nagar",
            "Marathahalli",
            "Electronic City",
        ]

    # Ensure fallback categories for standard inputs
    if "rest_type" not in categories or not categories["rest_type"]:
        categories["rest_type"] = ["Casual Dining", "Quick Bytes", "Cafe"]
    if "dish_liked" not in categories or not categories["dish_liked"]:
        categories["dish_liked"] = ["Pasta", "Pizza", "Biryani"]
    if "cuisines" not in categories or not categories["cuisines"]:
        categories["cuisines"] = ["North Indian", "Chinese", "Italian"]
    if "listed_in(type)" not in categories or not categories["listed_in(type)"]:
        categories["listed_in(type)"] = ["Delivery", "Dine-out", "Cafes"]

    return categories


# Alias for backward compatibility
extract_categories = load_categories


def using_full_categories(categories: dict) -> bool:
    return len(categories.get("dish_liked", [])) > 10


def build_input_row(
    online_order: str,
    book_table: str,
    votes,
    rest_type: str,
    dish_liked: str,
    cuisines: str,
    approx_cost,
    listed_in_type: str,
) -> pd.DataFrame:
    """Input row builder for Regression Model."""
    row = {
        "online_order": 1 if online_order == "Yes" else 0,
        "book_table": 1 if book_table == "Yes" else 0,
        "votes": int(votes),
        "rest_type": rest_type,
        "dish_liked": dish_liked,
        "cuisines": cuisines,
        "approx_cost(for two people)": float(approx_cost),
        "listed_in(type)": listed_in_type,
    }
    return pd.DataFrame([row], columns=REGRESSION_FEATURE_COLUMNS)


def build_classification_input_row(
    online_order: str,
    book_table: str,
    votes,
    location: str,
    approx_cost,
    categories: dict = None,
) -> pd.DataFrame:
    online_val = 1 if str(online_order).strip().lower() in ["yes", "1", "true"] else 0
    book_val = 1 if str(book_table).strip().lower() in ["yes", "1", "true"] else 0

    # Extract location list from loaded categories if available for label encoding
    location_list = categories.get("location", []) if categories else []
    
    if location in location_list:
        location_code = location_list.index(location)
    else:
        location_code = 0  # Fallback index

    row = {
        "online_order": int(online_val),
        "book_table": int(book_val),
        "votes": int(votes),
        "location": int(location_code),  # Converted to numerical code
        "approx_cost(for two people)": float(approx_cost),
    }

    df = pd.DataFrame([row])
    return df[CLASSIFICATION_FEATURE_COLUMNS]


def predict_rate(model, input_row: pd.DataFrame) -> float:
    """Predicts numerical rating bounded between 0.0 and 5.0."""
    pred = model.predict(input_row)[0]
    return float(min(5.0, max(0.0, pred)))


def predict_classification_category(model, input_row: pd.DataFrame) -> str:
    """Predicts category tier and maps binary outputs to human-readable labels."""
    prediction = model.predict(input_row)[0]
    
    # Handle integer, float, or numeric string outputs safely
    pred_val = int(prediction)
    
    mapping = {
        0: "High Rating (>=3.8)",
        1: "Low Rating (< 3.8)"
    }
    
    return mapping.get(pred_val, f"Category {pred_val}")


# Function Alias for backwards compatibility
predict_category = predict_classification_category