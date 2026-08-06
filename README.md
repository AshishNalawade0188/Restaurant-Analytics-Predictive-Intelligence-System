# Restaurant Analytics & Predictive Intelligence System

An end-to-end Machine Learning and Retrieval-Augmented Generation (RAG) platform that provides restaurant rating predictions, success classification, conversational AI analytics, and interactive dashboard visualisations using Zomato dataset insights.

---

## 🚀 Key Features

* **Rating Prediction (Regression):** Predicts exact restaurant ratings out of 5.0 using a Ridge Regression model.
* **Success Classification:** Classifies restaurants into High Success ($\ge 3.8$) or Low Rating categories using a Decision Tree classifier.
* **RAG Conversational Assistant:** Interactively query dataset insights using LlamaIndex, ChromaDB vector store, and Groq LLM integration.
* **Embedded Analytics:** Integrated Tableau dashboards for visual exploratory data analysis.

---

## 📁 Repository Structure

```text
Restaurant-Analytics-Predictive-Intelligence-System/
└── zomato_app/
    ├── artifacts/                   # Trained model pickles & JSON dropdown categories
    │   ├── categories.json
    │   ├── dropna_ridge_regression_model.pickle
    │   └── dt_classifier_without_imputation.pkl
    ├── data/                        # Dataset storage
    ├── src/                         # Core utility scripts and RAG pipeline
    │   ├── extract_categories.py
    │   ├── model_utils.py
    │   └── RAG_V3.py
    ├── static/                      # Styling assets
    ├── templates/                   # Frontend UI templates
    │   └── index.html
    ├── app.py                       # Flask application entry point
    ├── feature_schema.json          # Input feature definitions
    └── requirements.txt             # Python dependencies
