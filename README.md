# Restaurant Rating Predictor — Flask Dashboard

Loads the 3 trained models (Random Forest, CatBoost, TF-IDF + Naive Bayes)
plus the saved label encoders, and serves a form-based UI that returns a
"High Rating" probability from each model side by side.

## 1. Get the model files

Run `ML_project_imputed_clean.py` (or `ML_project_without_imputed_clean.py`)
end to end, in Colab or locally. Its final save step now produces a
`models/` folder containing:

```
models/
  best_rf_classifier.pkl
  best_catboost_classifier.pkl
  tfidf_nb_pipeline.pkl
  label_encoders.pkl        <- new: needed for the dropdowns to work
  feature_list.pkl          <- new: needed so the app builds rows in the right order
```

Copy that whole `models/` folder into this project, next to `app.py`, so
the layout looks like:

```
flask_app/
  app.py
  requirements.txt
  models/            <- copied in from the training run
  templates/
    index.html
  static/
    style.css
```

## 2. Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## 3. Run inside Google Colab (ngrok, like your original notebook)

```python
!pip install flask pyngrok flask-ngrok --quiet

# In app.py, right after `app = Flask(__name__)`, add:
#     from flask_ngrok import run_with_ngrok
#     run_with_ngrok(app)
# and change the last line from app.run(debug=True) to app.run()

!python app.py
```

ngrok will print a public URL you can click to open the dashboard.

## How predictions are combined

Each model looks at the problem differently:
- **Random Forest** and **CatBoost** use the structured fields (votes,
  cost, location, restaurant type, listing type, online order, table
  booking).
- **TF-IDF + Naive Bayes** uses *only* the cuisine text you type in.

The dashboard shows all three individually, plus a simple average
("ensemble") as the headline number — since no single model is
definitively "correct," averaging reduces the impact of any one model's
blind spot (e.g., CatBoost being over-sensitive to vote count).

## Notes / things to be aware of

- If you pick a location/restaurant type/listing type the model never saw
  during training, it's automatically routed to an "Unknown" bucket
  rather than crashing — but the dropdowns are built directly from what
  each encoder learned, so in practice this only matters if you're
  scripting requests directly instead of using the form.
- `app.run(debug=True)` is fine for local development. Turn `debug` off
  (or run behind `gunicorn`) before deploying anywhere other than your
  own machine — debug mode exposes an interactive debugger that can
  execute arbitrary code.
