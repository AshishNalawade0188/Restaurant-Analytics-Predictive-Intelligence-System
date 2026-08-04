# Zomato Flask ML dashboard

The dashboard has separate tabs for rating regression and sentiment classification. Both use dropdowns for location, online order, and table booking.

## Run

```powershell
cd zomato_flask_app
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Add your models

1. Save each fitted scikit-learn pipeline in `models/`, for example:

   ```python
   import joblib
   joblib.dump(best_regressor, "models/rating_regressor.joblib")
   joblib.dump(sentiment_classifier, "models/sentiment_classifier.joblib")
   ```

2. Edit `model_registry.json`. `file` must match the filename in `models/`. The `inputs` field names and order must match the columns your pipeline expects.
3. Duplicate a registry entry to add further classification or regression models.

For sentiment classification, train with the `sentiment` column as `y`; the included labels map `0`, `1`, and `2` to `Bad`, `Good`, and `Excellent`.

The example configuration expects numeric `online_order` and `book_table` values (Yes = 1, No = 0). If your model was trained on `Yes`/`No` text, change the option values in the registry to those strings.
