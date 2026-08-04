from functools import lru_cache
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REGISTRY_PATH = BASE_DIR / "model_registry.json"

app = Flask(__name__)


def registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["models"]


def get_model_config(model_id):
    for config in registry():
        if config["id"] == model_id:
            return config
    abort(404, "Unknown model")


def model_path(config):
    path = (MODELS_DIR / config["file"]).resolve()
    if MODELS_DIR.resolve() not in path.parents:
        abort(400, "Model path must stay inside the models folder")
    return path


@lru_cache(maxsize=16)
def load_model(path):
    return joblib.load(path)


def prediction_input(config, form):
    values = {}
    for field in config["inputs"]:
        value = form.get(field["name"], "").strip()
        if value == "":
            raise ValueError(f"{field['label']} is required.")
        if field["type"] == "number":
            values[field["name"]] = float(value)
        elif field["type"] == "integer":
            values[field["name"]] = int(value)
        else:
            values[field["name"]] = value
    return pd.DataFrame([values], columns=[field["name"] for field in config["inputs"]])


@app.get("/")
def index():
    models = registry()
    return render_template(
        "index.html",
        regression_models=[model for model in models if model["task"] == "regression"],
        classification_models=[model for model in models if model["task"] == "classification"],
    )


@app.post("/predict/<model_id>")
def predict(model_id):
    config = get_model_config(model_id)
    try:
        path = model_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Add {config['file']} to the models folder first.")

        model = load_model(str(path))
        features = prediction_input(config, request.form)
        raw_prediction = model.predict(features)[0]

        if config["task"] == "regression":
            result = f"Predicted restaurant rating: {float(raw_prediction):.2f} / 5"
            score = None
        else:
            labels = config.get("labels", {})
            label = labels.get(str(raw_prediction), str(raw_prediction))
            result = f"Predicted sentiment: {label}"
            score = None
            if hasattr(model, "predict_proba"):
                score = float(max(model.predict_proba(features)[0]))

        return render_template("result.html", config=config, result=result, score=score)
    except (ValueError, FileNotFoundError) as error:
        return render_template("result.html", config=config, error=str(error)), 400


if __name__ == "__main__":
    app.run(debug=True)
