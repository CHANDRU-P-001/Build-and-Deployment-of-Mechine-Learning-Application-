
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)

MODEL_PATH = "models/house_model.joblib"
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

model = joblib.load(MODEL_PATH)

with open("models/metrics.json") as file:
    metrics = json.load(file)


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object."}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": "Missing features: " + ", ".join(missing)
            }), 400

        values = [[float(data[f]) for f in FEATURES]]

        if not all(pd.notna(v) for v in values[0]):
            return jsonify({"error": "Values must be finite numbers."}), 400

        if not all(abs(v) < float("inf") for v in values[0]):
            return jsonify({"error": "Values must be finite numbers."}), 400

        input_data = pd.DataFrame(values, columns=FEATURES)
        prediction = float(model.predict(input_data)[0])

        return jsonify({
            "predicted_price": round(prediction * 100000, 2),
            "currency": "USD",
            "note": "Dataset-based estimate, not a live market valuation."
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Please enter valid numeric values."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
