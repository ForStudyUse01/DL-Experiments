"""Flask inference API for the Experiment 2 MNIST baseline ANN model."""
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "models" / "baseline_ann.h5"

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    pixels = payload.get("pixels")

    if not isinstance(pixels, list) or len(pixels) != 784:
        return jsonify({"error": "expected 'pixels': a flat list of 784 floats in [0, 1]"}), 400

    x = np.array(pixels, dtype="float32").reshape(1, 784)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    return jsonify({"digit": digit, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
