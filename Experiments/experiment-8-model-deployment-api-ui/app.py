"""Flask REST API for the Experiment 2 MNIST digit-classification model."""
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path(__file__).parent / "models" / "baseline_ann.h5"

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH)


def preprocess_image(image_file):
    img = Image.open(image_file).convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 784)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    processed = preprocess_image(file)
    prediction = model.predict(processed, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "predicted_digit": predicted_class,
        "confidence": round(confidence, 4),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000)
