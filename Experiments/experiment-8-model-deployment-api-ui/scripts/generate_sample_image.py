"""Saves a real MNIST test digit as a PNG for use in the API endpoint test."""
from pathlib import Path

from PIL import Image
from tensorflow import keras

OUT_PATH = Path(__file__).parent.parent / "sample_images" / "sample_digit.png"

(_, _), (X_test, y_test) = keras.datasets.mnist.load_data()

idx = 0
img = Image.fromarray(X_test[idx])
img.save(OUT_PATH)

print("Saved", OUT_PATH, "| true label:", int(y_test[idx]))
