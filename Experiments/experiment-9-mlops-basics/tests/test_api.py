"""Tests for the MNIST inference API (app.py)."""
import numpy as np
import pytest

from app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_returns_valid_digit(client):
    pixels = np.zeros(784).tolist()
    response = client.post("/predict", json={"pixels": pixels})
    assert response.status_code == 200

    body = response.get_json()
    assert 0 <= body["digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_bad_input(client):
    response = client.post("/predict", json={"pixels": [1, 2, 3]})
    assert response.status_code == 400
