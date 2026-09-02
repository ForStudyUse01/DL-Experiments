"""Tests the running Flask API's endpoints with real HTTP requests and records the results."""
import csv
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:5000"
SAMPLE_IMAGE = Path(__file__).parent.parent / "sample_images" / "sample_digit.png"
RESULTS_PATH = Path(__file__).parent.parent / "results.csv"

results = []


def timed_request(label, method, endpoint, **kwargs):
    start = time.time()
    response = requests.request(method, f"{BASE_URL}{endpoint}", **kwargs)
    elapsed_ms = (time.time() - start) * 1000
    print(f"{label}: {response.status_code} ({elapsed_ms:.1f} ms)")
    print(response.json())
    results.append({
        "endpoint": endpoint,
        "method": method,
        "case": label,
        "status_code": response.status_code,
        "response_time_ms": round(elapsed_ms, 1),
        "response_body": response.text,
    })
    return response


timed_request("health check", "GET", "/health")

with open(SAMPLE_IMAGE, "rb") as f:
    timed_request("predict (with file)", "POST", "/predict", files={"file": f})

timed_request("predict (no file)", "POST", "/predict")

with open(RESULTS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to {RESULTS_PATH}")
