# model-deployment-api-ui

Flask REST API + Streamlit UI for the Experiment 2 MNIST digit-classification model.

## Run the API

```bash
pip install -r requirements.txt
python app.py
# Running on http://127.0.0.1:5000
```

## Run the UI

```bash
streamlit run ui.py
```

## Test the API endpoints

```bash
python scripts/generate_sample_image.py   # writes sample_images/sample_digit.png
python scripts/test_endpoints.py          # hits /health and /predict, writes results.csv
```

```bash
curl -X POST http://127.0.0.1:5000/predict -F "file=@sample_images/sample_digit.png"
```
