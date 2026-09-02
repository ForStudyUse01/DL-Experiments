# mlops-basics

Flask inference API for the Experiment 2 MNIST baseline ANN, containerized with Docker, tested
and built automatically via GitHub Actions on every push.

## Run locally

```bash
pip install -r requirements.txt
python app.py
# in another shell:
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
     -d "{\"pixels\": [0, 0, ... 784 floats in [0,1] ...]}"
```

## Run with Docker

```bash
docker build -t dl-lab-app:latest .
docker run -p 5000:5000 dl-lab-app:latest
```

## Tests

```bash
python -m pytest tests/ -v
```

CI runs these same steps (install, pytest, `docker build`) on every push that touches this folder
— see `.github/workflows/ci.yml` at the repo root and the Actions tab on GitHub.
