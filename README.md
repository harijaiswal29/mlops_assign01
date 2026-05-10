# MLOps Assignment 01 — Iris Classification

End-to-end MLOps pipeline demonstrating CI/CD, experiment tracking, data versioning, hyperparameter tuning, containerization, and Kubernetes deployment for an Iris classification model.

Submitted for **BITS Pilani M.Tech — MLOps (Assignment 01)**.

## Milestones covered

| ID | Topic | Tools |
|----|-------|-------|
| M1 | CI/CD + Version Control | GitHub Actions, Git |
| M2 | Experiment Tracking + Data Versioning | MLflow, DVC |
| M3 | Hyperparameter Tuning + Packaging | Optuna, Docker, Flask, Gunicorn |
| M4 | Deployment + Orchestration | Kubernetes (`kind`), Helm |

The full write-up with screenshots is in [`reports/report.md`](reports/report.md).

## Quickstart

```bash
# 1. Create venv & install
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Generate the dataset CSV (M2 baseline)
python -m mlops_assign01.data --out data/iris.csv

# 3. Train a single model and log to MLflow
python -m mlops_assign01.train --n-estimators 100 --max-depth 5

# 4. Hyperparameter tuning with Optuna (M3)
python -m mlops_assign01.tune --n-trials 30

# 5. View experiments
mlflow ui --port 5000
```

## Running the model server

```bash
# Local (after training step above)
python -m mlops_assign01.serve

# Docker
docker build -t iris-classifier:latest .
docker run -p 8000:8000 iris-classifier:latest

# Predict
curl -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

## Kubernetes (M4)

```bash
bash scripts/setup_kind.sh        # creates cluster, loads image, helm install
kubectl get pods
kubectl port-forward svc/iris-classifier 8000:80
```

## Repo layout

```
src/mlops_assign01/   # data, train, tune, serve modules
tests/                # pytest suite
.github/workflows/    # CI pipeline
data/                 # DVC-tracked dataset
k8s/                  # plain Kubernetes manifests
helm/iris-classifier/ # Helm chart
scripts/              # setup_kind.sh, run_local.sh
reports/              # report.md + screenshots
```
