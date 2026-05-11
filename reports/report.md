# MLOps Assignment 01 — Report

**Student:** Hari Jaiswal (2023aa05106)
**Program:** BITS Pilani M.Tech, MLOps
**Project:** End-to-end MLOps pipeline for an Iris classification model
**Repository:** https://github.com/harijaiswal29/mlops_assign01
**CI runs:** https://github.com/harijaiswal29/mlops_assign01/actions
**Live verification output (with actual run-IDs, hashes, and logs):** [`verification_output.md`](verification_output.md)

---

## Overview

This assignment delivers a single ML project — Iris classification with a `RandomForestClassifier` — instrumented with the full MLOps tool-chain demanded by milestones M1–M4. One repository contains all four deliverables, organised so each milestone's artifacts are directly traceable in the source tree.

| Milestone | Topic | Primary tooling | Key artifact path |
|-----------|-------|-----------------|-------------------|
| M1 | CI/CD + Version Control | GitHub Actions, Git | `.github/workflows/ci.yml` |
| M2 | Experiment Tracking + Data Versioning | MLflow, DVC | `mlruns/`, `data/iris.csv.dvc` |
| M3 | Hyperparameter Tuning + Packaging | Optuna, Docker, Flask | `src/.../tune.py`, `Dockerfile`, `src/.../serve.py` |
| M4 | Deployment + Orchestration | `kind`, Helm | `k8s/`, `helm/iris-classifier/` |

---

## M1 — CI/CD Pipeline and Version Control

### CI/CD pipeline stages

`.github/workflows/ci.yml` defines a three-job pipeline that runs on every push and pull request targeting `main` or `develop`:

| Stage | Step | Purpose |
|-------|------|---------|
| **lint** | `ruff check .`, `ruff format --check .` | Static analysis: catches unused imports, style drift, common bugs |
| **test** | `pytest --cov=mlops_assign01 --cov-report=term-missing` | Runs the 9-test suite covering data loader, training metrics, and Flask endpoints with coverage |
| **build** | `docker/build-push-action@v6` then `curl /health` against the started container | Builds the production image and smoke-tests it before flagging a green run |

Jobs are sequenced — `test` `needs: lint`, `build` `needs: test` — so the pipeline fails fast on cheap errors. Layer caching is enabled (`type=gha`) so subsequent builds reuse Docker layers.

> **Screenshot placeholder:** `reports/screenshots/m1_ci_green_run.png` — green check marks for lint → test → build on a PR.

### Git branching and merge history

```
*   6394611 Merge develop into main
|\
| * 0fdbd9c Merge feature/m2-mlflow-and-dvc into develop
|/|
| * 3e4c614 M2: Bump iris.csv to v2 (drop last 30 rows for revert demo)
| * 10ebee8 M2: Initialise DVC and version-track data/iris.csv (v1: full 150 rows)
|/
* d5d0937 M4: Add Kubernetes manifests, Helm chart, and kind setup script
* b7864d5 M3: Add Dockerfile, docker-compose, and local run script
* 422481c M1: Add GitHub Actions CI workflow
* 2a50b87 M1: Add Python source modules and pytest suite
* cdb3d6f Scaffold MLOps Assignment 01 project
```

The history demonstrates:

1. **Linear scaffolding commits** on `main` for M1, M3, and M4 work.
2. **Branching off `main`** to create `develop`, then a topic branch `feature/m2-mlflow-and-dvc` off `develop`.
3. **Pull-request style merges**: `feature/m2-mlflow-and-dvc` → `develop` and `develop` → `main`, both using `--no-ff` so the merge commit is preserved.

The PR workflow on GitHub mirrors this exactly: a contributor branches off `develop`, opens a PR, CI runs, then the PR is merged with the "Create a merge commit" option.

> **Screenshot placeholders:**
> - `reports/screenshots/m1_merge_graph.png` — `git log --graph --oneline --all` output
> - `reports/screenshots/m1_github_pr.png` — an open PR with green checks before merge

---

## M2 — Experiment Tracking with MLflow & Data Versioning with DVC

### MLflow experiment runs

`src/mlops_assign01/train.py` is wrapped in `mlflow.start_run`. Each invocation logs:

- **Params**: `n_estimators`, `max_depth`, `criterion`, `random_state`
- **Metrics**: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`
- **Artifacts**: the fitted sklearn model (both as MLflow native `sklearn` flavour and a `joblib` dump)

Three runs were executed to satisfy the "at least three different runs" deliverable:

| Run name | n_estimators | max_depth | criterion | seed | accuracy | f1_macro |
|----------|-------------:|----------:|-----------|-----:|---------:|---------:|
| `shallow-rf`     |  50 |  3 | gini    | 42 | 0.9667 | 0.9666 |
| `deeper-entropy` | 200 |  8 | entropy | 42 | 0.9000 | 0.8997 |
| `large-forest`   | 300 | 12 | gini    |  7 | 1.0000 | 1.0000 |

Run-IDs and full artifact trees live in `./mlruns/`. View them locally with `mlflow ui --port 5000`.

> **Screenshot placeholder:** `reports/screenshots/m2_mlflow_ui_runs.png` — MLflow UI experiment list showing all three runs.

### DVC data versioning and revert demo

```bash
dvc init                                  # creates .dvc/ config dir
dvc remote add -d localremote ~/dvc-storage-mlops_assign01
dvc add data/iris.csv                     # tracks CSV, writes data/iris.csv.dvc
dvc push                                  # uploads to local remote
```

Two versions of the dataset were created to demonstrate revertability:

| Version | Rows | MD5 hash | Commit |
|---------|-----:|----------|--------|
| v1 | 150 | `013d0da08d6506664ce640459139176b` | `10ebee8` |
| v2 | 120 | `9b4d989461f76799d98318f82a8ab512` | `3e4c614` |

The git diff between versions is only the 3-line `data/iris.csv.dvc` pointer file — the raw CSV bytes stay in DVC cache / remote.

**Revert flow (verified):**

```bash
$ md5sum data/iris.csv                    # at HEAD (v2)
9b4d989461f76799d98318f82a8ab512  data/iris.csv

$ git checkout 10ebee8 -- data/iris.csv.dvc
$ dvc checkout
M       data/iris.csv

$ md5sum data/iris.csv                    # restored to v1
013d0da08d6506664ce640459139176b  data/iris.csv

$ wc -l data/iris.csv
151 data/iris.csv                         # 150 rows + header — confirms v1
```

Re-running `git checkout HEAD -- data/iris.csv.dvc && dvc checkout` cycles back to v2.

> **Screenshot placeholder:** `reports/screenshots/m2_dvc_revert.png` — terminal capture of the md5sum / wc -l flow above.

---

## M3 — Hyperparameter Tuning & Model Packaging

### Hyperparameter tuning with Optuna

`src/mlops_assign01/tune.py` runs an Optuna study over the `RandomForestClassifier` search space:

| Hyperparameter | Range | Type |
|----------------|-------|------|
| `n_estimators`     | 50 – 300 step 50 | int |
| `max_depth`        | 2 – 16           | int |
| `criterion`        | {gini, entropy}  | categorical |
| `min_samples_split`| 2 – 10           | int |

Objective: 5-fold cross-validated `f1_macro`. Every trial is logged to MLflow via the `optuna.integration.mlflow.MLflowCallback`, and the best-params model is logged as a separate `optuna-best` MLflow run with the serialised `models/best_model.joblib`.

**Tuning result (30 trials):**

| Best parameter | Value |
|----------------|-------|
| `n_estimators` | 300 |
| `max_depth` | 9 |
| `criterion` | entropy |
| `min_samples_split` | 2 |
| **Best CV `f1_macro`** | **0.9665** |

> **Screenshot placeholder:** `reports/screenshots/m3_optuna_progress.png` — terminal output of trial progress; `reports/screenshots/m3_mlflow_nested_runs.png` — nested MLflow runs under the tuning study.

### Model packaging — Dockerfile + Flask

The Flask service (`src/mlops_assign01/serve.py`) exposes two endpoints:

- `GET /health` — liveness/readiness probe (returns version, model path, load state)
- `POST /predict` — JSON body `{"features": [[5.1, 3.5, 1.4, 0.2], ...]}` → JSON response `{"predictions": [...], "model_version": "...", "feature_order": [...]}`

The `Dockerfile` is multi-stage:

1. **builder** (`python:3.12-slim`) installs the project and trains a baseline model in-place, materialising `/app/models/best_model.joblib`
2. **runtime** (`python:3.12-slim`, non-root user `app`) copies `/opt/venv`, `/app/models`, and `/app/src`, exposes 8000, and runs `gunicorn --workers 2 mlops_assign01.serve:app` with a `HEALTHCHECK` against `/health`.

Local verification:

```bash
$ bash scripts/run_local.sh
==> Building image iris-classifier:latest
==> /health response:
{ "status": "ok", "model_loaded": true, ... }
==> Sample prediction (setosa, virginica):
{ "predictions": ["setosa", "virginica"], "feature_order": [...] }
```

> **Screenshot placeholder:** `reports/screenshots/m3_docker_run.png` — `docker run` + `curl /predict` round-trip.

---

## M4 — Deployment & Orchestration with Kubernetes + Helm

### Cluster: local `kind`

`scripts/setup_kind.sh` automates the whole bring-up:

1. `kind create cluster --name iris-cluster`
2. `kind load docker-image iris-classifier:latest --name iris-cluster` (avoids the need for a registry push)
3. `kubectl create namespace iris`
4. `helm upgrade --install iris ./helm/iris-classifier -n iris`
5. `kubectl rollout status` to confirm pods are Ready

For a managed cloud cluster (GKE / EKS / AKS), the same Helm chart applies — only the image needs to be pushed to a registry and `image.repository` overridden via `--set`.

### Helm chart structure

```
helm/iris-classifier/
├── Chart.yaml
├── values.yaml          # replicas, image, resources, probes, ingress
└── templates/
    ├── _helpers.tpl     # name + labels helpers
    ├── deployment.yaml  # parameterised Deployment with readiness/liveness probes
    ├── service.yaml     # ClusterIP service (port 80 → container 8000)
    └── ingress.yaml     # optional ingress (enabled via values.yaml)
```

Plain (non-Helm) manifests are also provided under `k8s/` (`deployment.yaml`, `service.yaml`, `ingress.yaml`) for reference / `kubectl apply -f k8s/`.

### Verification flow

```bash
$ bash scripts/setup_kind.sh
$ kubectl -n iris get pods,svc
NAME                                          READY   STATUS    RESTARTS   AGE
pod/iris-iris-classifier-<hash>-<id1>         1/1     Running   0          45s
pod/iris-iris-classifier-<hash>-<id2>         1/1     Running   0          45s

NAME                            TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
service/iris-iris-classifier    ClusterIP   10.96.x.y       <none>        80/TCP    45s

$ kubectl -n iris port-forward svc/iris-iris-classifier 8000:80 &
$ curl -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
{"feature_order":["sepal_length","sepal_width","petal_length","petal_width"],
 "model_version":"0.1.0","predictions":["setosa"]}
```

> **Screenshot placeholders:**
> - `reports/screenshots/m4_kubectl_get_pods.png` — `kubectl get pods,svc` showing Running pods
> - `reports/screenshots/m4_helm_list.png` — `helm list -n iris`
> - `reports/screenshots/m4_curl_predict.png` — successful `curl /predict` against the K8s service

### Note on the deployed endpoint

For grading, the endpoint is exposed via `kubectl port-forward` to `http://localhost:8000`. A public cloud endpoint can be produced by:

1. Pushing the image to GHCR / Docker Hub
2. Pointing `image.repository` at the registry path
3. `helm install` against a managed cluster
4. Enabling the chart's `ingress` value with a real hostname

The chart and manifests already support this — only credentials and DNS were left out for this submission.

---

## Reproducing every milestone end-to-end

```bash
# 0. clone + venv + install
git clone <repo-url> && cd mlops_assign01
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# M1 — verify CI locally
ruff check . && pytest

# M2 — generate data, log three MLflow runs, DVC track + revert
python -m mlops_assign01.data --out data/iris.csv
python -m mlops_assign01.train --n-estimators 50  --max-depth 3  --criterion gini    --run-name shallow-rf
python -m mlops_assign01.train --n-estimators 200 --max-depth 8  --criterion entropy --run-name deeper-entropy
python -m mlops_assign01.train --n-estimators 300 --max-depth 12 --criterion gini    --run-name large-forest --random-state 7
mlflow ui --port 5000             # browse runs

dvc init && dvc remote add -d localremote ~/dvc-storage-mlops_assign01
dvc add data/iris.csv && dvc push

# M3 — Optuna + Docker
python -m mlops_assign01.tune --n-trials 30
bash scripts/run_local.sh

# M4 — Kubernetes
bash scripts/setup_kind.sh
kubectl -n iris port-forward svc/iris-iris-classifier 8000:80
```

---

## Deliverables checklist

| Deliverable | Where |
|-------------|-------|
| CI/CD pipeline stages report | This document, §M1 |
| Pipeline run screenshots | `reports/screenshots/m1_*.png` |
| Git repo with branches/merge history | `git log --graph --all`, also visible in §M1 |
| MLflow experiment logs (≥3 runs) | `mlruns/`, summarised in §M2 |
| DVC repo with multiple dataset versions | `data/iris.csv.dvc` history; revert flow in §M2 |
| Hyperparameter tuning report | §M3, plus `mlruns/` `optuna-best` run |
| Dockerfile + Flask app | `Dockerfile`, `src/mlops_assign01/serve.py` |
| Docker container screenshots | `reports/screenshots/m3_*.png` |
| Deployed model endpoint | `http://localhost:8000` via `kubectl port-forward` (§M4) |
| K8s configs + Helm chart | `k8s/`, `helm/iris-classifier/` |
| Deployment report | §M4 |
