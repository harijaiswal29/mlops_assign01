# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project is

End-to-end **MLOps Assignment 01** for BITS Pilani M.Tech, covering four milestones on a single Iris-classification ML project:

| Milestone | Topic | Primary tooling |
|-----------|-------|-----------------|
| M1 | CI/CD + Version Control | GitHub Actions, Git |
| M2 | Experiment Tracking + Data Versioning | MLflow, DVC |
| M3 | Hyperparameter Tuning + Packaging | Optuna, Docker, Flask, Gunicorn |
| M4 | Deployment + Orchestration | Kubernetes (`kind`), Helm |

Full milestone write-up: `reports/report.md`. Live verification capture: `reports/verification_output.md`. Public repo: https://github.com/harijaiswal29/mlops_assign01.

## Repo map

```
src/mlops_assign01/
  data.py     — Iris loader + CLI to write data/iris.csv. Holds shared constants
                FEATURE_COLUMNS, TARGET_COLUMN, EXPERIMENT_NAME, DEFAULT_MODEL_PATH.
  train.py    — Single MLflow-instrumented training run. CLI: --n-estimators,
                --max-depth, --criterion, --run-name.
  tune.py     — Optuna study (5-fold CV f1_macro). Logs every trial to MLflow
                via MLflowCallback; writes best model to models/best_model.joblib.
  serve.py    — Flask app. GET /health, POST /predict. Module-level `app` is
                the gunicorn entrypoint; create_app(model_path=...) is used in tests.
tests/        — pytest suite (9 tests, all green).
.github/workflows/ci.yml — lint → test → build Docker (with container smoke test).
Dockerfile    — multi-stage; trains a baseline model into the image; gunicorn runtime
                as non-root user `app` on port 8000.
k8s/          — plain Deployment/Service/Ingress for reference (kubectl apply -f k8s/).
helm/iris-classifier/ — parameterised Helm chart (used by setup_kind.sh).
scripts/run_local.sh   — build image, run container, hit /health and /predict.
scripts/setup_kind.sh  — create kind cluster, kind load docker-image, helm install.
data/iris.csv  — DVC-tracked (pointer at data/iris.csv.dvc; raw CSV gitignored).
```

## Common commands

All run from the project root, with the venv active (`source .venv/bin/activate`).

| Task | Command |
|------|---------|
| Lint | `ruff check . && ruff format --check .` |
| Tests | `pytest` |
| Materialise data CSV | `python -m mlops_assign01.data --out data/iris.csv` |
| Single MLflow training run | `python -m mlops_assign01.train --n-estimators 100 --max-depth 5 --run-name <name>` |
| Optuna tuning | `python -m mlops_assign01.tune --n-trials 30` |
| MLflow UI | `mlflow ui --port 5000` (browse `./mlruns`) |
| Flask dev server | `python -m mlops_assign01.serve --port 8000` |
| DVC push/pull | `dvc push` / `dvc pull` (local remote at `~/dvc-storage-mlops_assign01`) |
| Local Docker E2E | `bash scripts/run_local.sh` |
| K8s on kind | `bash scripts/setup_kind.sh` then `kubectl -n iris port-forward svc/iris-iris-classifier 8000:80` |

## Conventions

- **Branching**: feature work goes on `feature/<name>` branches off `develop` (or `main` for chores). Merge with `--no-ff` to keep the merge commit visible — the assignment grading depends on a readable `git log --graph --all`.
- **MLflow tracking URI**: defaults to `file:./mlruns` everywhere. Don't switch to a DB backend mid-flight; mlruns/ is the deliverable.
- **MLflow experiment name**: `iris-classifier` (constant in `src/mlops_assign01/data.py`).
- **Model artifact path**: `models/best_model.joblib`. Always overwritten by `train.py` and `tune.py`. The Docker image ships with a baseline trained at build time.
- **DVC remote**: local filesystem at `~/dvc-storage-mlops_assign01`. No cloud creds in play.
- **Helm release name**: `iris` in the `iris` namespace. Fullname template renders pods/svc as `iris-iris-classifier`.
- **Docker image tag**: `iris-classifier:latest`. The kind script loads this directly into the cluster (no registry).

## Environment quirks (WSL2 + Docker Desktop)

- The dev box is WSL2 (Ubuntu) under Windows. `docker` requires Docker Desktop's WSL integration enabled.
- `kind`, `kubectl`, `helm`, `gh` are installed to `~/.local/bin` (on PATH already).
- **Port 8000 is sometimes in use** by another local uvicorn dev server. If `curl localhost:8000` returns `{"detail":"Not Found"}` (FastAPI 404, not Flask's), publish to a different host port: `docker run -p 8001:8000 …` and `kubectl port-forward svc/iris-iris-classifier 8002:80`.
- For kind on WSL2: stick to `ClusterIP` services + `kubectl port-forward`. `LoadBalancer` services won't get an external IP.

## What NOT to do

- Don't expand scope into Airflow / Kubeflow / cloud deployment — the assignment doesn't require it, and locked-in choices are deliberate (see `~/.claude/projects/.../memory/project_mlops_assign01.md`).
- Don't switch MLflow to a SQLite/Postgres backend — the file-based `mlruns/` is the grading artifact.
- Don't commit `mlruns/`, `models/`, `data/iris.csv`, `.venv/`, `.dvc/cache/` — all gitignored by design.
- Don't run destructive cluster ops without checking with the user (`kind delete cluster`, `helm uninstall`).
- Don't auto-merge PRs on the public repo — the open PR is a screenshot deliverable.

## State of work (as of last session)

- All four milestones complete, pushed to GitHub, CI green on `main` and `develop`.
- PR #1 (https://github.com/harijaiswal29/mlops_assign01/pull/1) intentionally left open as the M1 "pull request" deliverable.
- `kind` cluster `iris-cluster` was live with 2/2 pods Running and a Helm release `iris/iris-iris-classifier`. May have been torn down since — check with `kind get clusters` before assuming.
- Remaining user-side: capture screenshots into `reports/screenshots/` per the placeholders in `reports/report.md`.
