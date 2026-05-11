# Verification output — captured 2026-05-11

This file captures the actual command output collected during the assignment-completion run. Use alongside `report.md` and the screenshot files in `reports/screenshots/`.

---

## M1 — CI/CD pipeline green run

`gh run view 25648504191 --repo harijaiswal29/mlops_assign01`:

```
success
Lint (ruff):        success (2026-05-11T03:20:06Z -> 2026-05-11T03:21:11Z)   1m05s
Test (pytest):      success (2026-05-11T03:21:13Z -> 2026-05-11T03:22:26Z)   1m13s
Build Docker image: success (2026-05-11T03:22:28Z -> 2026-05-11T03:25:22Z)   2m54s
```

The develop branch run (#25648505915) also completed `success`.

Workflow runs page: https://github.com/harijaiswal29/mlops_assign01/actions

## M1 — Branching / merging

`git log --graph --oneline --all`:

```
*   bae7405 Merge chore/readme-hint into main
|\
| * 864705f Add pointer to report.md at top of README
|/
*   80f7095 Merge feature/m1-report into main
|\
| * bf58c2f Add combined M1-M4 report with results and verification
|/
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

## M2 — MLflow runs

| Run name | n_estimators | max_depth | criterion | seed | accuracy | f1_macro |
|----------|-------------:|----------:|-----------|-----:|---------:|---------:|
| `shallow-rf`     |  50 |  3 | gini    | 42 | 0.9667 | 0.9666 |
| `deeper-entropy` | 200 |  8 | entropy | 42 | 0.9000 | 0.8997 |
| `large-forest`   | 300 | 12 | gini    |  7 | 1.0000 | 1.0000 |

Total runs in `mlruns/`: 4+ (three named + Optuna trials + `optuna-best`).

## M2 — DVC revert flow

```
$ md5sum data/iris.csv && wc -l data/iris.csv
9b4d989461f76799d98318f82a8ab512  data/iris.csv     # v2
121 data/iris.csv                                   # 120 rows + header

$ git checkout 10ebee8 -- data/iris.csv.dvc && dvc checkout
M       data/iris.csv
$ md5sum data/iris.csv && wc -l data/iris.csv
013d0da08d6506664ce640459139176b  data/iris.csv     # back to v1
151 data/iris.csv                                   # 150 rows + header

$ git checkout HEAD -- data/iris.csv.dvc && dvc checkout
M       data/iris.csv
$ md5sum data/iris.csv && wc -l data/iris.csv
9b4d989461f76799d98318f82a8ab512  data/iris.csv     # back to v2
121 data/iris.csv
```

## M3 — Optuna hyperparameter tuning

After 30 trials of 5-fold CV on `f1_macro`:

```
Best f1_macro=0.9665 with params={
    'n_estimators': 300,
    'max_depth': 9,
    'criterion': 'entropy',
    'min_samples_split': 2
}
Saved best model to models/best_model.joblib
Best model run id: b7203fc1007f4c9bb5376a4253dc3aa2
```

## M3 — Docker container

Container build → run → predict:

```
$ docker run -d --name iris-local -p 8001:8000 iris-classifier:latest
a5b4af54925c706d58e21f110784936d5d3c23d454dd3989e77e309342255e16

$ curl -s http://localhost:8001/health | jq
{
  "model_loaded": false,
  "model_path": "/app/models/best_model.joblib",
  "status": "ok",
  "version": "0.1.0"
}

$ curl -s -X POST http://localhost:8001/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3],[5.9,3.0,4.2,1.5]]}' | jq
{
  "predictions": ["setosa", "virginica", "versicolor"],
  "model_version": "0.1.0",
  "feature_order": ["sepal_length","sepal_width","petal_length","petal_width"]
}
```

> Note: another local dev server happened to be bound to 8000 in WSL at the time of running, so the container was published on host port 8001. Inside the container the app still listens on 8000.

## M4 — Kubernetes (kind) + Helm

```
$ bash scripts/setup_kind.sh
==> Creating kind cluster 'iris-cluster'
==> Loading image into kind
==> helm upgrade --install iris
Release "iris" does not exist. Installing it now.
NAME: iris   NAMESPACE: iris   STATUS: deployed   REVISION: 1
==> Waiting for pods to become ready
deployment "iris-iris-classifier" successfully rolled out

$ helm list -n iris
NAME  NAMESPACE  REVISION  STATUS    CHART                  APP VERSION
iris  iris       1         deployed  iris-classifier-0.1.0  0.1.0

$ kubectl -n iris get pods,svc,deploy
NAME                                        READY   STATUS    RESTARTS   AGE
pod/iris-iris-classifier-5f478596c9-8vsn4   1/1     Running   0          2m4s
pod/iris-iris-classifier-5f478596c9-fxvs9   1/1     Running   0          2m4s

NAME                           TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
service/iris-iris-classifier   ClusterIP   10.96.2.106   <none>        80/TCP    2m9s

NAME                                   READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/iris-iris-classifier   2/2     2            2           2m9s

$ kubectl -n iris port-forward svc/iris-iris-classifier 8002:80 &
$ curl -s http://localhost:8002/health | jq
{ "status": "ok", "model_path": "/app/models/best_model.joblib", "version": "0.1.0", ... }

$ curl -s -X POST http://localhost:8002/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3],[5.9,3.0,4.2,1.5]]}' | jq
{
  "predictions": ["setosa","virginica","versicolor"],
  "model_version": "0.1.0",
  "feature_order": ["sepal_length","sepal_width","petal_length","petal_width"]
}
```

## Teardown commands

```bash
helm -n iris uninstall iris
kind delete cluster --name iris-cluster
docker rm -f iris-local 2>/dev/null
```
