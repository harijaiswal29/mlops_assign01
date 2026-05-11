#!/usr/bin/env bash
# Spin up a local kind cluster, load the Docker image, and deploy via Helm.
# Requires: docker, kind, kubectl, helm (install hints printed if missing).
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-iris-cluster}"
IMAGE="${IMAGE:-iris-classifier:latest}"
RELEASE="${RELEASE:-iris}"
NAMESPACE="${NAMESPACE:-iris}"

cd "$(dirname "$0")/.."

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is not installed." >&2
        case "$1" in
            kind)
                echo "  Install: https://kind.sigs.k8s.io/docs/user/quick-start/#installation" >&2
                ;;
            kubectl)
                echo "  Install: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/" >&2
                ;;
            helm)
                echo "  Install: https://helm.sh/docs/intro/install/" >&2
                ;;
            docker)
                echo "  Enable Docker Desktop WSL integration." >&2
                ;;
        esac
        exit 1
    fi
}

for tool in docker kind kubectl helm; do require "$tool"; done

if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
    echo "==> Creating kind cluster '$CLUSTER_NAME'"
    kind create cluster --name "$CLUSTER_NAME"
else
    echo "==> Cluster '$CLUSTER_NAME' already exists"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "==> Image $IMAGE not found locally — building"
    docker build -t "$IMAGE" .
fi

echo "==> Loading image into kind"
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"

echo "==> Creating namespace $NAMESPACE (if missing)"
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 \
    || kubectl create namespace "$NAMESPACE"

echo "==> helm upgrade --install $RELEASE"
helm upgrade --install "$RELEASE" ./helm/iris-classifier \
    --namespace "$NAMESPACE" \
    --set image.repository=iris-classifier \
    --set image.tag=latest

echo "==> Waiting for pods to become ready"
kubectl -n "$NAMESPACE" rollout status deploy/"$RELEASE-iris-classifier" --timeout=120s

kubectl -n "$NAMESPACE" get pods,svc

cat <<EOM

Next steps:
  kubectl -n $NAMESPACE port-forward svc/$RELEASE-iris-classifier 8000:80
  curl -X POST http://localhost:8000/predict \\
    -H 'Content-Type: application/json' \\
    -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'

Teardown:
  helm -n $NAMESPACE uninstall $RELEASE
  kind delete cluster --name $CLUSTER_NAME
EOM
