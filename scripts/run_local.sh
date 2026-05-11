#!/usr/bin/env bash
# Local end-to-end smoke: build image -> run container -> health check -> sample predict.
set -euo pipefail

IMAGE="${IMAGE:-iris-classifier:latest}"
PORT="${PORT:-8000}"
CONTAINER="${CONTAINER:-iris-local}"

cd "$(dirname "$0")/.."

echo "==> Building image $IMAGE"
docker build -t "$IMAGE" .

echo "==> Removing any previous container named $CONTAINER"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "==> Starting container"
docker run -d --name "$CONTAINER" -p "$PORT:8000" "$IMAGE"

echo "==> Waiting for /health"
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then
        echo "Healthy after ${i}s"
        break
    fi
    sleep 1
done

echo "==> /health response:"
curl -s "http://localhost:$PORT/health" | python -m json.tool

echo "==> Sample prediction (setosa):"
curl -s -X POST "http://localhost:$PORT/predict" \
    -H 'Content-Type: application/json' \
    -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}' | python -m json.tool

echo
echo "Container '$CONTAINER' is running. Stop with: docker rm -f $CONTAINER"
