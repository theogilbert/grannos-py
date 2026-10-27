#!/usr/bin/env bash
# Start a MongoDB container, run integration tests, then tear down.
# Requires: pip install pymongo
set -euo pipefail

CONTAINER="mongodb-dev"
PORT=27017
IMAGE="docker.io/library/mongo:7.0"
DATABASE="grannos_test"
TIMEOUT=60

cleanup() {
    echo "Stopping container..."
    docker rm -f "$CONTAINER" &>/dev/null || true
}
trap cleanup EXIT

echo "Starting MongoDB container..."
docker run \
    -p "$PORT:27017" \
    --name "$CONTAINER" \
    --hostname "$CONTAINER" \
    --replace \
    -d "$IMAGE"

echo "Waiting for MongoDB to accept connections (timeout: ${TIMEOUT}s)..."
deadline=$(( $(date +%s) + TIMEOUT ))
until docker exec "$CONTAINER" \
        mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "MongoDB did not become ready within ${TIMEOUT}s." >&2
        exit 1
    fi
    sleep 2
done
echo "MongoDB is ready."

MONGODB_URI="mongodb://localhost:$PORT" \
MONGODB_DATABASE="$DATABASE" \
    python -m pytest tests/external/test_mongodb.py -v "$@"
