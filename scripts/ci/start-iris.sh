#!/usr/bin/env bash
# Start (or reuse) the IRIS Community container the test steps connect to.
#
# Why this exists:
#  - intersystemsdc/iris-community:2026.1's post-start init (irissqlcli via embedded
#    Python) crashes on amd64 and takes IRIS down with it, so we skip it with
#    IRIS_INIT=1 and do the one thing we need (unexpire passwords) ourselves.
#  - Running with IRIS_INIT=1 leaves _SYSTEM in "Password change required" state;
#    UnExpireUserPasswords("*") clears it so DBAPI logins work.
#
# The container and port default to the declaration in CLAUDE.md:
#
#   IRIS_CONTAINER: iris-vector-rag-iris
#   IRIS_PORT: 51972
#
# They used to default to a container named `iris` on port 1972, which is another
# project's container on a third project's port -- so the script could not be run
# on a developer machine at all, and "reproduce the CI step locally" was advice
# nobody could follow. CI passes --container/--port for its throwaway container.
#
# An existing container is reused (running) or started (stopped). Only --recreate
# deletes one, because deleting the declared container deletes this project's
# database.
#
# Usage:
#   bash scripts/ci/start-iris.sh [--container NAME] [--port PORT] [--image IMAGE]
#                                 [--recreate] [--dry-run]
set -uo pipefail

REPO_ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
CLAUDE_MD="$REPO_ROOT/CLAUDE.md"

declaration() { # <key>
    [[ -f "$CLAUDE_MD" ]] || return 0
    grep -E "^$1:" "$CLAUDE_MD" 2>/dev/null \
        | head -1 \
        | sed "s/$1:[[:space:]]*//" \
        | tr -d '[:space:]'
}

NAME="${IRIS_CONTAINER_NAME:-$(declaration IRIS_CONTAINER)}"
PORT="${IRIS_HOST_PORT:-$(declaration IRIS_PORT)}"
IMAGE="${IRIS_IMAGE:-intersystemsdc/iris-community:2026.1}"
RECREATE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "start-iris.sh: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$NAME" || "$NAME" == "none" ]]; then
    echo "start-iris.sh: no container to start." >&2
    echo "  Declare one in CLAUDE.md as 'IRIS_CONTAINER: <name>' (and" >&2
    echo "  'IRIS_PORT: <port>'), or pass --container/--port." >&2
    echo "  There is no safe default: the old one deleted another project's" >&2
    echo "  container and bound a third project's port." >&2
    exit 1
fi
: "${PORT:=1972}"

exists() { docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qx "$NAME"; }
running() { docker ps --format "{{.Names}}" 2>/dev/null | grep -qx "$NAME"; }

if $RECREATE; then
    ACTION="recreate"
elif running; then
    ACTION="reuse"
elif exists; then
    ACTION="start"
else
    ACTION="create"
fi

if $DRY_RUN; then
    echo "container=$NAME"
    echo "port=$PORT"
    echo "image=$IMAGE"
    echo "action=$ACTION"
    exit 0
fi

case "$ACTION" in
    reuse)
        echo "IRIS already running: container=${NAME} port=${PORT} (not recreated)"
        echo "Pass --recreate for a pristine database -- it deletes this one."
        exit 0
        ;;
    start)
        echo "Starting existing container ${NAME}..."
        docker start "$NAME" >/dev/null
        ;;
    recreate)
        echo "Recreating ${NAME} from ${IMAGE} — its current database is deleted."
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        docker run --detach --name "$NAME" -p "${PORT}:1972" \
            -e IRIS_INIT=1 "$IMAGE" >/dev/null
        ;;
    create)
        docker run --detach --name "$NAME" -p "${PORT}:1972" \
            -e IRIS_INIT=1 "$IMAGE" >/dev/null
        ;;
esac

echo "Waiting for IRIS on port ${PORT}..."
timeout 120 bash -c "until nc -z localhost ${PORT}; do sleep 2; done"
sleep 5

# The port opens before IRIS accepts terminal sessions; retry until the
# ObjectScript call actually runs (up to ~2 minutes).
unexpire() {
    printf 'do ##class(Security.Users).UnExpireUserPasswords("*")\nwrite "PASSWORDS_UNEXPIRED",!\nhalt\n' \
        | docker exec -i "$NAME" iris session IRIS -U%SYS 2>&1 | grep -q PASSWORDS_UNEXPIRED
}
for attempt in $(seq 1 24); do
    if unexpire; then
        echo "IRIS ready: container=${NAME} port=${PORT} (attempt ${attempt})"
        exit 0
    fi
    sleep 5
done

echo "IRIS did not become ready; container logs:" >&2
docker logs "$NAME" 2>&1 | tail -40 >&2
exit 1
