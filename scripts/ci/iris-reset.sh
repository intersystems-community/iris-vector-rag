#!/usr/bin/env bash
# Recover a wedged IRIS container: stop IRIS cleanly, restart the container, wait until it
# answers, and make the warm-up connection the test fixtures make.
#
# Why: the dev container is hand-prepared for ivg 4.1.1 and wedges when a Native API process
# is killed (docs/DEBT.md entry 10); a restart clears it. The restart, health check and
# connection test are iris-devtester's (`idt container ...`); this script adds the graceful
# stop and the warm-up connection.
#
# The container and port come from the declaration in CLAUDE.md, as in start-iris.sh:
#
#   IRIS_CONTAINER: iris-vector-rag-iris
#   IRIS_PORT: 51972
#
# IRIS_CONTAINER_NAME in the environment is deliberately ignored: a reset must not be
# re-aimed at another project's container by a stray shell variable. Only CI's throwaway
# container may be named, with --container, and only when CI is set.
#
# Usage: bash scripts/ci/iris-reset.sh [--container NAME --port PORT] [--dry-run]
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

DECLARED="$(declaration IRIS_CONTAINER)"
NAME="$DECLARED"
PORT="$(declaration IRIS_PORT)"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "iris-reset.sh: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DECLARED" ]]; then
    echo "iris-reset.sh: no container to reset." >&2
    echo "  Declare one in CLAUDE.md as 'IRIS_CONTAINER: <name>' (and 'IRIS_PORT: <port>')." >&2
    exit 1
fi

if [[ "$NAME" != "$DECLARED" && -z "${CI:-}" ]]; then
    echo "iris-reset.sh: refusing to reset '$NAME': this project's container is '$DECLARED'." >&2
    echo "  Only CI's throwaway container may be named, and only when CI is set." >&2
    exit 1
fi

TIMEOUT=180
STEPS=(
    "graceful-stop docker exec $NAME iris stop IRIS quietly"
    "restart idt container restart $NAME --timeout $TIMEOUT"
    "wait-healthy idt container test-connection $NAME"
    "warm-up IRIS_CONTAINER_NAME=$NAME python -m tests.support.iris_environment warm-up"
)

if $DRY_RUN; then
    echo "container=$NAME"
    echo "port=${PORT:-}"
    for i in "${!STEPS[@]}"; do echo "step$((i + 1))=${STEPS[$i]}"; done
    exit 0
fi

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "iris-reset.sh: container '$NAME' does not exist. Create it with scripts/ci/start-iris.sh." >&2
    exit 1
fi

echo "[1/4] stopping IRIS cleanly in $NAME (a wedged instance may refuse; that is fine)"
docker exec "$NAME" iris stop IRIS quietly >/dev/null 2>&1 || true

echo "[2/4] restarting $NAME"
idt container restart "$NAME" --timeout "$TIMEOUT" || exit 1

echo "[3/4] waiting for $NAME to answer"
for attempt in $(seq 1 24); do
    if idt container test-connection "$NAME" >/dev/null 2>&1; then break; fi
    if [[ "$attempt" -eq 24 ]]; then
        echo "iris-reset.sh: $NAME never answered; docker logs $NAME" >&2
        exit 1
    fi
    sleep 5
done

echo "[4/4] warm-up connection"
cd "$REPO_ROOT" && IRIS_CONTAINER_NAME="$NAME" python -m tests.support.iris_environment warm-up || exit 1
echo "IRIS reset: container=$NAME port=${PORT:-?}"
