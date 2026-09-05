#!/usr/bin/env bash
# Start (or restart) a pristine IRIS Community container for CI test steps.
#
# Why this exists:
#  - intersystemsdc/iris-community:2026.1's post-start init (irissqlcli via embedded
#    Python) crashes on amd64 and takes IRIS down with it, so we skip it with
#    IRIS_INIT=1 and do the one thing we need (unexpire passwords) ourselves.
#  - Running with IRIS_INIT=1 leaves _SYSTEM in "Password change required" state;
#    UnExpireUserPasswords("*") clears it so DBAPI logins work.
#  - Re-running the script gives a clean database between test phases.
set -euo pipefail

NAME="${IRIS_CONTAINER_NAME:-iris}"
PORT="${IRIS_HOST_PORT:-1972}"
IMAGE="${IRIS_IMAGE:-intersystemsdc/iris-community:2026.1}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --detach --name "$NAME" -p "${PORT}:1972" -e IRIS_INIT=1 "$IMAGE" >/dev/null

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
