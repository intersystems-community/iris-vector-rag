#!/bin/bash
set -e

echo "Starting RAG API server..."
cd /app

# Set default environment variables
export API_WORKERS=${API_WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-info}
export API_RELOAD=${API_RELOAD:-false}

# Start the FastAPI server
if [ "${API_RELOAD}" = "true" ]; then
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${API_WORKERS} \
        --log-level ${LOG_LEVEL} \
        --access-log \
        --reload
else
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers ${API_WORKERS} \
        --log-level ${LOG_LEVEL} \
        --access-log
fi