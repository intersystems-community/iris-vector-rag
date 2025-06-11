#!/bin/bash

# RAG Templates - Environment Activation Script
# This script ensures we're using the correct conda environment

set -e

ENV_NAME="rag-templates"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Environment '${ENV_NAME}' not found."
    echo "🏗️  Run './setup_environment.sh' to create it."
    exit 1
fi

# Activate environment
echo "🔄 Activating conda environment: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "${ENV_NAME}" ]]; then
    echo "✅ Environment '${ENV_NAME}' activated successfully"
    echo "🐍 Python: $(which python)"
    echo "📦 Conda env: $CONDA_DEFAULT_ENV"
else
    echo "❌ Failed to activate environment"
    exit 1
fi

# Test key imports
echo "🧪 Testing key imports..."
python -c "
try:
    from common.utils import get_embedding_func
    from common.iris_connection_manager import get_iris_connection
    import torch
    import transformers
    import sentence_transformers
    print('✅ All key imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo "🎯 Environment ready for RAG operations!"