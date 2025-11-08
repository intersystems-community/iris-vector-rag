#!/bin/bash

# =============================================================================
# DEPRECATED: RAG Templates - Conda Environment Setup
# =============================================================================
# ⚠️  DEPRECATION NOTICE: This conda-based setup is deprecated as of v1.6.0
#
# The project has migrated to uv for package management per constitutional requirement.
#
# For new setup, please use:
#   make setup-env    # Create .venv using uv
#   make install      # Install dependencies via uv sync
#
# This script is kept for compatibility but will be removed in a future version.
# =============================================================================

set -e  # Exit on any error

echo "=== ⚠️  DEPRECATED: RAG TEMPLATES CONDA ENVIRONMENT SETUP ==="
echo "This setup method is deprecated. Please use 'make setup-env && make install' instead."
echo "Continuing with conda setup for compatibility..."
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✓ Conda found: $(conda --version)"

# Environment name
ENV_NAME="rag-templates"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "📦 Using existing environment..."
        conda activate ${ENV_NAME}
        echo "✓ Environment activated"
        exit 0
    fi
fi

echo "🏗️  Creating new conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.11 -y

echo "🔄 Activating environment..."
conda activate ${ENV_NAME}

echo "📦 Installing core dependencies..."
# Install core packages first
conda install -n ${ENV_NAME} -c conda-forge -y \
    numpy=1.26.4 \
    pandas=2.0.3 \
    pyyaml=6.0 \
    requests=2.31.0 \
    tqdm=4.66.0 \
    psutil=5.9.0

echo "🤖 Installing ML/AI packages..."
# Install ML packages
conda install -n ${ENV_NAME} -c conda-forge -y \
    pytorch=2.1.0 \
    torchvision=0.16.0 \
    torchaudio=2.1.0 \
    transformers=4.35.0 \
    sentence-transformers=2.2.2 \
    huggingface_hub=0.17.0

echo "🧪 Installing testing packages..."
# Install testing packages
conda install -n ${ENV_NAME} -c conda-forge -y \
    pytest=7.4.0 \
    pytest-cov=4.1.0 \
    pytest-mock=3.11.0

echo "📊 Installing visualization packages..."
# Install visualization packages
conda install -n ${ENV_NAME} -c conda-forge -y \
    matplotlib=3.8.0 \
    seaborn=0.13.0 \
    plotly=5.17.0

echo "🔗 Installing database and connection packages..."
# Install via pip for packages not available in conda-forge
pip install \
    intersystems-irispython==5.1.2 \
    sqlalchemy>=2.0.0 \
    testcontainers-iris>=1.2.0 \
    testcontainers>=3.7.0 \
    jaydebeapi>=1.2.3 \
    pyodbc>=5.2.0

echo "🦜 Installing LangChain packages..."
pip install \
    langchain>=0.1.0 \
    langchain-openai>=0.1.0,<0.2.0

echo "📈 Installing evaluation packages..."
pip install \
    ragas>=0.1.0 \
    evidently>=0.4.0

echo "🕸️ Installing graph packages..."
pip install \
    networkx>=3.1

echo "🛠️ Installing development tools..."
pip install \
    python-dotenv>=1.0.0 \
    ruff>=0.1.5 \
    black>=23.11.0 \
    mypy>=1.7.0

echo "🔧 Installing additional utilities..."
pip install \
    datasets>=2.14.0 \
    accelerate>=1.7.0 \
    pytest-lazy-fixture>=0.6.3

echo "✅ Environment setup complete!"
echo
echo "🎯 To activate this environment in the future, run:"
echo "   conda activate ${ENV_NAME}"
echo
echo "🧪 To test the environment, run:"
echo "   conda activate ${ENV_NAME}"
echo "   python -c \"from common.utils import get_embedding_func; print('✓ Environment working!')\""
echo
echo "📋 Environment summary:"
conda list -n ${ENV_NAME} | grep -E "(torch|transformers|sentence|iris|langchain|pytest)" | head -10

echo
echo "=== SETUP COMPLETE ==="