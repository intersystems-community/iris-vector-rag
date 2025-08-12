# RAG Templates Makefile

# Use the bash terminal
SHELL := /bin/bash

# Standardized commands for development, testing, and data management
# Uses Python virtual environment (.venv) for consistent dependency management

.PHONY: help install clean setup-db load-data clear-rag-data lint format test-dbapi test-jdbc validate-iris-rag status setup-env check-data

# Python virtual environment directory (managed by uv)
VENV_DIR = .venv

# Python execution command for consistent environment usage
# uv automatically manages the virtual environment and PYTHONPATH
PYTHON_RUN = PYTHONDONTWRITEBYTECODE=1 uv run python

# Default target
help:
	@echo "RAG Templates - Available Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make setup-env        - Set up Python virtual environment (.venv)"
	@echo "  make install          - Install dependencies in the virtual environment"
	@echo "  make setup-db         - Initialize IRIS database schema"
	@echo ""
	@echo "Data Management:"
	@echo "  make load-data        - Load sample PMC documents (DBAPI)"
	@echo "  make check-data       - Check current document count"
	@echo "  make clear-rag-data   - Clear all rows from RAG document tables (DocumentChunks and SourceDocuments)"
	@echo ""
	@echo "Development:"
	@echo "  make clean            - Clean up temporary files"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code"
	@echo ""
	@echo "Environment Info:"
	@echo "  Environment managed by uv (automatic virtual environment)"
	@echo "  All commands use 'uv run' prefix for consistent execution"

# Environment setup
setup-env:
	@echo "Setting up Python environment with uv..."
	@if ! command -v uv &> /dev/null; then \
		echo "Error: uv is not installed. Please install uv first:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@echo "✓ uv is installed"

# Installation and setup
install: setup-env
	@echo "Installing all dependencies with uv..."
	uv sync --frozen --all-extras --dev

setup-db:
	@echo "Setting up IRIS database schema (DBAPI)..."
	uv run python -c "from common.iris_connection_manager import test_connection; print('✓ Connection test passed' if test_connection() else '✗ Connection test failed')"
	uv run python -m common.db_init_with_indexes

validate-iris-rag:
	@echo "Validating iris_rag package..."
	uv run python -c "import iris_rag; print('✓ iris_rag package imported successfully')"

validate-all-pipelines:
	@echo "Validating all RAG pipelines can be imported and registered..."
	uv run python -c "from iris_rag.config.manager import ConfigurationManager; from iris_rag.core.connection import ConnectionManager; from iris_rag.pipelines.registry import PipelineRegistry; from iris_rag.pipelines.factory import PipelineFactory; from iris_rag.config.pipeline_config_service import PipelineConfigService; from iris_rag.utils.module_loader import ModuleLoader; config_manager = ConfigurationManager(); connection_manager = ConnectionManager(config_manager); framework_dependencies = {'connection_manager': connection_manager, 'config_manager': config_manager, 'llm_func': lambda x: 'test', 'vector_store': None}; config_service = PipelineConfigService(); module_loader = ModuleLoader(); pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies); pipeline_registry = PipelineRegistry(pipeline_factory); pipeline_registry.register_pipelines(); pipelines = pipeline_registry.list_pipeline_names(); print(f'✓ Successfully registered {len(pipelines)} pipelines:'); [print(f'  - {name}') for name in sorted(pipelines)]"

# Data management (DBAPI-first)
load-data:
	@echo "Loading sample PMC documents using DBAPI..."
	uv run python -c "from data.loader_fixed import process_and_load_documents; result = process_and_load_documents('data/sample_10_docs', limit=10); print(f'Loaded: {result}')"

check-data:
	@echo "Checking current document count using schema manager..."
	uv run python scripts/utilities/schema_managed_data_utils.py --check

clear-rag-data:
	@echo "Clearing RAG document tables using schema manager..."
	uv run python scripts/utilities/schema_managed_data_utils.py --clear

# Development tools
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf reports/temp/

lint:
	@echo "Running code linting..."
	uv run flake8 iris_rag/ --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	uv run black iris_rag/ --line-length=120

# Docker commands
docker-up:
	@echo "Starting IRIS container..."
	docker-compose up -d

docker-down:
	@echo "Stopping IRIS container..."
	docker-compose down

docker-logs:
	@echo "Viewing IRIS container logs..."
	docker-compose logs -f iris

# Connection testing
test-dbapi:
	@echo "Testing DBAPI connection..."
	uv run python -c "from common.iris_connection_manager import get_dbapi_connection; conn = get_dbapi_connection(); print('✓ DBAPI connection successful'); conn.close()"

test-jdbc:
	@echo "Testing JDBC connection (fallback)..."
	uv run python -c "from common.iris_connection_manager import IRISConnectionManager; mgr = IRISConnectionManager(prefer_dbapi=False); conn = mgr.get_connection(); print(f'✓ {mgr.get_connection_type()} connection successful'); mgr.close()"


# Status check with auto-healing
status:
	@echo "=== SYSTEM STATUS CHECK ==="
	@echo "Checking environment..."
	$(MAKE) env-info
	@echo ""
	@echo "Checking database connection..."
	$(MAKE) test-dbapi
	@echo ""
	@echo "Checking data availability..."
	$(MAKE) check-data
	@echo ""
	@echo "Checking pipeline validation status..."
	$(MAKE) validate-all-pipelines
	@echo ""
	@echo "=== STATUS CHECK COMPLETE ==="
