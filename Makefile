.DEFAULT_GOAL := help
.PHONY: help install iris-up iris-down iris-logs test-unit lint format-check doctor

help: ## Show available targets
	@echo ""
	@echo "IRIS Vector RAG - Available Targets"
	@echo "===================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

install: ## Install package and dev dependencies
	pip install -e ".[dev,dspy]"

iris-up: ## Start IRIS container via docker compose
	docker compose up -d

iris-down: ## Stop IRIS container
	docker compose down

iris-logs: ## View IRIS container logs
	docker compose logs -f iris

test-unit: ## Run unit tests (no database required)
	python -m pytest tests/unit/ --timeout=30 -q

lint: ## Run ruff linting checks
	python -m ruff check iris_vector_rag/ --select=E9,F63,F7,F82

format-check: ## Check code formatting with black
	python -m black --check iris_vector_rag/ tests/

doctor: ## Verify environment and imports
	python -c "from iris_vector_rag import create_pipeline; print('✓ Import OK')"
