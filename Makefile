# Makefile for the IRIS RAG Templates project

.PHONY: help start-iris stop-iris pull-iris \
        export-py-deps install-py-deps setup-dev-env \
        init-db load-data setup-db \
        lint test test-unit test-real-pmc-1000 \
        docs-check clean

# Configuration
# Ensure .venv is activated before running make targets that use these
PYTHON_INTERPRETER := python
PYTEST_EXEC := pytest
UV_EXEC := uv
POETRY_EXEC := poetry

# IRIS Docker settings
IRIS_COMPOSE_FILE := docker-compose.iris-only.yml
IRIS_SERVICE_NAME := iris_db # Service name in docker-compose.iris-only.yml
# IRIS connection details should be managed by environment variables (see common/iris_connector.py)
# or directly by docker-compose.iris-only.yml

help:
	@echo "Available targets:"
	@echo "  start-iris          - Start the IRIS Docker container using docker-compose"
	@echo "  stop-iris           - Stop and remove the IRIS Docker container using docker-compose"
	@echo "  pull-iris           - Pull the IRIS Docker image specified in $(IRIS_COMPOSE_FILE)"
	@echo "  export-py-deps      - Export Python dependencies from pyproject.toml to requirements.txt using Poetry"
	@echo "  install-py-deps     - Install Python dependencies from requirements.txt using uv"
	@echo "  setup-dev-env       - Export and install Python dependencies"
	@echo "  init-db             - Initialize IRIS database schema (requires IRIS running)"
	@echo "  load-data           - Load PMC data into IRIS (requires IRIS running and schema initialized)"
	@echo "  setup-db            - Initialize schema and load data (requires IRIS running)"
	@echo "  lint                - Run linters (ruff, black, mypy)"
	@echo "  test                - Run all unit tests (alias for test-unit)"
	@echo "  test-unit           - Run Python unit tests (e.g., tests/test_basic_rag.py)"
	@echo "  test-real-pmc-1000  - Run E2E tests with 1000+ REAL PMC documents (requires IRIS running, schema, and data)"
	@echo "  docs-check          - Build and check documentation (placeholder)"
	@echo "  clean               - Clean up build artifacts and __pycache__"

# --- Docker Management ---
pull-iris:
	docker-compose -f $(IRIS_COMPOSE_FILE) pull $(IRIS_SERVICE_NAME)

start-iris:
	@echo "Starting IRIS container using $(IRIS_COMPOSE_FILE)..."
	docker-compose -f $(IRIS_COMPOSE_FILE) up -d --wait $(IRIS_SERVICE_NAME)
	@echo "IRIS container started. Waiting a bit longer for full initialization..."
	@sleep 15 # Adjust as needed, --wait might not be enough for full app readiness
	@echo "IRIS should be ready. Management Portal: http://localhost:52773 (default)"

stop-iris:
	@echo "Stopping and removing IRIS container defined in $(IRIS_COMPOSE_FILE)..."
	docker-compose -f $(IRIS_COMPOSE_FILE) down -v --remove-orphans
	@echo "IRIS container stopped and removed."

# --- Dependency Management (Host Python Environment with uv) ---
# Assumes .venv is created and activated:
# uv venv .venv --python python3.11
# source .venv/bin/activate
export-py-deps:
	@echo "Exporting dependencies from pyproject.toml to requirements.txt using Poetry..."
	$(POETRY_EXEC) export -f requirements.txt --output requirements.txt --without-hashes --with dev

install-py-deps:
	@echo "Installing Python dependencies from requirements.txt using uv..."
	$(UV_EXEC) pip install -r requirements.txt

setup-dev-env: export-py-deps install-py-deps
	@echo "Python development environment setup complete (dependencies exported and installed)."

# --- Database Setup (Host Python scripts against IRIS Docker) ---
init-db:
	@echo "Initializing IRIS database schema..."
	$(PYTHON_INTERPRETER) run_db_init_local.py --force-recreate

load-data:
	@echo "Loading PMC data into IRIS (limit 1100), including ColBERT token embeddings..."
	$(PYTHON_INTERPRETER) scripts_to_review/load_pmc_data.py --limit 1100 --load-colbert

setup-db: init-db load-data
	@echo "Database schema initialized and data loaded."

# --- Linting (Host Python Environment) ---
lint:
	@echo "Running linters (ruff, black, mypy)..."
	$(UV_EXEC) run ruff check .
	$(UV_EXEC) run black . --check
	$(UV_EXEC) run mypy .

# --- Testing (Host Python Environment against IRIS Docker) ---
# Ensure .venv is activated (source .venv/bin/activate)
# Ensure IRIS is running (make start-iris) and DB is set up (make setup-db) for E2E tests

test-unit:
	@echo "Running Python unit tests..."
	$(PYTEST_EXEC) tests/ # Runs all tests in the tests/ directory

# This is the primary target for E2E tests with 1000+ real PMC documents
test-real-pmc-1000:
	@echo "Running E2E tests with 1000+ REAL PMC documents (tests/test_all_with_real_pmc_1000.py)..."
	@echo "Ensure IRIS is running, schema initialized, and data loaded (make start-iris; make setup-db)."
	$(PYTEST_EXEC) -xvs tests/test_all_with_real_pmc_1000.py

# Simplified 'test' alias
test: test-unit

# test-all is now focused on linting and unit tests for quick checks.
# E2E tests are run separately via test-real-pmc-1000.
test-all: lint test-unit
	@echo "Linting and unit tests passed."
	@echo "For E2E tests with real data, run: make test-real-pmc-1000"


# --- Documentation ---
docs-check:
	@echo "Checking documentation..."
	# poetry run mkdocs build --strict # Uncomment when MkDocs is set up
	# poetry run codespell docs/       # Uncomment when docs exist
	# poetry run lychee docs/          # Uncomment when docs exist and lychee configured
	@echo "Documentation checks are placeholders."

# --- Cleaning ---
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf site/ # If using mkdocs
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
