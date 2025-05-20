# Makefile for the IRIS RAG Templates project

.PHONY: help start-iris stop-iris pull-iris install-py-deps load-data \
        lint test test-unit test-e2e-metrics test-globals test-all \
        test-1000 test-real-pmc-1000 test-all-1000-docs \
        docs-check clean

# Configuration
PYTHON_INTERPRETER := poetry run python
PIP_EXEC := poetry run pip
PYTEST_EXEC := poetry run pytest

# Default IRIS Docker settings (can be overridden by environment variables)
IRIS_IMAGE := intersystemsdc/iris-community:latest
IRIS_CONTAINER_NAME ?= iris_rag_demo
# Changed IRIS_PORT_JDBC to avoid conflict with existing 'iris' container
IRIS_PORT_JDBC ?= 51773
# Changed IRIS_PORT_WEB to avoid conflict with existing 'iris' container
IRIS_PORT_WEB ?= 52774
IRIS_USER ?= demo      # Reverted to original simpler value
IRIS_PASSWORD ?= demo   # Reverted to original simpler value
IRIS_NAMESPACE ?= DEMO  # Reverted to original simpler value

help:
	@echo "Available targets:"
	@echo "  start-iris          - Start the IRIS Docker container"
	@echo "  stop-iris           - Stop and remove the IRIS Docker container"
	@echo "  pull-iris           - Pull the specified IRIS Docker image"
	@echo "  install-py-deps     - Install Python dependencies using Poetry"
	@echo "  load-data           - Run the data loading script (eval/loader.py)"
	@echo "  lint                - Run linters (ruff, black, mypy)"
	@echo "  test                - Run all tests (alias for test-all)"
	@echo "  test-unit           - Run Python unit tests"
	@echo "  test-1000           - Run tests with 1000+ REAL PMC documents (as required by .clinerules)"
	@echo "  test-real-pmc-1000  - Run tests with 1000+ REAL PMC documents (alternative method)"
	@echo "  test-e2e-metrics    - Run end-to-end metrics tests (requires IRIS running)"
	@echo "  test-globals        - Run ObjectScript globals tests (requires IRIS running)"
	@echo "  test-all            - Run all linting, docs, and test suites"
	@echo "  docs-check          - Build and check documentation"
	@echo "  clean               - Clean up build artifacts and __pycache__"

# --- Docker Management ---
pull-iris:
	docker pull $(IRIS_IMAGE)

start-iris:
	@if [ $$(docker ps -q -f name=$(IRIS_CONTAINER_NAME)) ]; then \
		echo "IRIS container '$(IRIS_CONTAINER_NAME)' is already running."; \
	else \
		echo "Starting IRIS container '$(IRIS_CONTAINER_NAME)'..."; \
		echo "DEBUG: IRIS_IMAGE='$(IRIS_IMAGE)'"; \
		echo "DEBUG: Full docker command: docker run -d --name $(IRIS_CONTAINER_NAME) -p $(IRIS_PORT_JDBC):1972 -p $(IRIS_PORT_WEB):52773 -e IRISUSERNAME=$(IRIS_USER) -e IRISPASSWORD=$(IRIS_PASSWORD) -e IRISNAMESPACE=$(IRIS_NAMESPACE) $(IRIS_IMAGE)"; \
		docker run -d --name $(IRIS_CONTAINER_NAME) \
			-p $(IRIS_PORT_JDBC):1972 \
			-p $(IRIS_PORT_WEB):52773 \
			-e IRISUSERNAME=$(IRIS_USER) \
			-e IRISPASSWORD=$(IRIS_PASSWORD) \
			-e IRISNAMESPACE=$(IRIS_NAMESPACE) \
			$(IRIS_IMAGE); \
		echo "IRIS container started. Waiting a few seconds for it to initialize..."; \
		sleep 60; \
		echo "IRIS should be ready."; \
	fi

stop-iris:
	@echo "Attempting to stop and remove IRIS container '$(IRIS_CONTAINER_NAME)'..."
	@docker stop $(IRIS_CONTAINER_NAME) > /dev/null 2>&1 || true
	@docker rm -f $(IRIS_CONTAINER_NAME) > /dev/null 2>&1 || true
	@echo "IRIS container '$(IRIS_CONTAINER_NAME)' stopped and removed (if it existed)."

# --- Dependency Management ---
install-py-deps:
	poetry install

# --- Data Loading ---
load-data: install-py-deps
	@echo "Running data loader (eval/loader.py)..."
	# $(PYTHON_INTERPRETER) eval/loader.py # Uncomment when loader.py is implemented

# --- Linting ---
lint:
	@echo "Running linters..."
	poetry run ruff check .
	poetry run black . --check
	poetry run mypy .

# --- Testing ---
test-unit:
	@echo "Running Python unit tests..."
	$(PYTEST_EXEC) tests/ # Placeholder, adjust path if needed

test-1000:
	@echo "Running tests with 1000+ REAL PMC documents..."
	PYTEST_CONFTEST_PATH=tests/conftest_real_pmc.py poetry run pytest -xvs tests/test_minimal_real_pmc.py

test-real-pmc-1000:
	@echo "Running tests with 1000+ REAL PMC documents..."
	poetry run pytest -xvs tests/test_all_with_real_pmc_1000.py

test-all-1000-docs: test-1000 test-real-pmc-1000
	@echo "Running comprehensive tests with 1000+ documents..."
	./run_all_tests_with_1000_docs.sh

test-all-rag-1000:
	@echo "Running all RAG techniques with 1000+ documents..."
	python run_all_rag_with_1000_docs.py

test-all-rag-1000-with-log:
	@echo "Running all RAG techniques with 1000+ documents and logging results..."
	./run_all_rag_with_1000_docs.py | tee test_output_1000docs.log

test-pytest-1000:
	@echo "Running pytest for all RAG techniques with 1000+ documents..."
	./run_pytest_with_1000_docs.py -v

verify-1000-docs:
	@echo "Verifying all RAG techniques with 1000+ documents..."
	./verify_1000_docs_compliance.py

test-all-1000-docs-compliance:
	@echo "Running 1000+ document compliance tests for all RAG techniques..."
	@echo "=========================================================="
	@echo "This ensures compliance with .clinerules requirement of at least 1000 documents"
	@echo "=========================================================="
	@# First run with mock data for faster testing
	@echo "Step 1: Running with mock data for initial verification"
	@python -m pytest -v tests/test_all_with_1000_docs.py
	@# Then run with real PMC documents for proper compliance
	@echo "Step 2: Running with REAL PMC documents (as required by .clinerules)"
	@./run_real_pmc_1000_tests.py

# Target to ensure tests run with real PMC data in a real database
test-with-real-pmc-db: 
	@echo "Running tests with real PMC data in a real database as required by .clinerules"
	@echo "=========================================================="
	@bash -c "sleep 1 && ./run_with_real_pmc_data.sh"

test-e2e-metrics: start-iris # Ensure IRIS is running
	@echo "Running end-to-end metrics tests..."
	# $(PYTEST_EXEC) tests/test_*.py -m e2e_metrics # Example marker, adjust as needed
	@echo "Note: E2E metrics tests need IRIS running and data loaded."

test-globals: start-iris # Ensure IRIS is running
	@echo "Running ObjectScript globals tests..."
	# Placeholder: Add command to execute ObjectScript tests
	# e.g., $(PYTHON_INTERPRETER) scripts/run_objectscript_tests.py tests/test_globals.int
	@echo "Note: ObjectScript globals tests need IRIS running and data loaded."

test: test-all

test-all: lint docs-check test-unit # test-e2e-metrics test-globals
	@echo "All primary checks and unit tests passed."
	@echo "Note: E2E and Globals tests commented out by default in 'test-all'."
	@echo "Run 'make test-e2e-metrics' and 'make test-globals' separately after data loading."


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
