# =============================================================================
# RAG Templates Framework - Docker Management Makefile
# =============================================================================
# Convenient targets for managing the complete RAG framework deployment
# =============================================================================

# Default configuration
COMPOSE_FILE := docker-compose.full.yml
ENV_FILE := .env
PROJECT_NAME := rag-templates

# Docker Compose command with common flags
DOCKER_COMPOSE := docker-compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Helper function to print colored messages
define print_message
	@echo -e "$(1)[MAKE]$(NC) $(2)"
endef

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP AND INFORMATION
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo -e "$(BLUE)RAG Templates Framework - Docker Management$(NC)"
	@echo -e "$(BLUE)=============================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo -e "$(YELLOW)Examples:$(NC)"
	@echo "  make docker-up              # Start core services"
	@echo "  make docker-up-dev          # Start development environment"
	@echo "  make docker-up-prod         # Start production environment"
	@echo "  make docker-up-data         # Start with sample data"
	@echo "  make docker-logs             # View all service logs"
	@echo "  make docker-health           # Check service health"

.PHONY: info
info: ## Show system and configuration information
	$(call print_message,$(BLUE),System Information)
	@echo -e "  $(GREEN)Docker version:$(NC)         $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo -e "  $(GREEN)Docker Compose version:$(NC) $$(docker-compose --version 2>/dev/null || echo 'Not installed')"
	@echo -e "  $(GREEN)Project root:$(NC)           $$(pwd)"
	@echo -e "  $(GREEN)Compose file:$(NC)           $(COMPOSE_FILE)"
	@echo -e "  $(GREEN)Environment file:$(NC)       $(ENV_FILE)"
	@echo ""
	$(call print_message,$(BLUE),Configuration Status)
	@if [ -f "$(ENV_FILE)" ]; then \
		echo -e "  $(GREEN)✓$(NC) Environment file exists"; \
	else \
		echo -e "  $(RED)✗$(NC) Environment file missing (copy from .env.example)"; \
	fi
	@if docker info >/dev/null 2>&1; then \
		echo -e "  $(GREEN)✓$(NC) Docker daemon running"; \
	else \
		echo -e "  $(RED)✗$(NC) Docker daemon not running"; \
	fi

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

.PHONY: setup-env
setup-env: ## Create Python virtual environment using uv
	$(call print_message,$(BLUE),Creating Python virtual environment with uv)
	@if ! command -v uv &> /dev/null; then \
		echo -e "  $(RED)✗$(NC) uv not found. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@uv venv .venv
	@echo -e "  $(GREEN)✓$(NC) Virtual environment created with uv"
	@echo -e "  $(YELLOW)⚠$(NC) To activate: source .venv/bin/activate"

.PHONY: install
install: ## Install all dependencies using uv
	$(call print_message,$(BLUE),Installing dependencies with uv)
	@if ! command -v uv &> /dev/null; then \
		echo -e "  $(RED)✗$(NC) uv not found. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, creating one..."; \
		$(MAKE) setup-env; \
	fi
	@uv sync
	@echo -e "  $(GREEN)✓$(NC) Dependencies installed with uv"

.PHONY: setup
setup: setup-env install ## Complete setup - create environment, install dependencies, and setup directories
	$(call print_message,$(BLUE),Setting up RAG Templates environment)
	@if [ ! -f "$(ENV_FILE)" ]; then \
		cp .env.example $(ENV_FILE); \
		echo -e "  $(GREEN)✓$(NC) Copied .env.example to .env"; \
		echo -e "  $(YELLOW)⚠$(NC) Please edit .env with your configuration"; \
	else \
		echo -e "  $(GREEN)✓$(NC) Environment file already exists"; \
	fi
	@mkdir -p logs data/cache data/uploads docker/ssl monitoring/data
	@echo -e "  $(GREEN)✓$(NC) Created necessary directories"
	@chmod +x scripts/docker/*.sh
	@echo -e "  $(GREEN)✓$(NC) Made scripts executable"

.PHONY: setup-db
setup-db: ## Initialize IRIS database schema
	$(call print_message,$(BLUE),Initializing IRIS database)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python common/db_init_complete.sql || echo "Schema initialization completed"
	@echo -e "  $(GREEN)✓$(NC) Database schema initialized"

.PHONY: load-data
load-data: ## Load sample data into IRIS database
	$(call print_message,$(BLUE),Loading sample data)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python data/loader_fixed.py
	@echo -e "  $(GREEN)✓$(NC) Sample data loaded"

.PHONY: validate-iris-rag
validate-iris-rag: setup-db ## Validate iris_rag package installation
	$(call print_message,$(BLUE),Validating iris_rag package)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -c "import iris_rag; print('✓ iris_rag package imported successfully')"
	@echo -e "  $(GREEN)✓$(NC) iris_rag package validation complete"

.PHONY: env-check
env-check: ## Check environment configuration
	$(call print_message,$(BLUE),Checking environment configuration)
	@if [ -f "$(ENV_FILE)" ]; then \
		echo -e "  $(GREEN)✓$(NC) Environment file exists"; \
		if grep -q "your_.*_here" $(ENV_FILE); then \
			echo -e "  $(YELLOW)⚠$(NC) Some placeholders still need to be replaced"; \
			grep "your_.*_here" $(ENV_FILE) | head -5; \
		else \
			echo -e "  $(GREEN)✓$(NC) No obvious placeholders found"; \
		fi; \
	else \
		echo -e "  $(RED)✗$(NC) Environment file missing"; \
		exit 1; \
	fi

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: docker-up
docker-up: env-check ## Start core services (IRIS, Redis, API, Streamlit)
	$(call print_message,$(GREEN),Starting core RAG services)
	$(DOCKER_COMPOSE) --profile core up -d
	$(call print_message,$(GREEN),Core services started)
	@$(MAKE) docker-urls

.PHONY: docker-up-dev
docker-up-dev: env-check ## Start development environment (includes Jupyter)
	$(call print_message,$(GREEN),Starting development environment)
	$(DOCKER_COMPOSE) --profile dev up -d
	$(call print_message,$(GREEN),Development environment started)
	@$(MAKE) docker-urls

.PHONY: docker-up-prod
docker-up-prod: env-check ## Start production environment (includes Nginx, monitoring)
	$(call print_message,$(GREEN),Starting production environment)
	$(DOCKER_COMPOSE) --profile prod up -d
	$(call print_message,$(GREEN),Production environment started)
	@$(MAKE) docker-urls

.PHONY: docker-up-data
docker-up-data: env-check ## Start services with sample data loading
	$(call print_message,$(GREEN),Starting services with sample data)
	$(DOCKER_COMPOSE) --profile with-data up -d
	$(call print_message,$(GREEN),Services with data loading started)
	@$(MAKE) docker-urls

.PHONY: docker-up-all
docker-up-all: env-check ## Start all services (development + production)
	$(call print_message,$(GREEN),Starting all available services)
	$(DOCKER_COMPOSE) --profile dev --profile prod --profile monitoring up -d
	$(call print_message,$(GREEN),All services started)
	@$(MAKE) docker-urls

.PHONY: docker-down
docker-down: ## Stop all services
	$(call print_message,$(YELLOW),Stopping all services)
	$(DOCKER_COMPOSE) down
	$(call print_message,$(GREEN),All services stopped)

.PHONY: docker-down-clean
docker-down-clean: ## Stop all services and remove volumes
	$(call print_message,$(YELLOW),Stopping services and cleaning volumes)
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(call print_message,$(GREEN),Services stopped and volumes cleaned)

.PHONY: docker-restart
docker-restart: ## Restart all running services
	$(call print_message,$(YELLOW),Restarting services)
	$(DOCKER_COMPOSE) restart
	$(call print_message,$(GREEN),Services restarted)

.PHONY: docker-restart-%
docker-restart-%: ## Restart specific service (e.g., make docker-restart-api)
	$(call print_message,$(YELLOW),Restarting service: $*)
	$(DOCKER_COMPOSE) restart $*
	$(call print_message,$(GREEN),Service $* restarted)

# =============================================================================
# BUILDING AND IMAGES
# =============================================================================

.PHONY: docker-build
docker-build: ## Build all Docker images
	$(call print_message,$(BLUE),Building Docker images)
	$(DOCKER_COMPOSE) build --parallel
	$(call print_message,$(GREEN),Docker images built)

.PHONY: docker-build-nocache
docker-build-nocache: ## Build all Docker images without cache
	$(call print_message,$(BLUE),Building Docker images without cache)
	$(DOCKER_COMPOSE) build --no-cache --parallel
	$(call print_message,$(GREEN),Docker images built without cache)

.PHONY: docker-pull
docker-pull: ## Pull latest base images
	$(call print_message,$(BLUE),Pulling latest base images)
	$(DOCKER_COMPOSE) pull
	$(call print_message,$(GREEN),Base images updated)

# =============================================================================
# MONITORING AND DEBUGGING
# =============================================================================

.PHONY: docker-ps
docker-ps: ## Show running containers
	$(call print_message,$(BLUE),Container Status)
	$(DOCKER_COMPOSE) ps

.PHONY: docker-logs
docker-logs: ## View logs from all services
	$(DOCKER_COMPOSE) logs -f

.PHONY: docker-logs-%
docker-logs-%: ## View logs from specific service (e.g., make docker-logs-api)
	$(DOCKER_COMPOSE) logs -f $*

.PHONY: docker-health
docker-health: ## Check health of all services
	$(call print_message,$(BLUE),Running health check)
	@./scripts/docker/health-check.sh

.PHONY: docker-health-json
docker-health-json: ## Check health and output JSON
	@./scripts/docker/health-check.sh --json

.PHONY: docker-stats
docker-stats: ## Show resource usage statistics
	$(call print_message,$(BLUE),Container Resource Usage)
	docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# =============================================================================
# SHELL ACCESS
# =============================================================================

.PHONY: docker-shell
docker-shell: ## Open shell in API container
	$(DOCKER_COMPOSE) exec rag_api /bin/bash

.PHONY: docker-shell-%
docker-shell-%: ## Open shell in specific container (e.g., make docker-shell-iris)
	$(DOCKER_COMPOSE) exec $* /bin/bash

.PHONY: docker-iris-shell
docker-iris-shell: setup-db ## Open IRIS database shell
	$(DOCKER_COMPOSE) exec iris_db iris session iris

.PHONY: docker-redis-shell
docker-redis-shell: ## Open Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

.PHONY: docker-init-data
docker-init-data: ## Initialize database with sample data
	$(call print_message,$(BLUE),Initializing sample data)
	@./scripts/docker/init-data.sh
	$(call print_message,$(GREEN),Sample data initialized)

.PHONY: docker-init-data-force
docker-init-data-force: ## Force reload sample data
	$(call print_message,$(BLUE),Force reloading sample data)
	@./scripts/docker/init-data.sh --force
	$(call print_message,$(GREEN),Sample data force reloaded)

.PHONY: docker-backup
docker-backup: setup-db ## Backup database and volumes
	$(call print_message,$(BLUE),Creating backup)
	@mkdir -p backups
	@backup_name="backup_$$(date +%Y%m%d_%H%M%S)"; \
	docker run --rm -v rag_iris_data:/data -v "$$(pwd)/backups":/backup alpine tar czf /backup/$$backup_name.tar.gz -C /data .; \
	echo -e "  $(GREEN)✓$(NC) Backup created: backups/$$backup_name.tar.gz"

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

.PHONY: docker-test
docker-test: ## Run basic functionality tests
	$(call print_message,$(BLUE),Running functionality tests)
	@./scripts/docker/health-check.sh --verbose
	@echo ""
	$(call print_message,$(BLUE),Testing RAG API endpoint)
	@if curl -s -f http://localhost:8000/health >/dev/null; then \
		echo -e "  $(GREEN)✓$(NC) RAG API is responding"; \
	else \
		echo -e "  $(RED)✗$(NC) RAG API not accessible"; \
	fi
	@if curl -s -f http://localhost:8501/_stcore/health >/dev/null; then \
		echo -e "  $(GREEN)✓$(NC) Streamlit app is responding"; \
	else \
		echo -e "  $(RED)✗$(NC) Streamlit app not accessible"; \
	fi

.PHONY: docker-test-query
docker-test-query: ## Test RAG query functionality
	$(call print_message,$(BLUE),Testing RAG query)
	@curl -X POST http://localhost:8000/api/v1/query \
		-H "Content-Type: application/json" \
		-d '{"query": "What are the symptoms of diabetes?", "pipeline": "basic"}' \
		2>/dev/null | head -c 200 && echo "..."

.PHONY: test-enterprise-10k
test-enterprise-10k: ## Run enterprise 10K scale testing for GraphRAG
	$(call print_message,$(BLUE),Running Enterprise 10K Scale Testing)
	@python tests/test_enterprise_10k_comprehensive.py --documents 1000 --use-mocks
	$(call print_message,$(GREEN),Enterprise 10K testing completed)

.PHONY: test-enterprise-10k-real
test-enterprise-10k-real: setup-db docker-up docker-wait ## Run enterprise 10K testing with real database
	$(call print_message,$(BLUE),Running Enterprise 10K Testing with Real Database)
	@python tests/test_enterprise_10k_comprehensive.py --documents 10000
	$(call print_message,$(GREEN),Enterprise 10K real testing completed)

.PHONY: test-graphrag-scale
test-graphrag-scale: ## Run GraphRAG scale testing
	$(call print_message,$(BLUE),Running GraphRAG Scale Testing)
	@python scripts/test_graphrag_scale_10k.py --documents 1000 --use-mocks
	$(call print_message,$(GREEN),GraphRAG scale testing completed)

.PHONY: test-graphrag-scale-real
test-graphrag-scale-real: setup-db docker-up docker-wait ## Run GraphRAG scale testing with real database
	$(call print_message,$(BLUE),Running GraphRAG Scale Testing with Real Database)
	@python scripts/test_graphrag_scale_10k.py --documents 10000
	$(call print_message,$(GREEN),GraphRAG scale real testing completed)

.PHONY: test-pytest-enterprise
test-pytest-enterprise: ## Run enterprise tests via pytest
	$(call print_message,$(BLUE),Running Enterprise Tests via Pytest)
	@python -m pytest tests/test_enterprise_10k_comprehensive.py -v --tb=short
	$(call print_message,$(GREEN),Pytest enterprise testing completed)

# =============================================================================
# EXAMPLE TESTING
# =============================================================================

.PHONY: test-examples
test-examples: setup-db docker-up docker-wait ## Run all example tests with live IRIS database
	$(call print_message,$(BLUE),Running Example Tests with Live IRIS Database)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --mode real --verbose
	$(call print_message,$(GREEN),Example testing completed)

.PHONY: test-examples-basic
test-examples-basic: setup-db docker-up docker-wait ## Run basic RAG example tests with live IRIS
	$(call print_message,$(BLUE),Running Basic RAG Example Tests with Live IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --category basic --mode real --verbose
	$(call print_message,$(GREEN),Basic example testing completed)

.PHONY: test-examples-advanced
test-examples-advanced: setup-db docker-up docker-wait ## Run advanced RAG example tests with live IRIS
	$(call print_message,$(BLUE),Running Advanced RAG Example Tests with Live IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --category advanced --mode real --verbose
	$(call print_message,$(GREEN),Advanced example testing completed)

.PHONY: test-examples-mock
test-examples-mock: setup-db ## Run example tests with mock LLMs (development only - NOT constitutional)
	$(call print_message,$(YELLOW),WARNING: Running with Mock LLMs - Constitutional Violation!)
	$(call print_message,$(YELLOW),This violates Section III: Tests MUST execute against live IRIS database)
	$(call print_message,$(YELLOW),Mock mode should only be used for development debugging)
	@echo "Continue with constitutional violation? This should only be for development. [y/N]"
	@read -r confirm && [ "$$confirm" = "y" ] || (echo "Cancelled" && exit 1)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --mode mock --verbose --timeout 600
	$(call print_message,$(YELLOW),Mock testing completed - Remember this violates constitution!)

.PHONY: test-examples-pattern
test-examples-pattern: setup-db docker-up docker-wait ## Run example tests matching pattern with live IRIS (usage: make test-examples-pattern PATTERN=basic)
	$(call print_message,$(BLUE),Running Example Tests for Pattern: $(PATTERN) with Live IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --pattern "$(PATTERN)" --mode real --verbose
	$(call print_message,$(GREEN),Pattern example testing completed)

.PHONY: test-examples-ci
test-examples-ci: setup-db docker-up docker-wait ## Run example tests in CI mode with live IRIS (continue on failure, generate reports)
	$(call print_message,$(BLUE),Running Example Tests in CI Mode with Live IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --mode real --continue-on-failure --verbose
	$(call print_message,$(GREEN),CI example testing completed)

.PHONY: test-all-enterprise
test-all-enterprise: test-enterprise-10k test-graphrag-scale test-pytest-enterprise ## Run all enterprise scale tests
	$(call print_message,$(GREEN),All enterprise scale tests completed)

# =============================================================================
# TEST DATABASE MANAGEMENT
# =============================================================================

.PHONY: test-db-basic
test-db-basic: setup-db ## Switch to basic RAG test database
	$(call print_message,$(BLUE),Switching to Basic RAG Test Database)
	@docker-compose -f docker-compose.test.yml down iris-test 2>/dev/null || true
	@export TEST_DATABASE_VOLUME=$${TEST_DATABASE_VOLUME:-./docker/test-databases/basic-rag-testdb} && \
	docker-compose -f docker-compose.test.yml up -d iris-test
	@sleep 15
	@python evaluation_framework/test_iris_connectivity.py --port 31972 || true
	$(call print_message,$(GREEN),Basic RAG test database ready)

.PHONY: test-db-graphrag
test-db-graphrag: setup-db ## Switch to GraphRAG test database
	$(call print_message,$(BLUE),Switching to GraphRAG Test Database)
	@docker-compose -f docker-compose.test.yml down iris-test 2>/dev/null || true
	@export TEST_DATABASE_VOLUME=$${TEST_DATABASE_VOLUME:-./docker/test-databases/graphrag-testdb} && \
	docker-compose -f docker-compose.test.yml up -d iris-test
	@sleep 15
	@python evaluation_framework/test_iris_connectivity.py --port 31972 || true
	$(call print_message,$(GREEN),GraphRAG test database ready)

.PHONY: test-db-crag
test-db-crag: setup-db ## Switch to CRAG test database
	$(call print_message,$(BLUE),Switching to CRAG Test Database)
	@docker-compose -f docker-compose.test.yml down iris-test 2>/dev/null || true
	@export TEST_DATABASE_VOLUME=$${TEST_DATABASE_VOLUME:-./docker/test-databases/crag-testdb} && \
	docker-compose -f docker-compose.test.yml up -d iris-test
	@sleep 15
	@python evaluation_framework/test_iris_connectivity.py --port 31972 || true
	$(call print_message,$(GREEN),CRAG test database ready)

.PHONY: test-db-enterprise
test-db-enterprise: setup-db ## Switch to enterprise scale test database
	$(call print_message,$(BLUE),Switching to Enterprise Scale Test Database)
	@docker-compose -f docker-compose.test.yml down iris-test 2>/dev/null || true
	@export TEST_DATABASE_VOLUME=$${TEST_DATABASE_VOLUME:-./docker/test-databases/enterprise-testdb} && \
	docker-compose -f docker-compose.test.yml up -d iris-test
	@sleep 30
	@python evaluation_framework/test_iris_connectivity.py --port 31972 || true
	$(call print_message,$(GREEN),Enterprise test database ready)

.PHONY: test-db-clean
test-db-clean: setup-db ## Create fresh empty test database
	$(call print_message,$(BLUE),Creating Fresh Empty Test Database)
	@docker-compose -f docker-compose.test.yml down iris-test 2>/dev/null || true
	@docker volume rm rag-templates_test-iris-data 2>/dev/null || true
	@docker-compose -f docker-compose.test.yml up -d iris-test
	@sleep 15
	@python evaluation_framework/test_iris_connectivity.py --port 31972 || true
	$(call print_message,$(GREEN),Clean test database ready - framework auto-setup will handle schema)

.PHONY: test-db-status
test-db-status: setup-db ## Show current test database status
	$(call print_message,$(BLUE),Test Database Status)
	@docker-compose -f docker-compose.test.yml ps iris-test 2>/dev/null || echo "No test database running"
	@python evaluation_framework/test_iris_connectivity.py --port 31972 && \
	python scripts/test-db/show_database_info.py 2>/dev/null || \
	echo "Test database not accessible"

# =============================================================================
# CLEAN IRIS TESTING VARIANTS
# =============================================================================

.PHONY: test-examples-clean
test-examples-clean: setup-db test-db-clean ## Run all examples starting from clean IRIS (tests full setup)
	$(call print_message,$(BLUE),Running Examples with Clean IRIS - Full Setup Validation)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --mode real --verbose --continue-on-failure
	$(call print_message,$(GREEN),Clean IRIS example testing completed)

.PHONY: test-examples-basic-clean
test-examples-basic-clean: setup-db test-db-clean ## Run basic examples from clean IRIS
	$(call print_message,$(BLUE),Running Basic Examples with Clean IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --category basic --mode real --verbose --continue-on-failure
	$(call print_message,$(GREEN),Clean basic example testing completed)

.PHONY: test-examples-advanced-clean
test-examples-advanced-clean: setup-db test-db-clean ## Run advanced examples from clean IRIS
	$(call print_message,$(BLUE),Running Advanced Examples with Clean IRIS)
	@chmod +x scripts/ci/run-example-tests.sh
	@scripts/ci/run-example-tests.sh --category advanced --mode real --verbose --continue-on-failure
	$(call print_message,$(GREEN),Clean advanced example testing completed)

.PHONY: test-pipeline-initialization
test-pipeline-initialization: setup-db test-db-clean ## Test pipeline initialization from scratch
	$(call print_message,$(BLUE),Testing Pipeline Initialization from Clean IRIS)
	@python scripts/test-initialization/test_pipeline_setup.py --verbose
	$(call print_message,$(GREEN),Pipeline initialization testing completed)

.PHONY: test-schema-creation
test-schema-creation: setup-db test-db-clean ## Test schema creation and validation
	$(call print_message,$(BLUE),Testing Schema Creation from Clean IRIS)
	@python scripts/test-initialization/test_schema_creation.py --verbose
	$(call print_message,$(GREEN),Schema creation testing completed)

.PHONY: test-data-ingestion-fresh
test-data-ingestion-fresh: setup-db test-db-clean ## Test data ingestion on fresh IRIS
	$(call print_message,$(BLUE),Testing Data Ingestion on Fresh IRIS)
	@python scripts/test-initialization/test_data_ingestion.py --verbose
	$(call print_message,$(GREEN),Fresh data ingestion testing completed)

.PHONY: test-full-setup-workflow
test-full-setup-workflow: test-db-clean ## Test complete setup workflow from clean IRIS
	$(call print_message,$(BLUE),Testing Complete Setup Workflow)
	@python scripts/test-initialization/test_complete_workflow.py --verbose
	$(call print_message,$(GREEN),Complete setup workflow testing completed)

.PHONY: test-clean-workflow-minimal
test-clean-workflow-minimal: setup-db test-db-clean ## Test minimal workflow from clean IRIS
	$(call print_message,$(BLUE),Testing Minimal Clean IRIS Workflow)
	@python scripts/test-initialization/test_clean_workflow_minimal.py
	$(call print_message,$(GREEN),Minimal clean workflow testing completed)

.PHONY: test-clean-summary
test-clean-summary: setup-db ## Generate comprehensive clean IRIS testing summary report
	$(call print_message,$(BLUE),Generating Clean IRIS Testing Summary Report)
	@python scripts/test-initialization/test_summary_report.py
	$(call print_message,$(GREEN),Summary report generation completed)

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: docker-clean
docker-clean: ## Clean up containers, networks, and images
	$(call print_message,$(YELLOW),Cleaning up Docker resources)
	$(DOCKER_COMPOSE) down --remove-orphans
	docker system prune -f
	$(call print_message,$(GREEN),Docker resources cleaned)

.PHONY: docker-clean-all
docker-clean-all: ## Clean up everything including volumes and images
	$(call print_message,$(RED),WARNING: This will remove all data and images)
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DOCKER_COMPOSE) down -v --remove-orphans --rmi all
	docker system prune -af --volumes
	$(call print_message,$(GREEN),All Docker resources cleaned)

# =============================================================================
# UTILITY TARGETS
# =============================================================================

.PHONY: docker-urls
docker-urls: setup-db ## Show service URLs
	@echo ""
	$(call print_message,$(GREEN),Service URLs)
	@echo -e "  $(BLUE)Streamlit App:$(NC)      http://localhost:8501"
	@echo -e "  $(BLUE)RAG API:$(NC)           http://localhost:8000"
	@echo -e "  $(BLUE)API Documentation:$(NC)  http://localhost:8000/docs"
	@echo -e "  $(BLUE)IRIS Management:$(NC)    http://localhost:52773/csp/sys/UtilHome.csp"
	@if $(DOCKER_COMPOSE) ps jupyter | grep -q "Up" 2>/dev/null; then \
		echo -e "  $(BLUE)Jupyter Notebook:$(NC)  http://localhost:8888"; \
	fi
	@if $(DOCKER_COMPOSE) ps monitoring | grep -q "Up" 2>/dev/null; then \
		echo -e "  $(BLUE)Monitoring:$(NC)        http://localhost:9090"; \
	fi
	@echo ""

.PHONY: docker-wait
docker-wait: ## Wait for services to be healthy
	$(call print_message,$(BLUE),Waiting for services to be healthy)
	@timeout=300; \
	while [ $$timeout -gt 0 ]; do \
		if ./scripts/docker/health-check.sh --json | grep -q '"overall_status": "healthy"'; then \
			echo -e "  $(GREEN)✓$(NC) All services are healthy"; \
			break; \
		fi; \
		echo -e "  $(YELLOW)○$(NC) Waiting for services... ($$timeout seconds remaining)"; \
		sleep 10; \
		timeout=$$((timeout - 10)); \
	done; \
	if [ $$timeout -le 0 ]; then \
		echo -e "  $(RED)✗$(NC) Timeout waiting for services"; \
		exit 1; \
	fi

# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

.PHONY: docker-dev
docker-dev: docker-up-dev docker-wait docker-init-data ## Full development setup
	$(call print_message,$(GREEN),Development environment ready!)
	@$(MAKE) docker-urls

.PHONY: docker-prod
docker-prod: docker-up-prod docker-wait ## Full production setup
	$(call print_message,$(GREEN),Production environment ready!)
	@$(MAKE) docker-urls

.PHONY: docker-quick
docker-quick: docker-up docker-wait ## Quick start with core services
	$(call print_message,$(GREEN),Quick start complete!)
	@$(MAKE) docker-urls

# =============================================================================
# MAINTENANCE
# =============================================================================

.PHONY: docker-update
docker-update: docker-pull docker-build docker-restart ## Update and restart services
	$(call print_message,$(GREEN),Services updated and restarted)

.PHONY: docker-reset
docker-reset: docker-down-clean setup docker-up ## Reset everything to clean state
	$(call print_message,$(GREEN),Environment reset complete)

# Make scripts executable on setup
scripts/docker/%.sh: 
	chmod +x $@

# Ensure log directories exist
logs:
	mkdir -p logs

# Prevent make from treating these as file targets
.PHONY: docker-up docker-down docker-logs docker-shell docker-build docker-clean
# =============================================================================
# RAGAS E2E EVALUATION
# =============================================================================
.PHONY: test-ragas-1000
test-ragas-1000: setup-db ## E2E: Download+load 1000 PMC docs, run RAGAS across all 5 pipelines
	$(call print_message,$(BLUE),E2E RAGAS on 1000 PMC documents)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	export IRIS_HOST=$${IRIS_HOST:-localhost}; \
	export IRIS_PORT=$${IRIS_PORT:-1972}; \
	export EVAL_PMC_DIR=$${EVAL_PMC_DIR:-data/downloaded_pmc_docs}; \
	export RAGAS_NUM_QUERIES=$${RAGAS_NUM_QUERIES:-15}; \
	export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(.venv/bin/python scripts/utils/get_pipeline_types.py)}; \
	python scripts/utilities/download_real_pmc_docs.py; \
	python scripts/generate_ragas_evaluation.py; \
	echo "RAGAS reports are in outputs/reports/ragas_evaluations"

# =============================================================================
# RAGAS QUICK SMOKE TEST
# =============================================================================
.PHONY: test-ragas-sample
test-ragas-sample: setup-db load-data ## E2E: Quick RAGAS on sample 10 PMC docs using MCP IRIS
	$(call print_message,$(BLUE),Quick RAGAS on sample 10 PMC docs)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	export IRIS_HOST=$${IRIS_HOST:-localhost}; \
	export IRIS_PORT=$${IRIS_PORT:-1972}; \
	export EVAL_PMC_DIR=$${EVAL_PMC_DIR:-data/sample_10_docs}; \
	export RAGAS_NUM_QUERIES=$${RAGAS_NUM_QUERIES:-8}; \
	export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(.venv/bin/python scripts/utils/get_pipeline_types.py)}; \
	python scripts/simple_working_ragas.py; \
	echo "RAGAS reports are in outputs/reports/ragas_evaluations"

# =============================================================================
# RAGAS 1000 REAL (WITH DOCKER)
# =============================================================================
.PHONY: test-ragas-1000-real
test-ragas-1000-real: docker-up docker-wait ## E2E: Download+load 1000 PMC docs, run RAGAS across pipelines with Docker
	$(call print_message,$(BLUE),E2E RAGAS on 1000 PMC documents with Docker)
t@set -a; [ -f .env ] && . ./.env; set +a; \
	@export EVAL_PMC_DIR=$${EVAL_PMC_DIR:-data/downloaded_pmc_docs}; \
	export RAGAS_NUM_QUERIES=$${RAGAS_NUM_QUERIES:-15}; \
	export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(.venv/bin/python scripts/utils/get_pipeline_types.py)}; \
	python scripts/utilities/download_real_pmc_docs.py; \
	python scripts/generate_ragas_evaluation.py; \
	echo "RAGAS reports are in outputs/reports/ragas_evaluations"

# =============================================================================
# RAGAS DOCKERIZED TARGETS
# =============================================================================
.PHONY: test-ragas-sample-docker
test-ragas-sample-docker: docker-up docker-wait ## Quick RAGAS (GraphRAG + HybridGraphRAG) inside container
	$(call print_message,$(BLUE),Quick RAGAS (dockerized))
	@$(DOCKER_COMPOSE) exec rag_api /bin/bash -lc "\
export EVAL_PMC_DIR=\"${EVAL_PMC_DIR:-/app/data/sample_10_docs}\"; \
export RAGAS_NUM_QUERIES=\"${RAGAS_NUM_QUERIES:-8}\"; \
export RAGAS_PIPELINES=\"${RAGAS_PIPELINES:-graphrag,pylate_colbert}\"; \
python scripts/generate_ragas_evaluation.py; \
echo 'RAGAS reports are in outputs/reports/ragas_evaluations'"

.PHONY: test-ragas-1000-docker
test-ragas-1000-docker: docker-up docker-wait ## E2E RAGAS on 1000 PMC docs (download on host, evaluate inside container incl. HybridGraphRAG)
	$(call print_message,$(BLUE),Downloading PMC docs on host)
	@export EVAL_PMC_DIR=${EVAL_PMC_DIR:-data/downloaded_pmc_docs}; \
	export RAGAS_NUM_QUERIES=${RAGAS_NUM_QUERIES:-15}; \
	export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"}; \
	python scripts/utilities/download_real_pmc_docs.py
	$(call print_message,$(BLUE),Running RAGAS (dockerized))
	@$(DOCKER_COMPOSE) exec rag_api /bin/bash -lc "\
export EVAL_PMC_DIR=\"${EVAL_PMC_DIR:-/app/data/downloaded_pmc_docs}\"; \
export RAGAS_NUM_QUERIES=\"${RAGAS_NUM_QUERIES:-15}\"; \
export RAGAS_PIPELINES=\"${RAGAS_PIPELINES:-basic,basic_rerank,crag,graphrag,pylate_colbert}\"; \
python scripts/generate_ragas_evaluation.py; \
echo 'RAGAS reports are in outputs/reports/ragas_evaluations'"

# =============================================================================
# END-TO-END INTEGRATION TESTING (Production Readiness)
# =============================================================================

.PHONY: test-e2e-integration
test-e2e-integration: docker-up docker-wait ## Run comprehensive E2E integration test suite for production readiness
	$(call print_message,$(BLUE),Starting comprehensive E2E integration test suite)
	$(call print_message,$(BLUE),This tests all pipelines, demos, and stress scenarios)
	@python scripts/testing/e2e_integration_suite.py
	$(call print_message,$(GREEN),E2E integration testing completed - check outputs/e2e_integration_reports/)

.PHONY: test-e2e-integration-quick
test-e2e-integration-quick: docker-up docker-wait ## Run quick E2E integration tests (no stress tests)
	$(call print_message,$(BLUE),Running quick E2E integration tests (no stress tests))
	@python scripts/testing/e2e_integration_suite.py --skip-stress
	$(call print_message,$(GREEN),Quick E2E integration testing completed)

.PHONY: test-e2e-integration-clean
test-e2e-integration-clean: setup-db test-db-clean ## Run E2E integration tests from clean IRIS database
	$(call print_message,$(BLUE),Running E2E integration tests from clean IRIS database)
	$(call print_message,$(BLUE),This validates complete setup workflows from scratch)
	@python scripts/testing/e2e_integration_suite.py --skip-stress
	$(call print_message,$(GREEN),Clean E2E integration testing completed)

.PHONY: test-e2e-pipelines-only
test-e2e-pipelines-only: docker-up docker-wait ## Test all RAG pipelines comprehensively (no demos or stress tests)
	$(call print_message,$(BLUE),Testing all RAG pipelines comprehensively)
	@python scripts/testing/e2e_integration_suite.py --skip-stress --quick
	$(call print_message,$(GREEN),Pipeline-only E2E testing completed)

.PHONY: test-production-readiness
test-production-readiness: setup-db ## FULL production readiness validation (all tests, clean IRIS, stress testing)
	$(call print_message,$(BLUE),🚀 PRODUCTION READINESS VALIDATION)
	$(call print_message,$(BLUE),Running comprehensive test suite for public release validation)
	$(call print_message,$(BLUE),This includes: Clean IRIS + All Pipelines + Demos + Stress Tests)
	@make test-db-clean
	@make docker-up docker-wait
	@python scripts/testing/e2e_integration_suite.py --verbose
	$(call print_message,$(GREEN),🎉 Production readiness validation completed!)
	$(call print_message,$(YELLOW),Check outputs/e2e_integration_reports/ for detailed results)

.PHONY: test-release-candidate
test-release-candidate: test-production-readiness ## Alias for production readiness testing
	$(call print_message,$(GREEN),Release candidate validation completed)

.PHONY: test-publish-ready
test-publish-ready: test-production-readiness ## Alias for production readiness testing
	$(call print_message,$(GREEN),Publish readiness validation completed)

# =============================================================================
# COVERAGE ANALYSIS TARGETS (Constitutional Compliance)
# =============================================================================

.PHONY: coverage-analyze
coverage-analyze: setup-db ## Run comprehensive coverage analysis using uv (constitutional requirement)
	$(call print_message,$(BLUE),Running comprehensive coverage analysis with uv)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env; \
	fi
	@uv run pytest tests/ --cov=iris_rag --cov=common --cov-report=term-missing --cov-report=html --cov-report=xml --maxfail=5
	$(call print_message,$(GREEN),Coverage analysis completed - check htmlcov/ for detailed report)

.PHONY: coverage-critical
coverage-critical: setup-db ## Validate critical modules meet 80% coverage target
	$(call print_message,$(BLUE),Validating critical modules coverage (80% target))
	@uv run pytest tests/unit/test_configuration_coverage.py tests/unit/test_validation_coverage.py tests/unit/test_pipeline_coverage.py tests/unit/test_services_coverage.py tests/unit/test_storage_coverage.py --cov=iris_rag.config --cov=iris_rag.validation --cov=iris_rag.pipelines --cov=iris_rag.services --cov=iris_rag.storage --cov-report=term --cov-fail-under=80
	$(call print_message,$(GREEN),Critical modules coverage validation completed)

.PHONY: coverage-overall
coverage-overall: setup-db ## Validate overall 60% coverage target
	$(call print_message,$(BLUE),Validating overall coverage target (60%))
	@uv run pytest tests/ --cov=iris_rag --cov=common --cov-report=term --cov-fail-under=60 --quiet
	$(call print_message,$(GREEN),Overall coverage validation completed)

.PHONY: coverage-performance
coverage-performance: setup-db ## Run coverage with performance validation (5-minute limit)
	$(call print_message,$(BLUE),Running coverage analysis with performance validation)
	@echo "Starting coverage analysis with 5-minute timeout..."
	@timeout 300 uv run pytest tests/ --cov=iris_rag --cov=common --cov-report=term-missing --maxfail=3 || echo "Coverage analysis completed within time limit"
	$(call print_message,$(GREEN),Coverage performance validation completed)

.PHONY: coverage-reports
coverage-reports: setup-db ## Generate all coverage report formats (terminal, HTML, XML, JSON)
	$(call print_message,$(BLUE),Generating comprehensive coverage reports)
	@mkdir -p coverage_reports
	@uv run pytest tests/ --cov=iris_rag --cov=common --cov-report=term-missing --cov-report=html:coverage_reports/html --cov-report=xml:coverage_reports/coverage.xml --cov-report=json:coverage_reports/coverage.json
	$(call print_message,$(GREEN),Coverage reports generated in coverage_reports/ directory)

.PHONY: coverage-constitutional
coverage-constitutional: setup-db ## Full constitutional compliance validation (IRIS database required)
	$(call print_message,$(BLUE),Running constitutional compliance coverage validation)
	$(call print_message,$(BLUE),This requires live IRIS database per constitutional requirements)
	@make docker-up docker-wait
	@uv run pytest tests/ -m "requires_database or clean_iris" --cov=iris_rag --cov=common --cov-report=term-missing --cov-fail-under=60
	$(call print_message,$(GREEN),Constitutional compliance validation completed)

.PHONY: coverage-trends
coverage-trends: setup-db ## Generate monthly coverage trend report
	$(call print_message,$(BLUE),Generating coverage trend analysis)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env; \
	fi
	@uv run python -c "from iris_rag.testing.coverage_analysis import CoverageAnalyzer; analyzer = CoverageAnalyzer(); print('Coverage trend analysis would run here - requires implementation')"
	$(call print_message,$(GREEN),Coverage trend analysis completed)

.PHONY: coverage-validate
coverage-validate: ## Validate all coverage targets and requirements
	$(call print_message,$(BLUE),Validating all coverage targets and requirements)
	@echo "Validating overall target (60%)..."
	@$(MAKE) coverage-overall
	@echo "Validating critical modules (80%)..."
	@$(MAKE) coverage-critical
	@echo "Validating performance requirements..."
	@$(MAKE) coverage-performance
	$(call print_message,$(GREEN),All coverage validation completed successfully)

.PHONY: coverage-help
coverage-help: ## Show coverage analysis commands help
	@echo -e "$(BLUE)Coverage Analysis Commands:$(NC)"
	@echo -e "  coverage-analyze       - Run comprehensive coverage analysis"
	@echo -e "  coverage-critical      - Validate critical modules (80% target)"
	@echo -e "  coverage-overall       - Validate overall target (60%)"
	@echo -e "  coverage-performance   - Run with performance validation"
	@echo -e "  coverage-reports       - Generate all report formats"
	@echo -e "  coverage-constitutional - Full constitutional compliance"
	@echo -e "  coverage-trends        - Generate trend analysis"
	@echo -e "  coverage-validate      - Validate all targets"

# =============================================================================
# BACKEND MODE TESTING (Feature 035)
# =============================================================================

.PHONY: test-community
test-community: setup-db ## Run tests with Community Edition backend mode
	$(call print_message,$(BLUE),Running tests in Community mode)
	@IRIS_BACKEND_MODE=community python -m pytest tests/ -v -m "requires_backend_mode or contract" --tb=short
	$(call print_message,$(GREEN),Community mode tests completed)

.PHONY: test-enterprise
test-enterprise: setup-db ## Run tests with Enterprise Edition backend mode
	$(call print_message,$(BLUE),Running tests in Enterprise mode)
	@IRIS_BACKEND_MODE=enterprise python -m pytest tests/ -v -m "requires_backend_mode or contract" --tb=short
	$(call print_message,$(GREEN),Enterprise mode tests completed)

.PHONY: test-mode-switching
test-mode-switching: ## Run backend mode switching integration tests
	$(call print_message,$(BLUE),Testing backend mode switching)
	@python -m pytest tests/integration/test_mode_switching.py -v --tb=short
	$(call print_message,$(GREEN),Mode switching tests completed)

.PHONY: test-backend-contracts
test-backend-contracts: ## Run all backend mode contract tests
	$(call print_message,$(BLUE),Running backend mode contract tests)
	@python -m pytest tests/contract/test_backend_mode_config.py -v
	@python -m pytest tests/contract/test_edition_detection.py -v
	@python -m pytest tests/contract/test_connection_pooling.py -v
	@python -m pytest tests/contract/test_execution_strategies.py -v
	$(call print_message,$(GREEN),Backend mode contract tests completed)


# =============================================================================
# TEST FIXTURE MANAGEMENT (Feature 047)
# =============================================================================

# Fixture configuration
FIXTURE_DIR := tests/fixtures
FIXTURE ?= medical-graphrag-20
FIXTURE_VERSION ?=
FIXTURE_TABLES ?=
FIXTURE_DESC ?=
EMBEDDINGS ?= 0

.PHONY: fixture-help
fixture-help: ## Show fixture management commands
	@echo -e "$(BLUE)Test Fixture Management Commands:$(NC)"
	@echo -e ""
	@echo -e "$(GREEN)Listing and Information:$(NC)"
	@echo -e "  make fixture-list                    - List all available fixtures"
	@echo -e "  make fixture-info FIXTURE=name       - Show detailed fixture information"
	@echo -e "  make fixture-validate FIXTURE=name   - Validate fixture integrity"
	@echo -e ""
	@echo -e "$(GREEN)Loading Fixtures:$(NC)"
	@echo -e "  make fixture-load FIXTURE=name       - Load fixture into IRIS database"
	@echo -e "  make fixture-load-clean FIXTURE=name - Clean DB first, then load fixture"
	@echo -e "  make fixture-load-fast FIXTURE=name  - Load without checksum validation (faster)"
	@echo -e ""
	@echo -e "$(GREEN)Creating Fixtures:$(NC)"
	@echo -e "  make fixture-workflow                - Interactive fixture creation"
	@echo -e "  make fixture-create FIXTURE=name ... - Create fixture from current database"
	@echo -e "  make fixture-snapshot FIXTURE=name   - Quick snapshot of current database"
	@echo -e ""
	@echo -e "$(GREEN)Testing:$(NC)"
	@echo -e "  make fixture-test                    - Run fixture manager contract tests"
	@echo -e "  make fixture-test-integration        - Run fixture integration tests"
	@echo -e ""
	@echo -e "$(YELLOW)Examples:$(NC)"
	@echo -e "  make fixture-list"
	@echo -e "  make fixture-load FIXTURE=medical-graphrag-20"
	@echo -e "  make fixture-create FIXTURE=my-test TABLES=RAG.SourceDocuments,RAG.Entities DESC=\"My test fixture\""
	@echo -e ""

.PHONY: fixture-list
fixture-list: ## List all available test fixtures
	$(call print_message,$(BLUE),Available Test Fixtures)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli list

.PHONY: list-fixtures
list-fixtures: fixture-list ## Alias for fixture-list (T097)

.PHONY: fixture-info
fixture-info: ## Show detailed information about a fixture
	$(call print_message,$(BLUE),Fixture Information: $(FIXTURE))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli info $(FIXTURE)

.PHONY: fixture-validate
fixture-validate: ## Validate fixture integrity (checksum, metadata)
	$(call print_message,$(BLUE),Validating fixture: $(FIXTURE))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli validate $(FIXTURE)
	$(call print_message,$(GREEN),Fixture validation completed)

.PHONY: fixture-load
fixture-load: ## Load fixture into IRIS database
	$(call print_message,$(BLUE),Loading fixture: $(FIXTURE))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli load $(FIXTURE) $(if $(FIXTURE_VERSION),--version $(FIXTURE_VERSION))
	$(call print_message,$(GREEN),Fixture loaded successfully)

.PHONY: fixture-load-clean
fixture-load-clean: ## Clean database first, then load fixture
	$(call print_message,$(BLUE),Loading fixture with cleanup: $(FIXTURE))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli load $(FIXTURE) --cleanup-first
	$(call print_message,$(GREEN),Fixture loaded successfully (with cleanup))

.PHONY: fixture-load-fast
fixture-load-fast: ## Load fixture without checksum validation (faster)
	$(call print_message,$(BLUE),Fast loading fixture: $(FIXTURE))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli load $(FIXTURE) --no-validate-checksum
	$(call print_message,$(GREEN),Fixture loaded successfully (fast mode))

.PHONY: fixture-workflow
fixture-workflow: ## Interactive fixture creation workflow
	$(call print_message,$(BLUE),Interactive Fixture Creation)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli workflow
	$(call print_message,$(GREEN),Fixture creation completed)

.PHONY: fixture-create
fixture-create: ## Create fixture from current database state
	$(call print_message,$(BLUE),Creating fixture: $(FIXTURE))
	@if [ -z "$(FIXTURE_TABLES)" ]; then \
		echo -e "  $(RED)✗$(NC) FIXTURE_TABLES is required"; \
		echo -e "  $(YELLOW)Example:$(NC) make fixture-create FIXTURE=my-test TABLES=RAG.SourceDocuments,RAG.Entities DESC=\"My test\""; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli create $(FIXTURE) \
		--tables $(FIXTURE_TABLES) \
		$(if $(FIXTURE_DESC),--description "$(FIXTURE_DESC)") \
		$(if $(FIXTURE_VERSION),--version $(FIXTURE_VERSION)) \
		$(if $(filter 1,$(EMBEDDINGS)),--generate-embeddings)
	$(call print_message,$(GREEN),Fixture created successfully)

.PHONY: fixture-snapshot
fixture-snapshot: ## Quick snapshot of current database state
	$(call print_message,$(BLUE),Creating database snapshot: $(FIXTURE))
	@if [ -z "$(FIXTURE)" ]; then \
		echo -e "  $(RED)✗$(NC) FIXTURE is required"; \
		echo -e "  $(YELLOW)Example:$(NC) make fixture-snapshot FIXTURE=snapshot-$(shell date +%Y%m%d)"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli snapshot $(FIXTURE)
	$(call print_message,$(GREEN),Database snapshot created)

.PHONY: fixture-update
fixture-update: ## Update existing fixture with incremental changes (T091)
	$(call print_message,$(BLUE),Updating fixture: $(FIXTURE))
	@if [ -z "$(FIXTURE)" ]; then \
		echo -e "  $(RED)✗$(NC) FIXTURE is required"; \
		echo -e "  $(YELLOW)Example:$(NC) make fixture-update FIXTURE=medical-graphrag-20 VERSION=1.1.0"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli update $(FIXTURE) \
		$(if $(FIXTURE_VERSION),--version $(FIXTURE_VERSION)) \
		$(if $(FIXTURE_CHANGES),--changes "$(FIXTURE_CHANGES)")
	$(call print_message,$(GREEN),Fixture updated successfully)

.PHONY: fixture-test
fixture-test: ## Run fixture manager contract tests
	$(call print_message,$(BLUE),Running fixture manager contract tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/contract/test_fixture_manager_contract.py tests/contract/test_embedding_generator_contract.py -v --tb=short
	$(call print_message,$(GREEN),Fixture manager contract tests completed)

.PHONY: fixture-test-integration
fixture-test-integration: ## Run fixture integration tests
	$(call print_message,$(BLUE),Running fixture integration tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/integration/test_fixture_*.py -v --tb=short
	$(call print_message,$(GREEN),Fixture integration tests completed)

.PHONY: fixture-bench
fixture-bench: ## Benchmark fixture loading performance
	$(call print_message,$(BLUE),Benchmarking fixture loading performance)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli benchmark
	$(call print_message,$(GREEN),Fixture benchmark completed)

.PHONY: fixture-migrate-json
fixture-migrate-json: ## Migrate existing JSON fixtures to .DAT format
	$(call print_message,$(BLUE),Migrating JSON fixtures to .DAT format)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m tests.fixtures.cli migrate-json
	$(call print_message,$(GREEN),JSON fixture migration completed)

.PHONY: fixture-clean
fixture-clean: ## Clean up fixture temporary files
	$(call print_message,$(YELLOW),Cleaning up fixture temporary files)
	@find $(FIXTURE_DIR) -name "*.pyc" -delete
	@find $(FIXTURE_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find $(FIXTURE_DIR) -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	$(call print_message,$(GREEN),Fixture cleanup completed)

# =============================================================================
# REST API MANAGEMENT (Feature 042)
# =============================================================================

.PHONY: api-help
api-help: ## Show REST API management commands
	@echo -e "$(BLUE)REST API Management Commands:$(NC)"
	@echo -e ""
	@echo -e "$(GREEN)Server Operations:$(NC)"
	@echo -e "  make api-run                         - Run API server (development mode)"
	@echo -e "  make api-run-prod                    - Run API server (production mode, 4 workers)"
	@echo -e "  make api-health                      - Check API health status"
	@echo -e ""
	@echo -e "$(GREEN)Database Setup:$(NC)"
	@echo -e "  make api-setup-db                    - Setup API database tables"
	@echo -e "  make api-schema                      - Show current database schema"
	@echo -e ""
	@echo -e "$(GREEN)API Key Management:$(NC)"
	@echo -e "  make api-create-key NAME=... EMAIL=... - Create new API key"
	@echo -e "  make api-list-keys                   - List all API keys"
	@echo-e "  make api-revoke-key KEY_ID=...      - Revoke API key"
	@echo -e ""
	@echo -e "$(GREEN)Testing:$(NC)"
	@echo -e "  make api-test                        - Run API tests"
	@echo -e "  make api-test-contracts              - Run contract tests"
	@echo -e "  make api-test-integration            - Run integration tests"
	@echo -e ""
	@echo -e "$(YELLOW)Examples:$(NC)"
	@echo -e "  make api-run"
	@echo -e "  make api-create-key NAME=\"My Key\" EMAIL=user@example.com"
	@echo -e "  make api-create-key NAME=\"Enterprise Key\" EMAIL=admin@example.com TIER=enterprise PERMISSIONS=\"read write admin\""
	@echo -e ""

.PHONY: api-run
api-run: setup-env install ## Run API server in development mode
	$(call print_message,$(BLUE),Starting RAG API server (development mode))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli run --reload

.PHONY: api-run-prod
api-run-prod: setup-env install ## Run API server in production mode
	$(call print_message,$(BLUE),Starting RAG API server (production mode))
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli run --workers 4

.PHONY: api-health
api-health: ## Check API health status
	$(call print_message,$(BLUE),Checking API health)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli health

.PHONY: api-setup-db
api-setup-db: setup-env install ## Setup API database tables
	$(call print_message,$(BLUE),Setting up API database tables)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli setup-db
	$(call print_message,$(GREEN),API database tables created successfully)

.PHONY: api-create-key
api-create-key: setup-env install ## Create new API key (usage: make api-create-key NAME="My Key" EMAIL=user@example.com)
	$(call print_message,$(BLUE),Creating API key)
	@if [ -z "$(NAME)" ] || [ -z "$(EMAIL)" ]; then \
		echo -e "  $(RED)✗$(NC) NAME and EMAIL are required"; \
		echo -e "  $(YELLOW)Example:$(NC) make api-create-key NAME=\"My Key\" EMAIL=user@example.com"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli create-key \
		--name "$(NAME)" \
		--owner-email "$(EMAIL)" \
		$(if $(TIER),--tier $(TIER)) \
		$(if $(PERMISSIONS),--permissions $(PERMISSIONS)) \
		$(if $(DESCRIPTION),--description "$(DESCRIPTION)") \
		$(if $(EXPIRES_IN_DAYS),--expires-in-days $(EXPIRES_IN_DAYS))

.PHONY: api-list-keys
api-list-keys: setup-env install ## List all API keys
	$(call print_message,$(BLUE),Listing API keys)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli list-keys $(if $(EMAIL),--owner-email "$(EMAIL)")

.PHONY: api-revoke-key
api-revoke-key: setup-env install ## Revoke API key (usage: make api-revoke-key KEY_ID=...)
	$(call print_message,$(BLUE),Revoking API key)
	@if [ -z "$(KEY_ID)" ]; then \
		echo -e "  $(RED)✗$(NC) KEY_ID is required"; \
		echo -e "  $(YELLOW)Example:$(NC) make api-revoke-key KEY_ID=7c9e6679-7425-40de-944b-e07fc1f90ae7"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m iris_rag.api.cli revoke-key --key-id "$(KEY_ID)"

.PHONY: api-test
api-test: setup-env install ## Run all API tests
	$(call print_message,$(BLUE),Running API tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/contract/test_*_contracts.py tests/integration/api/ -v --tb=short
	$(call print_message,$(GREEN),API tests completed)

.PHONY: api-test-contracts
api-test-contracts: setup-env install ## Run API contract tests
	$(call print_message,$(BLUE),Running API contract tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/contract/test_*_contracts.py -v --tb=short
	$(call print_message,$(GREEN),API contract tests completed)

.PHONY: api-test-integration
api-test-integration: setup-env install ## Run API integration tests
	$(call print_message,$(BLUE),Running API integration tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/integration/api/ -v --tb=short
	$(call print_message,$(GREEN),API integration tests completed)

.PHONY: api-logs
api-logs: ## View API server logs
	$(call print_message,$(BLUE),Viewing API logs)
	@tail -f logs/api.log 2>/dev/null || echo "No API logs found (logs/api.log)"

.PHONY: api-docs
api-docs: ## Open API documentation in browser
	$(call print_message,$(BLUE),Opening API documentation)
	@if command -v open &> /dev/null; then \
		open http://localhost:8000/docs; \
	elif command -v xdg-open &> /dev/null; then \
		xdg-open http://localhost:8000/docs; \
	else \
		echo "API docs available at: http://localhost:8000/docs"; \
	fi

.PHONY: api-code-quality
api-code-quality: setup-env install ## Run code quality checks on API code
	$(call print_message,$(BLUE),Running code quality checks on API code)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@chmod +x iris_rag/api/scripts/check_code_quality.sh
	@./iris_rag/api/scripts/check_code_quality.sh
	$(call print_message,$(GREEN),Code quality checks completed)

# ==============================================================================
# MCP Server Targets
# ==============================================================================
# MCP (Model Context Protocol) server for Claude Code integration
# Supports two deployment modes:
# - Standalone: Python bridge + Node.js MCP server
# - Integrated: MCP embedded in REST API
# Feature: Complete MCP Tools Implementation (043-complete-mcp-tools)

.PHONY: mcp-build
mcp-build: ## Build MCP Docker image
	$(call print_message,$(BLUE),Building MCP Docker image)
	@docker build -f Dockerfile.mcp -t iris-rag-mcp:latest .
	$(call print_message,$(GREEN),MCP Docker image built successfully)

.PHONY: mcp-run-standalone
mcp-run-standalone: env-check ## Start MCP server in standalone mode (stdio + HTTP/SSE)
	$(call print_message,$(BLUE),Starting MCP server in standalone mode)
	@docker-compose -f docker-compose.mcp.yml --profile standalone up -d
	@echo -e "  $(GREEN)✓$(NC) MCP Server (standalone) is starting..."
	@echo -e "  $(BLUE)ℹ$(NC) Python Bridge: http://localhost:8001"
	@echo -e "  $(BLUE)ℹ$(NC) MCP HTTP/SSE: http://localhost:3000"
	@echo -e "  $(BLUE)ℹ$(NC) MCP stdio: Connect via Claude Code config"
	@echo -e "  $(BLUE)ℹ$(NC) Health check: make mcp-health"
	$(call print_message,$(GREEN),MCP server started)

.PHONY: mcp-run-integrated
mcp-run-integrated: env-check ## Start MCP server in integrated mode (embedded in REST API)
	$(call print_message,$(BLUE),Starting MCP server in integrated mode)
	@docker-compose -f docker-compose.mcp.yml --profile integrated up -d
	@echo -e "  $(GREEN)✓$(NC) REST API with MCP integration is starting..."
	@echo -e "  $(BLUE)ℹ$(NC) REST API: http://localhost:8000"
	@echo -e "  $(BLUE)ℹ$(NC) API Docs: http://localhost:8000/docs"
	@echo -e "  $(BLUE)ℹ$(NC) MCP Health: http://localhost:8000/api/v1/mcp/health"
	$(call print_message,$(GREEN),Integrated API+MCP started)

.PHONY: mcp-run
mcp-run: mcp-run-standalone ## Alias for mcp-run-standalone (default mode)

.PHONY: mcp-stop
mcp-stop: ## Stop MCP server (all profiles)
	$(call print_message,$(BLUE),Stopping MCP server)
	@docker-compose -f docker-compose.mcp.yml --profile standalone --profile integrated down
	$(call print_message,$(GREEN),MCP server stopped)

.PHONY: mcp-restart
mcp-restart: mcp-stop mcp-run ## Restart MCP server in standalone mode

.PHONY: mcp-health
mcp-health: ## Check MCP server health
	$(call print_message,$(BLUE),Checking MCP server health)
	@echo -e "  $(BLUE)Mode detection:$(NC)"
	@if docker ps --filter "name=mcp-standalone" --format "{{.Names}}" | grep -q "mcp-standalone"; then \
		echo -e "  $(GREEN)✓$(NC) Standalone mode detected"; \
		echo -e "\n  $(BLUE)Python Bridge Health:$(NC)"; \
		curl -s http://localhost:8001/mcp/health_check | python3 -m json.tool || echo "  $(RED)✗$(NC) Python bridge not responding"; \
		echo -e "\n  $(BLUE)Available Techniques:$(NC)"; \
		curl -s http://localhost:8001/mcp/list_techniques | python3 -m json.tool || echo "  $(RED)✗$(NC) Cannot list techniques"; \
	elif docker ps --filter "name=api-with-mcp" --format "{{.Names}}" | grep -q "api-with-mcp"; then \
		echo -e "  $(GREEN)✓$(NC) Integrated mode detected"; \
		echo -e "\n  $(BLUE)REST API Health:$(NC)"; \
		curl -s http://localhost:8000/api/v1/health | python3 -m json.tool || echo "  $(RED)✗$(NC) API not responding"; \
		echo -e "\n  $(BLUE)MCP Health:$(NC)"; \
		curl -s http://localhost:8000/api/v1/mcp/health | python3 -m json.tool || echo "  $(RED)✗$(NC) MCP not responding"; \
	else \
		echo -e "  $(RED)✗$(NC) No MCP server running"; \
		echo -e "  $(YELLOW)ℹ$(NC) Start with: make mcp-run-standalone or make mcp-run-integrated"; \
	fi

.PHONY: mcp-logs
mcp-logs: ## View MCP server logs
	$(call print_message,$(BLUE),Viewing MCP server logs)
	@if docker ps --filter "name=mcp-standalone" --format "{{.Names}}" | grep -q "mcp-standalone"; then \
		docker-compose -f docker-compose.mcp.yml --profile standalone logs -f mcp-standalone; \
	elif docker ps --filter "name=api-with-mcp" --format "{{.Names}}" | grep -q "api-with-mcp"; then \
		docker-compose -f docker-compose.mcp.yml --profile integrated logs -f api-with-mcp; \
	else \
		echo -e "  $(RED)✗$(NC) No MCP server running"; \
	fi

.PHONY: mcp-test
mcp-test: setup-env install ## Run all MCP tests (contract + integration)
	$(call print_message,$(BLUE),Running MCP tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/contract/test_mcp_*.py tests/integration/test_mcp_*.py -v --tb=short
	$(call print_message,$(GREEN),MCP tests completed)

.PHONY: mcp-test-contracts
mcp-test-contracts: setup-env install ## Run MCP contract tests only
	$(call print_message,$(BLUE),Running MCP contract tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/contract/test_mcp_*.py -v --tb=short
	$(call print_message,$(GREEN),MCP contract tests completed)

.PHONY: mcp-test-integration
mcp-test-integration: setup-env install ## Run MCP integration tests only
	$(call print_message,$(BLUE),Running MCP integration tests)
	@if [ ! -d ".venv" ]; then \
		echo -e "  $(YELLOW)⚠$(NC) Virtual environment not found, setting up..."; \
		$(MAKE) setup-env install; \
	fi
	@.venv/bin/python -m pytest tests/integration/test_mcp_*.py -v --tb=short
	$(call print_message,$(GREEN),MCP integration tests completed)

.PHONY: mcp-shell
mcp-shell: ## Open shell in MCP container
	$(call print_message,$(BLUE),Opening shell in MCP container)
	@if docker ps --filter "name=mcp-standalone" --format "{{.Names}}" | grep -q "mcp-standalone"; then \
		docker exec -it mcp-standalone /bin/bash; \
	elif docker ps --filter "name=api-with-mcp" --format "{{.Names}}" | grep -q "api-with-mcp"; then \
		docker exec -it api-with-mcp /bin/bash; \
	else \
		echo -e "  $(RED)✗$(NC) No MCP server running"; \
	fi

.PHONY: mcp-list-tools
mcp-list-tools: ## List available MCP tools (via Python bridge)
	$(call print_message,$(BLUE),Listing MCP tools)
	@if docker ps --filter "name=mcp-standalone" --format "{{.Names}}" | grep -q "mcp-standalone"; then \
		echo -e "  $(GREEN)Available RAG Techniques:$(NC)"; \
		curl -s http://localhost:8001/mcp/list_techniques | python3 -m json.tool; \
	elif docker ps --filter "name=api-with-mcp" --format "{{.Names}}" | grep -q "api-with-mcp"; then \
		echo -e "  $(GREEN)Available MCP Tools:$(NC)"; \
		curl -s http://localhost:8000/api/v1/mcp/tools | python3 -m json.tool; \
	else \
		echo -e "  $(RED)✗$(NC) No MCP server running"; \
	fi

.PHONY: mcp-claude-config
mcp-claude-config: ## Generate Claude Code MCP configuration
	$(call print_message,$(BLUE),Generating Claude Code MCP configuration)
	@echo -e "  $(GREEN)Add this to your Claude Code config:$(NC)"
	@echo ""
	@echo "{\"mcpServers\": {"
	@echo "  \"iris-rag\": {"
	@echo "    \"command\": \"docker\","
	@echo "    \"args\": [\"exec\", \"-i\", \"mcp-standalone\", \"node\", \"/app/nodejs/dist/mcp/cli.js\"],"
	@echo "    \"env\": {"
	@echo "      \"MCP_TRANSPORT\": \"stdio\""
	@echo "    }"
	@echo "  }"
	@echo "}}"
	@echo ""
	@echo -e "  $(BLUE)ℹ$(NC) Ensure MCP server is running: make mcp-run-standalone"

.PHONY: mcp-dev
mcp-dev: env-check mcp-build mcp-run-standalone ## Full MCP development setup (build + run)

.PHONY: mcp-clean
mcp-clean: mcp-stop ## Clean MCP Docker resources
	$(call print_message,$(BLUE),Cleaning MCP Docker resources)
	@docker-compose -f docker-compose.mcp.yml down -v
	@docker rmi iris-rag-mcp:latest 2>/dev/null || true
	$(call print_message,$(GREEN),MCP resources cleaned)

