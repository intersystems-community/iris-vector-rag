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

.PHONY: setup
setup: ## Initial setup - copy environment file and create directories
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
docker-iris-shell: ## Open IRIS database shell
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
docker-backup: ## Backup database and volumes
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
test-enterprise-10k-real: docker-up docker-wait ## Run enterprise 10K testing with real database
	$(call print_message,$(BLUE),Running Enterprise 10K Testing with Real Database)
	@python tests/test_enterprise_10k_comprehensive.py --documents 10000
	$(call print_message,$(GREEN),Enterprise 10K real testing completed)

.PHONY: test-graphrag-scale
test-graphrag-scale: ## Run GraphRAG scale testing
	$(call print_message,$(BLUE),Running GraphRAG Scale Testing)
	@python scripts/test_graphrag_scale_10k.py --documents 1000 --use-mocks
	$(call print_message,$(GREEN),GraphRAG scale testing completed)

.PHONY: test-graphrag-scale-real
test-graphrag-scale-real: docker-up docker-wait ## Run GraphRAG scale testing with real database
	$(call print_message,$(BLUE),Running GraphRAG Scale Testing with Real Database)
	@python scripts/test_graphrag_scale_10k.py --documents 10000
	$(call print_message,$(GREEN),GraphRAG scale real testing completed)

.PHONY: test-pytest-enterprise
test-pytest-enterprise: ## Run enterprise tests via pytest
	$(call print_message,$(BLUE),Running Enterprise Tests via Pytest)
	@python -m pytest tests/test_enterprise_10k_comprehensive.py -v --tb=short
	$(call print_message,$(GREEN),Pytest enterprise testing completed)

.PHONY: test-all-enterprise
test-all-enterprise: test-enterprise-10k test-graphrag-scale test-pytest-enterprise ## Run all enterprise scale tests
	$(call print_message,$(GREEN),All enterprise scale tests completed)

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
docker-urls: ## Show service URLs
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