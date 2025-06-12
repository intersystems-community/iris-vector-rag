# RAG Templates Makefile
# Standardized commands for development, testing, and data management
# Uses conda environment 'iris_vector' for consistent dependency management

.PHONY: help install test test-unit test-integration test-e2e test-1000 test-ragas-1000-enhanced debug-ragas-hyde debug-ragas-graphrag debug-ragas-crag debug-ragas-colbert debug-ragas-basic debug-ragas-noderag debug-ragas-hybrid_ifind eval-all-ragas-1000 ragas-debug ragas-test ragas-full ragas-cache-check ragas-clean ragas-no-cache ragas clean setup-db load-data clear-rag-data validate-iris-rag validate-pipeline validate-all-pipelines auto-setup-pipeline auto-setup-all setup-env make-test-echo test-performance-ragas-tdd test-scalability-ragas-tdd test-tdd-comprehensive-ragas test-1000-enhanced test-tdd-ragas-quick ragas-with-tdd

# Simple test target to verify make execution
make-test-echo:
	@echo "--- Makefile echo test successful ---"

# Conda environment name
CONDA_ENV = iris_vector

# Conda run command for consistent environment usage
CONDA_RUN = conda run -n $(CONDA_ENV)

# Default target
help:
	@echo "RAG Templates - Available Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  make setup-env        - Set up conda environment (iris_vector)"
	@echo "  make install          - Install dependencies in conda environment"
	@echo "  make setup-db         - Initialize IRIS database schema"
	@echo ""
	@echo "Testing (DBAPI-first):"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-e2e         - Run end-to-end tests"
	@echo "  make test-1000        - Run comprehensive test with 1000 docs (legacy)"
	@echo "  make test-ragas-1000-enhanced  - Run RAGAs evaluation (original script) on all 7 pipelines with 1000 docs"
	@echo "  make eval-all-ragas-1000 - Run comprehensive RAGAs evaluation with full metrics (original script)"
	@echo "  make validate-iris-rag - Validate iris_rag package"
	@echo ""
	@echo "Lightweight RAGAs Testing:"
	@echo "  make ragas-debug      - Quick debug run (basic pipeline, core metrics, 3 queries)"
	@echo "  make ragas-test       - Standard test run (basic+hyde, extended metrics)"
	@echo "  make ragas-full       - Full evaluation (all pipelines, full metrics)"
	@echo "  make ragas-cache-check - Check cache status"
	@echo "  make ragas-clean      - Clear cache and run debug"
	@echo "  make ragas-no-cache   - Run without cache"
	@echo "  make ragas PIPELINES=basic,hyde METRICS=core - Parameterized run"
	@echo ""
	@echo "RAGAs Debug Testing (individual pipelines):"
	@echo "  make debug-ragas-basic      - Debug Basic RAG pipeline"
	@echo "  make debug-ragas-hyde       - Debug HyDE pipeline"
	@echo "  make debug-ragas-crag       - Debug CRAG pipeline"
	@echo "  make debug-ragas-colbert    - Debug ColBERT pipeline"
	@echo "  make debug-ragas-noderag    - Debug NodeRAG pipeline"
	@echo "  make debug-ragas-graphrag   - Debug GraphRAG pipeline"
	@echo "  make debug-ragas-hybrid_ifind - Debug Hybrid iFind pipeline"
	@echo ""
	@echo "TDD with RAGAS Testing (New):"
	@echo "  make test-performance-ragas-tdd - Run TDD performance benchmark tests with RAGAS quality metrics"
	@echo "  make test-scalability-ragas-tdd - Run TDD scalability tests with RAGAS across document scales"
	@echo "  make test-tdd-comprehensive-ragas - Run all TDD RAGAS tests (performance & scalability)"
	@echo "  make test-1000-enhanced   - Run TDD RAGAS tests with 1000+ documents for comprehensive validation"
	@echo "  make test-tdd-ragas-quick - Run a quick version of TDD RAGAS performance tests for development"
	@echo "  make ragas-with-tdd       - Run comprehensive TDD RAGAS tests and generate detailed report"
	@echo ""
	@echo "Validation & Auto-Setup:"
	@echo "  make validate-pipeline PIPELINE=<type> - Validate specific pipeline"
	@echo "  make validate-all-pipelines - Validate all 7 pipeline types"
	@echo "  make auto-setup-pipeline PIPELINE=<type> - Auto-setup pipeline with validation"
	@echo "  make auto-setup-all     - Auto-setup all pipelines with validation"
	@echo "  make test-with-auto-setup - Run tests with automatic setup"
	@echo ""
	@echo "Data Management:"
	@echo "  make load-data        - Load sample PMC documents (DBAPI)"
	@echo "  make load-1000        - Load 1000+ PMC documents for testing"
	@echo "  make check-data       - Check current document count"
	@echo "  make clear-rag-data   - Clear all rows from RAG document tables (DocumentChunks and SourceDocuments)"
	@echo ""
	@echo "Development:"
	@echo "  make clean            - Clean up temporary files"
	@echo "  make lint             - Run code linting"
	@echo "  make format           - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        - Start IRIS container"
	@echo "  make docker-down      - Stop IRIS container"
	@echo "  make docker-logs      - View IRIS container logs"
	@echo ""
	@echo "Environment Info:"
	@echo "  Current conda environment: $(CONDA_ENV)"
	@echo "  Use 'conda activate $(CONDA_ENV)' to activate manually"

# Environment setup
setup-env:
	@echo "Setting up conda environment: $(CONDA_ENV)"
	@if conda env list | grep -q "^$(CONDA_ENV) "; then \
		echo "✓ Environment $(CONDA_ENV) already exists"; \
	else \
		echo "Creating new environment $(CONDA_ENV)..."; \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
	fi
	@echo "Installing core dependencies..."
	$(CONDA_RUN) pip install pyyaml sentence-transformers transformers torch intersystems-irispython pytest

# Installation and setup
install: setup-env
	@echo "Installing all dependencies in conda environment: $(CONDA_ENV)"
	$(CONDA_RUN) pip install -r requirements.txt || $(CONDA_RUN) pip install intersystems-irispython pytest numpy pandas

setup-db:
	@echo "Setting up IRIS database schema (DBAPI)..."
	$(CONDA_RUN) python -c "from common.iris_connection_manager import test_connection; print('✓ Connection test passed' if test_connection() else '✗ Connection test failed')"
	$(CONDA_RUN) python common/db_init_with_indexes.py

# Testing commands (DBAPI-first)
test: test-unit test-integration

test-unit:
	@echo "Running unit tests..."
	$(CONDA_RUN) pytest tests/test_core/ tests/test_pipelines/ -v

test-integration:
	@echo "Running integration tests (DBAPI)..."
	$(CONDA_RUN) pytest tests/test_integration/ -v

test-e2e:
	@echo "Running end-to-end tests (DBAPI)..."
	$(CONDA_RUN) pytest tests/test_e2e_* -v

test-1000:
	@echo "Running comprehensive E2E test with 1000 PMC documents..."
	cd tests && $(CONDA_RUN) python test_comprehensive_e2e_iris_rag_1000_docs.py

test-ragas-1000-enhanced:
	@echo "Running RAGAs evaluation (original script) on all 7 pipelines with 1000 documents (verbose)..."
	@echo "This will evaluate: basic, hyde, crag, colbert, noderag, graphrag, hybrid_ifind"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines basic hyde crag colbert noderag graphrag hybrid_ifind --iterations 3

debug-ragas-hyde:
	@echo "Running debug RAGAs evaluation for HyDE pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test HyDE pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines hyde --iterations 1 --no-ragas

debug-ragas-graphrag:
	@echo "Running debug RAGAs evaluation for GraphRAG pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test GraphRAG pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines graphrag --iterations 1 --no-ragas

debug-ragas-crag:
	@echo "Running debug RAGAs evaluation for CRAG pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test CRAG pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines crag --iterations 1 --no-ragas

debug-ragas-colbert:
	@echo "Running debug RAGAs evaluation for ColBERT pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test ColBERT pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines colbert --iterations 1 --no-ragas

debug-ragas-basic:
	@echo "Running debug RAGAs evaluation for Basic pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test Basic pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines basic --iterations 1 --no-ragas

debug-ragas-noderag:
	@echo "Running debug RAGAs evaluation for NodeRAG pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test NodeRAG pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines noderag --iterations 1 --no-ragas

debug-ragas-hybrid_ifind:
	@echo "Running debug RAGAs evaluation for Hybrid iFind pipeline (no RAGAs metrics, 1 iteration)..."
	@echo "This will test Hybrid iFind pipeline execution and data readiness without RAGAs metric calculation"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines hybrid_ifind --iterations 1 --no-ragas

eval-all-ragas-1000:
	@echo "Running comprehensive RAGAs evaluation on all pipelines with 1000 documents..."
	@echo "This includes full RAGAs metrics calculation for all 7 pipeline types"
	eval "$$(conda shell.bash hook)" && conda activate iris_vector && python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines basic hyde crag colbert noderag graphrag hybrid_ifind --iterations 5 > comprehensive_ragas_results/eval_all_ragas_1000.out 2> comprehensive_ragas_results/eval_all_ragas_1000.err

validate-iris-rag:
	@echo "Validating iris_rag package..."
	$(CONDA_RUN) python -c "import iris_rag; print('✓ iris_rag package imported successfully')"
	$(CONDA_RUN) python -c "from iris_rag.pipelines.basic import BasicRAGPipeline; print('✓ BasicRAGPipeline imported')"
	$(CONDA_RUN) python -c "from iris_rag.pipelines.colbert import ColBERTRAGPipeline; print('✓ ColBERTRAGPipeline imported')"
	$(CONDA_RUN) python -c "from iris_rag.pipelines.crag import CRAGPipeline; print('✓ CRAGPipeline imported')"
	$(CONDA_RUN) python -c "from iris_rag.core.models import Document; d = Document(page_content='test'); print(f'✓ Document model works: {d.id}')"

# Data management (DBAPI-first)
load-data:
	@echo "Loading sample PMC documents using DBAPI..."
	$(CONDA_RUN) python -c "from data.loader import process_and_load_documents; result = process_and_load_documents('data/sample_10_docs', limit=10, use_mock=False); print(f'Loaded: {result}')"

load-1000:
	@echo "Loading 1000+ PMC documents for comprehensive testing..."
	$(CONDA_RUN) python -c "from data.loader import process_and_load_documents; result = process_and_load_documents('data/pmc_oas_downloaded', limit=1000, batch_size=50, use_mock=False); print(f'Loaded: {result}')"

check-data:
	@echo "Checking current document count in database..."
	$(CONDA_RUN) python -c "from common.iris_connection_manager import get_iris_connection; conn = get_iris_connection(); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments'); print(f'Total documents: {cursor.fetchone()[0]}'); cursor.close(); conn.close()"

clear-rag-data:
	@echo "Clearing RAG document tables..."
	$(CONDA_RUN) python -c "from common.iris_connection_manager import get_iris_connection; conn = get_iris_connection(); cursor = conn.cursor(); cursor.execute('DELETE FROM RAG.DocumentChunks'); print(f'{cursor.rowcount} rows deleted from RAG.DocumentChunks'); cursor.execute('DELETE FROM RAG.SourceDocuments'); conn.commit(); print(f'{cursor.rowcount} rows deleted from RAG.SourceDocuments'); cursor.close(); conn.close()"

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
	python -m flake8 iris_rag/ tests/ --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	python -m black iris_rag/ tests/ --line-length=120

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
	python -c "from common.iris_connection_manager import get_dbapi_connection; conn = get_dbapi_connection(); print('✓ DBAPI connection successful'); conn.close()"

test-jdbc:
	@echo "Testing JDBC connection (fallback)..."
	python -c "from common.iris_connection_manager import IRISConnectionManager; mgr = IRISConnectionManager(prefer_dbapi=False); conn = mgr.get_connection(); print(f'✓ {mgr.get_connection_type()} connection successful'); mgr.close()"

# Pipeline-specific validation with auto-setup
validate-pipeline:
	@if [ -z "$(PIPELINE)" ]; then \
		echo "Error: PIPELINE parameter required. Usage: make validate-pipeline PIPELINE=basic"; \
		echo "Available pipelines: basic, colbert, crag, hyde, graphrag, noderag, hybrid_ifind"; \
		exit 1; \
	fi
	@echo "Validating $(PIPELINE) pipeline with pre-condition checks..."
	@$(CONDA_RUN) python scripts/validate_pipeline.py validate $(PIPELINE)

auto-setup-pipeline:
	@if [ -z "$(PIPELINE)" ]; then \
		echo "Error: PIPELINE parameter required. Usage: make auto-setup-pipeline PIPELINE=basic"; \
		echo "Available pipelines: basic, colbert, crag, hyde, graphrag, noderag, hybrid_ifind"; \
		exit 1; \
	fi
	@echo "Auto-setting up $(PIPELINE) pipeline with validation and embedding generation..."
	@$(CONDA_RUN) python scripts/validate_pipeline.py setup $(PIPELINE)

# Demonstration targets
demo-validation:
	@echo "Running validation system demonstration..."
	@python scripts/demo_validation_system.py

validate-all-pipelines:
	@echo "Validating all 7 pipeline types..."
	@for pipeline in basic colbert crag hyde graphrag noderag hybrid_ifind; do \
		echo ""; \
		echo "=== Validating $$pipeline ==="; \
		$(MAKE) validate-pipeline PIPELINE=$$pipeline || echo "⚠ $$pipeline validation failed"; \
	done
	@echo ""
	@echo "=== ALL PIPELINE VALIDATION COMPLETE ==="

auto-setup-all:
	@echo "Auto-setting up all 7 pipeline types with validation..."
	@for pipeline in basic colbert crag hyde graphrag noderag hybrid_ifind; do \
		echo ""; \
		echo "=== Auto-setting up $$pipeline ==="; \
		$(MAKE) auto-setup-pipeline PIPELINE=$$pipeline || echo "⚠ $$pipeline auto-setup failed"; \
	done
	@echo ""
	@echo "=== ALL PIPELINE AUTO-SETUP COMPLETE ==="

# Enhanced comprehensive validation with auto-setup
validate-all: validate-iris-rag test-dbapi check-data validate-all-pipelines
	@echo ""
	@echo "=== COMPREHENSIVE VALIDATION COMPLETE ==="
	@echo "✓ iris_rag package validated"
	@echo "✓ DBAPI connection tested"
	@echo "✓ Database data checked"
	@echo "✓ All pipeline types validated"
	@echo ""
	@echo "System is ready for RAG operations!"

# Quick development setup with auto-setup
dev-setup: install setup-db load-data auto-setup-all validate-all
	@echo ""
	@echo "=== DEVELOPMENT ENVIRONMENT READY ==="
	@echo "✓ All pipelines auto-configured with validation"
	@echo "Run 'make test-1000' to execute comprehensive E2E validation"

# Self-healing test that auto-fixes issues
test-with-auto-setup:
	@echo "Running tests with automatic setup and validation..."
	@echo "Step 1: Auto-setup all pipelines"
	$(MAKE) auto-setup-all
	@echo ""
	@echo "Step 2: Validate all pipelines"
	$(MAKE) validate-all-pipelines
	@echo ""
	@echo "Step 3: Run comprehensive E2E test"
	$(MAKE) test-1000

# Production readiness check with auto-setup
prod-check: validate-iris-rag test-dbapi auto-setup-all
	@echo "Running production readiness checks with auto-setup..."
	python -c "from iris_rag import create_pipeline; print('✓ Pipeline factory works')"
	python -c "from common.iris_connection_manager import test_connection; assert test_connection(), 'Connection test failed'"
	@echo "Testing all pipeline types with auto-setup..."
	@for pipeline in basic colbert crag hyde graphrag noderag hybrid_ifind; do \
		echo "Testing $$pipeline pipeline..."; \
		python -c "import iris_rag; from common.utils import get_llm_func; from common.iris_connection_manager import get_iris_connection; pipeline = iris_rag.create_pipeline('$$pipeline', llm_func=get_llm_func(), external_connection=get_iris_connection(), auto_setup=True); result = pipeline.run('test query', top_k=3); print('✓ $$pipeline pipeline works: ' + str(len(result.get('retrieved_documents', []))) + ' docs retrieved')" || echo "⚠ $$pipeline pipeline test failed"; \
	done
	@echo "✓ Production readiness validated with auto-setup"

# Benchmark and performance
benchmark:
	@echo "Running performance benchmarks..."
	cd tests && python -m pytest test_comprehensive_e2e_iris_rag_1000_docs.py::test_comprehensive_e2e_all_rag_techniques_1000_docs -v

# Documentation
docs:
	@echo "Available documentation:"
	@echo "  - README.md - Project overview"
	@echo "  - docs/ - Detailed documentation"
	@echo "  - specs/ - Technical specifications"
	@echo "  - .clinerules - Development rules and standards"

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "IRIS_HOST: $(shell echo $$IRIS_HOST || echo 'localhost')"
	@echo "IRIS_PORT: $(shell echo $$IRIS_PORT || echo '1972')"
	@echo "IRIS_NAMESPACE: $(shell echo $$IRIS_NAMESPACE || echo 'USER')"

# Self-healing demonstration targets
demo-validation:
	@echo "=== DEMONSTRATING VALIDATION SYSTEM ==="
	@echo "This will show the pre-condition validation for all pipeline types..."
	$(MAKE) validate-all-pipelines

demo-auto-setup:
	@echo "=== DEMONSTRATING AUTO-SETUP SYSTEM ==="
	@echo "This will automatically fix any validation issues..."
	$(MAKE) auto-setup-all

demo-self-healing:
	@echo "=== DEMONSTRATING SELF-HEALING SYSTEM ==="
	@echo "This shows the complete validation -> auto-setup -> test cycle..."
	$(MAKE) test-with-auto-setup

# Quick pipeline testing
test-pipeline:
	@if [ -z "$(PIPELINE)" ]; then \
		echo "Error: PIPELINE parameter required. Usage: make test-pipeline PIPELINE=basic"; \
		echo "Available pipelines: basic, colbert, crag, hyde, graphrag, noderag, hybrid_ifind"; \
		exit 1; \
	fi
	@echo "Testing $(PIPELINE) pipeline with auto-setup..."
	$(MAKE) auto-setup-pipeline PIPELINE=$(PIPELINE)
	@echo "Running quick test for $(PIPELINE)..."
	@python -c "\
import iris_rag; \
from common.utils import get_llm_func; \
from common.iris_connection_manager import get_iris_connection; \
pipeline = iris_rag.create_pipeline('$(PIPELINE)', llm_func=get_llm_func(), external_connection=get_iris_connection(), auto_setup=True); \
result = pipeline.run('What are the effects of BRCA1 mutations?', top_k=3); \
print('✓ $(PIPELINE) pipeline test: ' + str(len(result.get('retrieved_documents', []))) + ' docs retrieved, answer length: ' + str(len(result.get('answer', ''))) + ' chars')"

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

# Self-healing data population targets
heal-data:
	@echo "=== SELF-HEALING DATA POPULATION ==="
	@echo "Running comprehensive self-healing cycle to achieve 100% table readiness..."
	$(CONDA_RUN) python scripts/data_population_manager.py populate --missing
	@echo ""
	@echo "=== SELF-HEALING COMPLETE ==="

check-readiness:
	@echo "=== CHECKING SYSTEM READINESS ==="
	@echo "Analyzing current table population status..."
	$(CONDA_RUN) python scripts/data_population_manager.py status --json
	@echo ""
	@echo "=== READINESS CHECK COMPLETE ==="

populate-missing:
	@echo "=== POPULATING MISSING TABLES ==="
	@echo "Identifying and populating missing table data..."
	$(CONDA_RUN) python scripts/data_population_manager.py populate --missing --json
	@echo ""
	@echo "=== POPULATION COMPLETE ==="

validate-healing:
	@echo "=== VALIDATING HEALING EFFECTIVENESS ==="
	@echo "Checking if self-healing achieved target readiness..."
	$(CONDA_RUN) python scripts/data_population_manager.py validate --target 100
	@echo ""
	@echo "=== VALIDATION COMPLETE ==="

auto-heal-all:
	@echo "=== COMPLETE SELF-HEALING WORKFLOW ==="
	@echo "Step 1: Check current readiness..."
	$(MAKE) check-readiness
	@echo ""
	@echo "Step 2: Populate missing data..."
	$(MAKE) populate-missing
	@echo ""
	@echo "Step 3: Validate healing effectiveness..."
	$(MAKE) validate-healing
	@echo ""
	@echo "=== AUTO-HEALING WORKFLOW COMPLETE ==="

heal-to-target:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET parameter required. Usage: make heal-to-target TARGET=85"; \
		echo "TARGET should be a percentage (e.g., 85 for 85% readiness)"; \
		exit 1; \
	fi
	@echo "=== HEALING TO TARGET $(TARGET)% READINESS ==="
	@echo "Running self-healing until $(TARGET)% table readiness is achieved..."
	$(CONDA_RUN) python rag_templates/validation/self_healing_orchestrator.py --target-readiness $(TARGET) --max-cycles 3
	@echo ""
	@echo "=== TARGET HEALING COMPLETE ==="

heal-progressive:
	@echo "=== PROGRESSIVE HEALING (INCREMENTAL) ==="
	@echo "Running incremental healing with dependency-aware ordering..."
	$(CONDA_RUN) python scripts/data_population_manager.py populate --missing --json
	@echo ""
	@echo "=== PROGRESSIVE HEALING COMPLETE ==="

heal-emergency:
	@echo "=== EMERGENCY HEALING (FORCE REPOPULATION) ==="
	@echo "WARNING: This will force repopulation of all tables!"
	@echo "Forcing complete data repopulation..."
	$(CONDA_RUN) python rag_templates/validation/self_healing_orchestrator.py --force-repopulation --max-cycles 5
	@echo ""
	@echo "=== EMERGENCY HEALING COMPLETE ==="

# Self-healing status and monitoring
heal-status:
	@echo "=== SELF-HEALING STATUS REPORT ==="
	$(CONDA_RUN) python scripts/table_status_detector.py --detailed --cache-ttl 0
	@echo ""
	@echo "=== STATUS REPORT COMPLETE ==="

heal-monitor:
	@echo "=== CONTINUOUS HEALING MONITOR ==="
	@echo "Monitoring system readiness and auto-healing as needed..."
	@echo "Press Ctrl+C to stop monitoring"
	$(CONDA_RUN) python rag_templates/validation/self_healing_orchestrator.py --monitor --interval 300
	@echo ""
	@echo "=== MONITORING STOPPED ==="

# Integration with existing targets
heal-and-test: heal-data test-1000
	@echo "=== HEAL AND TEST COMPLETE ==="
	@echo "✓ Data healing completed"
	@echo "✓ Comprehensive testing completed"

heal-and-validate: heal-data validate-all
	@echo "=== HEAL AND VALIDATE COMPLETE ==="
	@echo "✓ Data healing completed"
	@echo "✓ System validation completed"

# Quick healing shortcuts
quick-heal:
	@echo "=== QUICK HEALING (ESSENTIAL TABLES ONLY) ==="
	$(CONDA_RUN) python scripts/data_population_manager.py populate --missing --json
	@echo ""
	@echo "=== QUICK HEALING COMPLETE ==="

deep-heal:
	@echo "=== DEEP HEALING (ALL TABLES + OPTIMIZATION) ==="
	$(CONDA_RUN) python rag_templates/validation/self_healing_orchestrator.py --deep-healing --optimize-tables
	@echo ""
	@echo "=== DEEP HEALING COMPLETE ==="

# Lightweight RAGAs Testing Targets
ragas-debug:
	@echo "--- Starting make ragas-debug target ---"
	@echo "=== LIGHTWEIGHT RAGAS DEBUG RUN ==="
	@echo "Running quick debug with basic pipeline, core metrics, 3 queries"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --pipelines basic --metrics-level core --max-queries 3 --verbose

ragas-test:
	@echo "=== LIGHTWEIGHT RAGAS TEST RUN ==="
	@echo "Running standard test with basic+hyde pipelines, extended metrics"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --pipelines basic hyde --metrics-level extended --verbose

ragas-full:
	@echo "=== LIGHTWEIGHT RAGAS FULL EVALUATION ==="
	@echo "Running full evaluation with all pipelines, full metrics"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --pipelines basic hyde crag colbert noderag graphrag hybrid_ifind --metrics-level full --verbose

ragas-cache-check:
	@echo "=== RAGAS CACHE STATUS CHECK ==="
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --cache-check

ragas-clean:
	@echo "=== RAGAS CLEAN RUN (CLEAR CACHE + DEBUG) ==="
	@echo "Clearing cache and running debug evaluation"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --clear-cache --pipelines basic --metrics-level core --max-queries 3 --verbose

ragas-no-cache:
	@echo "=== RAGAS NO-CACHE RUN ==="
	@echo "Running evaluation without cache"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py --no-cache --pipelines basic --metrics-level core --max-queries 5 --verbose

# Parameterized RAGAs target
ragas:
	@if [ -z "$(PIPELINES)" ]; then \
		echo "Usage: make ragas PIPELINES=basic,hyde [METRICS=core] [QUERIES=10]"; \
		echo "Available pipelines: basic, hyde, crag, colbert, noderag, graphrag, hybrid_ifind"; \
		echo "Available metrics: core, extended, full"; \
		exit 1; \
	fi
	@echo "=== PARAMETERIZED RAGAS EVALUATION ==="
	@echo "Pipelines: $(PIPELINES)"
	@echo "Metrics: $(or $(METRICS),core)"
	@echo "Max Queries: $(or $(QUERIES),all)"
	eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && python eval/run_ragas.py \
		--pipelines $(shell echo "$(PIPELINES)" | tr ',' ' ') \
		--metrics-level $(or $(METRICS),core) \
		$(if $(QUERIES),--max-queries $(QUERIES),) \
		--verbose

# TDD with RAGAS Testing
# These targets leverage the comprehensive TDD+RAGAS integration in tests/test_tdd_performance_with_ragas.py
# They provide performance benchmarking with RAGAS quality metrics and scalability testing

# Run TDD performance benchmark tests with RAGAS quality metrics
# Tests pipeline performance while measuring RAGAS metrics (answer relevancy, context precision, faithfulness, context recall)
# Uses pytest marker: performance_ragas
test-performance-ragas-tdd:
	@echo "=== Running TDD Performance Benchmark Tests with RAGAS ==="
	@echo "This validates pipeline performance and RAGAS quality metrics meet minimum thresholds"
	$(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py -m performance_ragas -v

# Run TDD scalability tests with RAGAS across different document corpus sizes
# Tests how performance and quality metrics change as document count increases
# Uses pytest marker: scalability_ragas
test-scalability-ragas-tdd:
	@echo "=== Running TDD Scalability Tests with RAGAS ==="
	@echo "This tests performance and quality scaling across different document corpus sizes"
	$(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py -m scalability_ragas -v

# Run all TDD RAGAS integration tests (both performance and scalability)
# Comprehensive test suite covering all TDD+RAGAS integration aspects
# Uses pytest marker: ragas_integration
test-tdd-comprehensive-ragas:
	@echo "=== Running All TDD RAGAS Integration Tests (Performance & Scalability) ==="
	@echo "This runs the complete TDD+RAGAS test suite with comprehensive validation"
	$(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py -m ragas_integration -v

# Run TDD RAGAS tests with 1000+ documents for comprehensive validation
# Sets TEST_DOCUMENT_COUNT environment variable to ensure large-scale testing
# Requires iris_with_pmc_data fixture to respect the document count setting
test-1000-enhanced:
	@echo "=== Running TDD RAGAS Tests with 1000 Documents ==="
	@echo "This ensures comprehensive testing with large document corpus"
	@echo "Ensure TEST_DOCUMENT_COUNT is respected by iris_with_pmc_data fixture in conftest.py"
	TEST_DOCUMENT_COUNT=1000 $(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py -m ragas_integration -v

# Run a quick version of TDD RAGAS performance tests for development
# Uses TDD_RAGAS_QUICK_MODE environment variable to limit test scope
# Ideal for rapid development feedback cycles
test-tdd-ragas-quick:
	@echo "=== Running Quick TDD RAGAS Performance Test ==="
	@echo "This runs a limited test set for rapid development feedback"
	@echo "Uses TDD_RAGAS_QUICK_MODE environment variable to limit scope"
	TDD_RAGAS_QUICK_MODE=true $(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py -m performance_ragas -v
	# Example for running a specific test:
	# $(CONDA_RUN) pytest tests/test_tdd_performance_with_ragas.py::TestPerformanceBenchmarkingWithRagas::test_complete_pipeline_performance_with_ragas -v

# Run comprehensive TDD RAGAS tests and generate detailed performance report
# First runs all TDD+RAGAS tests, then generates a comprehensive Markdown report
# Report includes performance analysis, RAGAS metrics, scalability trends, and recommendations
ragas-with-tdd: test-tdd-comprehensive-ragas
	@echo "=== Generating TDD RAGAS Performance Report ==="
	@echo "Searching for latest test results to generate comprehensive report"
	@LATEST_JSON=$$(ls -t comprehensive_ragas_results/raw_data/test_performance_ragas_results_*.json 2>/dev/null | head -n 1); \
	if [ -f "$$LATEST_JSON" ]; then \
		echo "Found results file: $$LATEST_JSON"; \
		echo "Generating comprehensive TDD+RAGAS performance report..."; \
		$(CONDA_RUN) python scripts/generate_tdd_ragas_performance_report.py "$$LATEST_JSON"; \
		echo "Report generated in reports/tdd_ragas_reports/ directory"; \
	else \
		echo "Warning: No TDD RAGAS JSON result file found in comprehensive_ragas_results/raw_data/"; \
		echo "Expected pattern: test_performance_ragas_results_*.json"; \
		echo "Run 'make test-tdd-comprehensive-ragas' first to generate test results"; \
	fi
