# RAG Templates for InterSystems IRIS

This repository contains implementation templates for various Retrieval Augmented Generation (RAG) techniques using InterSystems IRIS.

---
**Navigate the Documentation**

For a comprehensive guide to all project documents, including setup, technical deep-dives, IRIS issue analyses, RAG technique implementations, testing, and benchmarking, please start with the:
### **[📚 Complete Documentation Index](docs/README.md)**

The documentation is now organized into logical categories:
- **🏗️ Implementation**: Technical implementation details for all RAG techniques
- **✅ Validation**: Enterprise validation reports and testing results
- **🚀 Deployment**: Production deployment guides and operational docs
- **🔧 Fixes**: Technical fixes and troubleshooting documentation
- **📊 Summaries**: High-level project summaries and status reports
---
## 🧭 Project Orientation & Key Documents

To quickly understand the current state, goals, and ongoing development of this RAG Templates project, please refer to the following key documents:

1.  **This `README.md`**: Provides a general overview, setup instructions, and links to other important resources.
2.  **[`PROJECT_STATUS_DASHBOARD.md`](PROJECT_STATUS_DASHBOARD.md:1)**: This is the **single source of truth** for the operational status of each RAG technique, known issues, and the progress of the overall project restructuring and improvement plan.
3.  **[`docs/COMPREHENSIVE_SCALING_EVALUATION_FRAMEWORK.md`](docs/COMPREHENSIVE_SCALING_EVALUATION_FRAMEWORK.md:1)**: Details the new, extensive framework for testing and evaluating RAG pipelines across various data scales, which is a major focus of recent and ongoing work.
4.  **[`tests/test_e2e_rag_pipelines.py`](tests/test_e2e_rag_pipelines.py:1)**: This script contains the comprehensive end-to-end tests for all core RAG pipelines. It demonstrates how pipelines are currently validated, including RAGAS integration for quality assessment and detailed CSV logging of results.
5.  **[`test_results/rag_evaluation_log.csv`](test_results/rag_evaluation_log.csv:1)**: (Generated after running tests via `pytest tests/test_e2e_rag_pipelines.py`) This CSV file will contain the most up-to-date, detailed results from individual pipeline evaluations, including parameters used, RAGAS scores, and pass/fail status.

Reviewing these documents will provide a solid understanding of the project's current trajectory and how its components are being managed and validated.

## Project Status (As of June 1, 2025)

🚨 **IMPORTANT: For the most accurate and up-to-date status of each RAG technique, please refer to the [PROJECT_STATUS_DASHBOARD.md](PROJECT_STATUS_DASHBOARD.md).** 🚨

**Note on Historical Status Reporting:** Previous versions of this README and other status documents contained inaccuracies regarding the operational status and success rates of various RAG techniques. The [`PROJECT_STATUS_DASHBOARD.md`](PROJECT_STATUS_DASHBOARD.md:1) and its linked `project_status_logs/` provide the corrected, evidence-based status.

**Key Recent Updates:**
- ✅ **ColBERT Operational & Verified (June 1, 2025)**:
    - Token-level retrieval with 937K+ token embeddings (using `fjmgAI/reason-colBERT-150M-GTE-ModernColBERT` for 768-dim embeddings).
    - Successfully fixed issues in `scripts/populate_colbert_token_embeddings_native_vector.py` (interactive loop, SQL stream field handling).
    - Corrected `colbert_query_encoder` fixture in `tests/conftest.py` for dimension and format consistency.
    - `test_colbert_with_real_data` E2E test **passing** with 100 documents.
    - See [`COLBERT_PIPELINE_FIX_SUMMARY.md`](COLBERT_PIPELINE_FIX_SUMMARY.md) for details.
    - Next Step: Scale ColBERT E2E testing to 1000 documents.
- **JDBC Migration**: Believed to have solved critical vector parameter binding issues. See [`docs/JDBC_V2_MIGRATION_COMPLETE.md`](docs/JDBC_V2_MIGRATION_COMPLETE.md). However, some techniques like BasicRAG still exhibit vector-related errors (see dashboard).
- **Data Scale**: Significant data (99,990 documents, 937K+ token embeddings, 273K+ entities) has been processed.

**For detailed status on individual RAG techniques, including BasicRAG, CRAG, GraphRAG, HyDE, HybridIFindRAG, and NodeRAG, please consult the [PROJECT_STATUS_DASHBOARD.md](PROJECT_STATUS_DASHBOARD.md).**

**📊 Key Supporting Documents for Recent Changes**:
- ColBERT Fixes: See [`COLBERT_PIPELINE_FIX_SUMMARY.md`](COLBERT_PIPELINE_FIX_SUMMARY.md)
- JDBC Solution: See [`docs/JDBC_V2_MIGRATION_COMPLETE.md`](docs/JDBC_V2_MIGRATION_COMPLETE.md)
- For historical (potentially outdated) evaluations: See `FINAL_COLBERT_RECOVERY_AND_7_TECHNIQUE_EVALUATION_SUMMARY.md` (Note: cross-reference with dashboard for actual current status).

The project uses a proven local development setup:
- **Python Environment:** Managed on the host machine using `uv` with dependencies defined in `pyproject.toml`
- **InterSystems IRIS Database:** Runs in a dedicated Docker container via `docker-compose.iris-only.yml`
- **Database Interaction:** JDBC connection for safe parameter binding with vector functions
- **Vector Storage:** VARCHAR columns with TO_VECTOR() conversion and HNSW indexes for performance

**🔧 JDBC Breakthrough**: The migration from ODBC to JDBC solved critical parameter binding issues, enabling safe vector queries with prepared statements. This eliminates SQL injection risks and enables full use of V2 tables with HNSW indexes. See [`jdbc_exploration/JDBC_SOLUTION_SUMMARY.md`](jdbc_exploration/JDBC_SOLUTION_SUMMARY.md) for technical details.

**🚀 Performance Optimization Achievement:** A major performance breakthrough was achieved by systematically diagnosing and resolving severe ingestion degradation through strategic database index optimization. See [INGESTION_PERFORMANCE_OPTIMIZATION.md](docs/INGESTION_PERFORMANCE_OPTIMIZATION.md) for complete details on the 1.6x-2.6x speedup achieved.

## 🔧 JDBC Setup for Vector Operations

To use the JDBC-based vector search (required for safe parameter binding):

1. **Download JDBC Driver**:
   ```bash
   curl -L -o intersystems-jdbc-3.8.4.jar \
     https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar
   ```

2. **Install Dependencies**:
   ```bash
   pip install jaydebeapi jpype1
   ```

3. **Update Connection Code**:
   ```python
   # Use JDBC connector instead of ODBC
   from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection
   conn = get_iris_jdbc_connection()
   ```

For detailed migration instructions, see [`jdbc_exploration/JDBC_MIGRATION_PLAN.md`](jdbc_exploration/JDBC_MIGRATION_PLAN.md).

## RAG Techniques Implemented

This repository contains implementations for various RAG techniques.
**For the current operational status, known issues, and last successful test date for each technique, please refer to the [PROJECT_STATUS_DASHBOARD.md](PROJECT_STATUS_DASHBOARD.md).**

The dashboard provides an accurate, evidence-based overview, superseding any generalized claims found in older documentation.

Key techniques explored include:
- BasicRAG
- ColBERT (Recently fixed and verified with 100 documents as of June 1, 2025)
- CRAG
- GraphRAG
- HyDE
- HybridIFindRAG
- NodeRAG

### Enterprise Features:
- **Real Data Processing**: 1,825+ authentic PMC biomedical articles
- **Enhanced Chunking**: 4 strategies (Recursive, Semantic, Adaptive, Hybrid)
- **Native IRIS Integration**: ObjectScript classes and vector search optimization
- **Parallel Pipeline**: Simultaneous download and ingestion at enterprise scale
- **Production Monitoring**: Real-time system health and performance tracking

## Features

### Core RAG Implementation
- Implementations for 7 RAG techniques using Python and InterSystems IRIS. (See [PROJECT_STATUS_DASHBOARD.md](PROJECT_STATUS_DASHBOARD.md) for current operational status of each).
- ✅ **ColBERT token-level retrieval** with 937K+ token embeddings operational (as of June 1, 2025, verified with 100 docs).
- ✅ **Enhanced chunking system** with 4 strategies (Recursive, Semantic, Adaptive, Hybrid).
- ✅ **Hybrid iFind RAG** implemented (see dashboard for current status).
- Strives for real semantic search with meaningful similarity scores.

### Enterprise Validation
- ✅ **Comprehensive Test-Driven Development (TDD)** approach
- ✅ **1000+ real PMC medical documents** validated with embeddings
- ✅ **Enterprise-scale testing** up to 50,000 documents
- ✅ **Performance benchmarking** framework ready for full LLM integration
- ✅ **Production-ready architecture** with comprehensive error handling and monitoring

### Technical Excellence
- ✅ **Scalable architecture** with HNSW indexing for Enterprise Edition
- ✅ **Zero external dependencies** for chunking (no LangChain/TikToken)
- ✅ **Biomedical optimization** with 95%+ token accuracy
- ✅ **Native IRIS integration** through ObjectScript and vector search
- ✅ **ObjectScript RAG Framework** - SQL-callable RAG methods via Embedded Python ([`objectscript/`](objectscript/))
- ✅ **Comprehensive documentation** with deployment guides

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` (Python package installer and virtual environment manager). Installation: `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`.
- Poetry (Optional, but recommended for a one-time export of `requirements.txt` if `uv` has trouble with direct `pyproject.toml` dependency installation for Poetry projects. If your Poetry version is < 1.1, you might need to upgrade it for the `export` command).
- InterSystems IRIS 2025.1+ (Community Edition or licensed).
- Docker (and Docker Compose) for running the IRIS database container.
- At least 2GB of free disk space for PMC data and database files

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd rag-templates
    ```

2.  **Set up Python Environment with `uv`:**
    *   Ensure Python 3.11+ is installed on your host.
    *   Install `uv` if you haven't already:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # OR
        # pip install uv
        ```
    *   Create a virtual environment (e.g., named `.venv`):
        ```bash
        uv venv .venv --python python3.11 # Or specify your Python 3.11+ executable
        ```
    *   Activate the virtual environment:
        ```bash
        source .venv/bin/activate
        ```
        (Your shell prompt should change to indicate the venv is active, e.g., `(.venv)`).

3.  **Install Python Dependencies:**
    *   **Recommended Method (using Poetry for export):** If you have Poetry (version 1.1+ recommended for the `export` command), run this once:
        ```bash
        # Ensure your .venv is active
        poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev
        uv pip install -r requirements.txt
        ```
        The `--with dev` flag includes development dependencies like `pytest`.
    *   **Alternative (if `uv` direct processing works or Poetry export is problematic):**
        Temporarily comment out the `[tool.poetry.scripts]` section in `pyproject.toml` if you encounter "invalid console script" errors with `uv pip install .`.
        ```bash
        # Ensure your .venv is active
        uv pip install . # Installs main dependencies
        # To install development dependencies (like pytest, ruff, black, mypy):
        # You might need to list them explicitly or add them to a dev requirements file
        # e.g., uv pip install pytest pytest-cov pytest-mock ruff black mypy
        # (Refer to [tool.poetry.group.dev.dependencies] in pyproject.toml for the full list)
        ```

4.  **Set up and Start IRIS Docker Container:**
    *   The IRIS database runs in a dedicated Docker container.
    *   Start the IRIS container:
        ```bash
        docker-compose -f docker-compose.iris-only.yml up -d
        ```
    *   Verify it's running: Check `docker ps` or access the IRIS Management Portal (default: `http://localhost:52773`, username `SuperUser`, password `SYS`).
        *Note: If port 1972 or 52773 is in use on your host, adjust the port mappings in `docker-compose.iris-only.yml` and update connection environment variables (`IRIS_PORT`, `IRIS_WEB_PORT`) if necessary.*

5.  **Initialize Database Schema:**
    *   Ensure your `.venv` is active.
    *   Run the database initialization script:
        ```bash
        python run_db_init_local.py --force-recreate
        ```

6.  **Load Real PMC Data:**
    *   Ensure your `.venv` is active.
    *   Download and process PMC articles into the IRIS database:
        ```bash
        # Load at least 1000 documents for proper testing
        python scripts_to_review/load_pmc_data.py --limit 1100 --load-colbert
        ```
        This script will:
        - Download PMC articles to `data/pmc_oas_downloaded/`
        - Process and load them into the IRIS database
        - Generate embeddings for all documents (Note: This step is CURRENTLY BLOCKED due to ODBC driver limitations with `TO_VECTOR` when loading embeddings. Text data can be loaded, but vector searches on this newly loaded data will not work until this is resolved.)
        - Prepare token-level embeddings for ColBERT (Also subject to the embedding load blocker)

    *   Verify data was loaded correctly:
        ```bash
        # This will check if at least 1000 documents are available
        python scripts/run_e2e_tests.py --skip-docker-check --min-docs 1000
        ```

## Testing and Benchmarking

We've developed comprehensive testing and benchmarking frameworks to ensure all RAG techniques work correctly with real data and to compare their performance.
**Note:** Full end-to-end testing and benchmarking requiring vector embeddings from newly loaded real data are currently **blocked** by the ODBC/`TO_VECTOR` issue detailed in the "IRIS SQL Vector Operations Limitations" section. Tests on text-based components or with pre-existing/mocked embeddings may still function.

### Running End-to-End Tests

The `scripts/run_e2e_tests.py` script automates end-to-end testing with real PMC data:

```bash
# Run all end-to-end tests with at least 1000 documents
python scripts/run_e2e_tests.py --min-docs 1000 --output-dir test_results

# Run a specific test with verbose output
python scripts/run_e2e_tests.py --test test_basic_rag_with_real_data --verbose

# Skip Docker container checks (if you've already verified it's running)
python scripts/run_e2e_tests.py --skip-docker-check
```

This script:
1. Checks if the IRIS Docker container is running and starts it if needed
2. Verifies the database has been initialized with real PMC data (at least 1000 documents)
3. Runs the end-to-end tests with pytest
4. Generates test reports in both JSON and HTML formats

### Running Benchmarks

The `scripts/run_rag_benchmarks.py` script executes benchmarks for all RAG techniques:

```bash
# Run benchmarks for all techniques with default settings
python scripts/run_rag_benchmarks.py

# Run benchmarks for specific techniques
python scripts/run_rag_benchmarks.py --techniques basic_rag hyde colbert

# Run benchmarks with a specific dataset and number of queries
python scripts/run_rag_benchmarks.py --dataset medical --num-queries 20
```

This script:
1. Runs each RAG technique against a set of test queries
2. Measures retrieval quality, answer quality, and performance metrics
3. Generates comparative visualizations (radar charts, bar charts)
4. Creates detailed benchmark reports in the `benchmark_results` directory

### Understanding Test Reports and Benchmark Results

- **Test Reports**: Located in the `test_results` directory, these include:
  - JSON reports with detailed test results and metrics
  - HTML reports with interactive test summaries
  - Logs of test execution and any errors encountered

- **Benchmark Results**: Located in the `benchmark_results` directory, these include:
  - JSON files with raw benchmark data
  - Markdown reports with analysis and comparisons
  - Visualizations comparing techniques across different metrics
  - Performance metrics (throughput, latency percentiles)

### Running Basic Tests

For simpler unit testing during development:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run tests for a specific pipeline
pytest tests/test_basic_rag.py
pytest tests/test_hyde.py
pytest tests/test_crag.py
pytest tests/test_colbert.py
pytest tests/test_noderag.py
pytest tests/test_graphrag.py

# Run all unit tests
pytest tests/
```

## Key Project Documents

While the **[Project Documentation Index](docs/INDEX.md)** provides a comprehensive list, some key starting points include:

- **[`PLAN_STATUS.md`](PLAN_STATUS.md)**: Detailed project status and task breakdown.
- **[`docs/MANAGEMENT_SUMMARY.md`](docs/MANAGEMENT_SUMMARY.md)**: High-level summary for project managers, including JIRA issue suggestions.
- **[`docs/DEVELOPMENT_STRATEGY_EVOLUTION.md`](docs/DEVELOPMENT_STRATEGY_EVOLUTION.md)**: Explains the evolution of the project's development approach and the pivot to client-side SQL.
- **[`docs/TESTING.md`](docs/TESTING.md)**: The primary guide for all testing procedures.

## IRIS SQL Vector Operations Limitations

### Current Known Limitations

InterSystems IRIS vector search capabilities are essential for RAG pipelines, but some limitations in the SQL implementation may affect standard parameterized queries with vector operations:

1. **TOP K Parameter Limitation**: The `TOP` clause in vector similarity searches may not accept parameter markers in all contexts, requiring careful handling of result limits.

2. **Vector Query Optimization**: Complex vector queries may require specific syntax patterns for optimal performance.

These limitations are addressed through our implementation in the [`common/vector_sql_utils.py`](common/vector_sql_utils.py) module that provides:

- Strict validation of vector strings and top-k values
- Safe query construction with security checks
- Helper functions to construct and execute vector search queries
- Production-tested patterns for reliable vector operations

Our current implementation has been validated with real PMC data at enterprise scale (50,000+ documents) and provides reliable vector search capabilities.

For detailed information about these limitations, our investigation findings, and recommended solutions, see:
- [IRIS_SQL_VECTOR_LIMITATIONS.md](docs/IRIS_SQL_VECTOR_LIMITATIONS.md) - Details ongoing investigations and current understanding.
- [VECTOR_SEARCH_SYNTAX_FINDINGS.md](docs/VECTOR_SEARCH_SYNTAX_FINDINGS.md) - **Key document for correct `TO_VECTOR` syntax in IRIS 2025.1.**
- [VECTOR_SEARCH_TECHNICAL_DETAILS.md](docs/VECTOR_SEARCH_TECHNICAL_DETAILS.md) - Technical specifics of client libraries and driver behavior.
- [VECTOR_SEARCH_ALTERNATIVES.md](docs/VECTOR_SEARCH_ALTERNATIVES.md) - Evaluation of different vector search strategies.
- [HNSW_INDEXING_RECOMMENDATIONS.md](docs/HNSW_INDEXING_RECOMMENDATIONS.md) - Best practices for HNSW indexing.

## Project Structure

A high-level overview of the project structure:
```
rag-templates/
├── .gitignore
├── README.md
├── PLAN_STATUS.md
├── pyproject.toml               # Project dependencies and metadata
├── docker-compose.iris-only.yml # Docker compose for the dedicated IRIS container
├── run_db_init_local.py         # Script to initialize local DB schema
|
├── basic_rag/                   # Basic RAG pipeline
├── colbert/                     # ColBERT pipeline
├── common/                      # Shared utilities
│   ├── iris_connector.py
│   └── vector_sql_utils.py      # Workarounds for IRIS SQL vector operations
├── config/
│   └── odbc/                    # ODBC configuration files
│       ├── odbc.ini
│       └── odbcinst_docker.ini
├── crag/                        # CRAG pipeline
├── data/                        # Data loading scripts and raw data (e.g., pmc_oas_downloaded/)
├── docs/                        # Project documentation
│   └── INDEX.md                 # Main documentation index
├── eval/                        # Evaluation and benchmarking scripts
├── graphrag/                    # GraphRAG pipeline
├── hyde/                        # HyDE pipeline
├── noderag/                     # NodeRAG pipeline
├── objectscript/                # ObjectScript RAG Framework
│   ├── RAGDemo.Invoker.cls      # SQL-callable RAG methods
│   ├── RAGDemo.KeywordFinder.cls # Keyword processing utilities
│   ├── RAGDemo.KeywordProcessor.cls # Advanced keyword analysis
│   ├── RAGDemo.TestBed.cls      # Testing framework
│   └── python_bridge.py        # Python-ObjectScript bridge
├── scripts/                     # Utility and execution scripts
│   ├── performance/             # Performance optimization scripts
│   ├── validation/              # Validation and testing scripts
│   ├── migration/               # Database migration scripts
│   ├── run_e2e_tests.py
│   └── run_rag_benchmarks.py
├── tests/                       # Pytest test suite
│   ├── conftest.py
│   └── test_e2e_rag_pipelines.py
├── bug_reproductions/           # Bug reproduction scripts for IRIS issues
│   ├── README.md                # Documentation of known IRIS bugs
│   ├── bug1_parameter_binding.py
│   ├── bug2_hnsw_varchar.py
│   ├── bug3_vector_driver_support.py
│   └── bug4_stored_procedures.py
└── ... (other RAG pipeline directories, configuration files, etc.)
```

## Test-Driven Development

This project follows strict TDD principles:

1. **Test-First Development**: All features start with failing tests
2. **Red-Green-Refactor**: Write failing test, implement minimum code to pass, refactor
3. **Real End-to-End Tests**: Tests verify that RAG techniques actually work with real data
4. **Complete Pipeline Testing**: Test full pipeline from data ingestion to answer generation
5. **Assert Actual Results**: Tests make assertions on actual result properties

## Running the Demo Scripts

Each RAG technique has a demo script in its respective directory (e.g., `basic_rag/pipeline.py` has a `if __name__ == '__main__':` block for demo).

To run a demo:
1.  Ensure your `.venv` is activated (`source .venv/bin/activate`).
2.  Ensure IRIS Docker container is running and the database is initialized. (Note: Loading real document *embeddings* is currently blocked. Demos requiring vector search on newly loaded data will be affected. Demos might work with text-only features or pre-existing/mocked data if applicable.)
3.  Run the desired pipeline script directly:
    ```bash
    python basic_rag/pipeline.py
    python hyde/pipeline.py
    python colbert/pipeline.py
    python crag/pipeline.py
    python noderag/pipeline.py
    python graphrag/pipeline.py
    ```

For a more comprehensive demonstration that compares all techniques:
```bash
# Run a benchmark with a small number of queries for quick comparison
python scripts/run_rag_benchmarks.py --num-queries 3 --output-dir benchmark_results/quick_demo
```

## 🗺️ Roadmap & Future Enhancements

### 🚀 Immediate Priorities (Next Sprint)

#### **ColBERT Optimization**
- **Content Volume Management**: Implement intelligent content limiting for ColBERT to prevent LLM context overflow
- **Performance Tuning**: Optimize ColBERT retrieval speed (currently 5000+ seconds for complex queries)
- **Production Integration**: Add ColBERT to standard RAGAS evaluation pipeline with proper constraints

#### **Advanced RAG Techniques**
- **RAG-Fusion**: Multi-query fusion for enhanced retrieval coverage
- **Self-RAG**: Self-reflective RAG with answer quality assessment
- **Adaptive RAG**: Dynamic technique selection based on query complexity

### 📈 Medium-Term Goals (Next Quarter)

#### **Performance & Scalability**
- **Parallel Processing**: Multi-threaded document retrieval for faster response times
- **Caching Layer**: Intelligent caching for frequently accessed embeddings and results
- **Load Balancing**: Distribute queries across multiple IRIS instances for enterprise scale

#### **Quality Improvements**
- **Advanced Chunking**: Implement document-aware chunking with section boundaries
- **Embedding Optimization**: Fine-tune embeddings for medical domain specificity
- **Answer Validation**: Automated fact-checking and confidence scoring

#### **Enterprise Features**
- **Multi-Tenant Support**: Isolated data and configurations per organization
- **Audit Logging**: Comprehensive query and response tracking for compliance
- **API Gateway**: RESTful API with authentication and rate limiting

### 🔬 Research & Innovation (Long-Term)

#### **Next-Generation RAG**
- **Multimodal RAG**: Integration with image and table processing for comprehensive medical documents
- **Temporal RAG**: Time-aware retrieval considering document publication dates and medical guideline updates
- **Federated RAG**: Cross-institutional knowledge sharing while maintaining privacy

#### **AI/ML Enhancements**
- **Custom LLM Fine-Tuning**: Domain-specific language models for medical terminology
- **Reinforcement Learning**: Query optimization based on user feedback and success metrics
- **Automated Evaluation**: Self-improving systems with continuous quality assessment

#### **Integration & Ecosystem**
- **FHIR Integration**: Direct integration with healthcare data standards
- **Clinical Decision Support**: Integration with electronic health record systems
- **Real-Time Updates**: Live document ingestion and index updates

### 🛠️ Technical Debt & Maintenance

#### **Code Quality**
- **Type Safety**: Complete migration to typed Python with mypy validation
- **Test Coverage**: Achieve 95%+ test coverage across all RAG techniques
- **Documentation**: Interactive API documentation with live examples

#### **Infrastructure**
- **Container Orchestration**: Kubernetes deployment configurations
- **Monitoring**: Comprehensive observability with metrics, logs, and traces
- **Security**: Vulnerability scanning and security best practices implementation

#### **Database Optimization**
- **Index Tuning**: Advanced HNSW index optimization for specific use cases
- **Query Optimization**: SQL query performance analysis and improvement
- **Storage Efficiency**: Compression and archival strategies for large datasets

### 📊 Evaluation & Benchmarking

#### **Comprehensive Benchmarks**
- **Domain-Specific Datasets**: Evaluation against medical, legal, and technical document collections
- **Multilingual Support**: RAG performance across different languages
- **Comparative Analysis**: Regular benchmarking against latest research and commercial solutions

#### **Quality Metrics**
- **Human Evaluation**: Expert assessment of answer quality and relevance
- **Automated Metrics**: Expanded RAGAS evaluation with custom medical domain metrics
- **Performance Tracking**: Continuous monitoring of response times and resource usage

### 🎯 Success Metrics

#### **Performance Targets**
- **Response Time**: Sub-second response for 90% of queries (currently achieved by GraphRAG)
- **Quality Score**: RAGAS scores above 0.8 for all techniques
- **Scalability**: Support for 1M+ documents with linear performance scaling

#### **Business Objectives**
- **Production Adoption**: Deployment in 5+ enterprise environments
- **User Satisfaction**: 95%+ user satisfaction in production deployments
- **Cost Efficiency**: 50% reduction in infrastructure costs through optimization

---

**📝 Note**: This roadmap is actively maintained and updated based on user feedback, research developments, and production requirements. Priority items may shift based on enterprise customer needs and emerging RAG research.

**🤝 Community Input**: We welcome suggestions and contributions for roadmap items. Please open an issue or discussion to propose new features or improvements.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
