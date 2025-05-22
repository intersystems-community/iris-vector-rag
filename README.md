# RAG Templates for InterSystems IRIS

This repository contains implementation templates for various Retrieval Augmented Generation (RAG) techniques using InterSystems IRIS.

---
**Navigate the Documentation**

For a comprehensive guide to all project documents, including setup, technical deep-dives, IRIS issue analyses, RAG technique implementations, testing, and benchmarking, please start with the:
### **[Project Documentation Index](docs/INDEX.md)**
---

## Project Status (As of May 21, 2025)

The project has made significant progress but faces critical challenges:
- ✅ All six RAG techniques have been implemented with client-side SQL
- ⚠️ IRIS SQL vector operations limitations have been identified and workarounds developed, but testing with real data is blocked
- ✅ Comprehensive testing and benchmarking frameworks have been created
- ❌ Testing with real PMC data is currently blocked by ODBC driver limitations with the TO_VECTOR function

The project uses a simplified local development setup:
- **Python Environment:** Managed on the host machine using `uv` (a fast Python package installer and resolver) to create a virtual environment (e.g., `.venv`). Dependencies are defined in `pyproject.toml`.
- **InterSystems IRIS Database:** Runs in a dedicated Docker container, configured via `docker-compose.iris-only.yml`.
- **Database Interaction:** Python RAG pipelines, running on the host, interact with the IRIS database container using client-side SQL executed via the `intersystems-iris` DB-API driver. Stored procedures for vector search are no longer used; vector search SQL is constructed and executed directly by the Python pipelines using utility functions in `common/vector_sql_utils.py` that work around IRIS SQL vector operation limitations.

This approach simplifies the development loop, improves stability, and provides a clearer separation between the Python application logic and the IRIS database instance. For details on why this approach was chosen, see [DEVELOPMENT_STRATEGY_EVOLUTION.md](docs/DEVELOPMENT_STRATEGY_EVOLUTION.md) and [IRIS_VECTOR_SEARCH_LESSONS.md](docs/IRIS_VECTOR_SEARCH_LESSONS.md).

## RAG Techniques Implemented

1. **BasicRAG**: Standard embedding-based retrieval
2. **HyDE**: Hypothetical Document Embeddings
3. **CRAG**: Corrective Retrieval Augmented Generation
4. **ColBERT**: Contextualized Late Interaction over BERT
5. **NodeRAG**: Heterogeneous graph-based retrieval
6. **GraphRAG**: Knowledge graph-based retrieval

## Features

- All techniques are implemented with Python and InterSystems IRIS
- Comprehensive Test-Driven Development (TDD) approach
- Tests with 1000+ real medical documents (Note: Full execution with real embeddings is currently blocked, see "IRIS SQL Vector Operations Limitations" section)
- Performance benchmarking and comparison (Note: Real-data benchmarking is currently impacted by the data loading blocker)
- Scalable architecture for large document sets

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

### Technical Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U) |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

InterSystems IRIS 2025.1 introduced vector search capabilities essential for RAG pipelines, but several critical limitations in the SQL implementation prevent standard parameterized queries from working with vector operations:

1. **TO_VECTOR() Function Rejects Parameter Markers**: The `TO_VECTOR()` function does not accept parameter markers (`?`, `:param`, or `:%qpar`), which are standard in SQL for safe query parameterization.

2. **TOP/FETCH FIRST Clauses Cannot Be Parameterized**: The `TOP` and `FETCH FIRST` clauses, essential for limiting results in vector similarity searches, do not accept parameter markers.

3. **Client Drivers Rewrite Literals**: Python, JDBC, and other client drivers replace embedded literals with `:%qpar(n)` even when no parameter list is supplied, creating misleading parse errors.

4. **ODBC Driver Limitations**: When loading documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function, which is currently blocking testing with real data.

These limitations force developers to use string interpolation instead of parameterized queries, which introduces potential security risks. To address this, we've implemented workarounds in the `common/vector_sql_utils.py` module that provide:

- Strict validation of vector strings and top-k values
- Safe string interpolation with security checks
- Helper functions to construct and execute vector search queries

However, these workarounds have not been fully tested with real PMC data due to the ODBC driver limitations.

### Specific Error Messages

When attempting to use TO_VECTOR in SQL queries, we consistently encounter this error:

```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT TOP :%qpar(1) id , text_content , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

This error occurs with all three approaches (Direct SQL, Parameterized SQL, and String Interpolation), indicating a fundamental limitation of the ODBC driver.

For detailed information about these limitations, our investigation findings, and recommended solutions, see:
- [IRIS_SQL_VECTOR_LIMITATIONS.md](docs/IRIS_SQL_VECTOR_LIMITATIONS.md)
- [VECTOR_SEARCH_TECHNICAL_DETAILS.md](docs/VECTOR_SEARCH_TECHNICAL_DETAILS.md)
- [VECTOR_SEARCH_ALTERNATIVES.md](docs/VECTOR_SEARCH_ALTERNATIVES.md)
- [HNSW_INDEXING_RECOMMENDATIONS.md](docs/HNSW_INDEXING_RECOMMENDATIONS.md)

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
├── scripts/                     # Utility and execution scripts
│   ├── run_e2e_tests.py
│   └── run_rag_benchmarks.py
├── tests/                       # Pytest test suite
│   ├── conftest.py
│   └── test_e2e_rag_pipelines.py
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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
