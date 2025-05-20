# RAG Templates for InterSystems IRIS

This repository contains implementation templates for various Retrieval Augmented Generation (RAG) techniques using InterSystems IRIS.

## Current Development Strategy (As of May 20, 2025)

This project has transitioned to a simplified local development setup:
- **Python Environment:** Managed on the host machine using `uv` (a fast Python package installer and resolver) to create a virtual environment (e.g., `.venv`). Dependencies are defined in `pyproject.toml`.
- **InterSystems IRIS Database:** Runs in a dedicated Docker container, configured via `docker-compose.iris-only.yml`.
- **Database Interaction:** Python RAG pipelines, running on the host, interact with the IRIS database container using client-side SQL executed via the `intersystems-iris` DB-API driver. Stored procedures for vector search are no longer used; vector search SQL is constructed and executed directly by the Python pipelines.

This approach simplifies the development loop, improves stability, and provides a clearer separation between the Python application logic and the IRIS database instance.

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
- Tests with 1000+ real medical documents
- Performance benchmarking and comparison
- Scalable architecture for large document sets

## Requirements

- Python 3.11+
- `uv` (Python package installer and virtual environment manager). Installation: `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`.
- Poetry (Optional, but recommended for a one-time export of `requirements.txt` if `uv` has trouble with direct `pyproject.toml` dependency installation for Poetry projects. If your Poetry version is < 1.1, you might need to upgrade it for the `export` command).
- InterSystems IRIS 2025.1+ (Community Edition or licensed).
- Docker (and Docker Compose) for running the IRIS database container.

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

6.  **Load Test Data (PMC Articles):**
    *   Ensure your `.venv` is active.
    *   Download and process PMC articles into the IRIS database:
        ```bash
        python load_pmc_data.py --limit 1100 --force-recreate-schema no 
        ```
        (Adjust `--limit` as needed. `--force-recreate-schema no` prevents re-dropping tables if schema is already initialized).
        This script will download data to `data/pmc_oas_downloaded/` and load it.

## Running Tests

Ensure your `.venv` is activated (`source .venv/bin/activate`) before running tests.

### Running Unit Tests for a Specific Pipeline
```bash
pytest tests/test_basic_rag.py
pytest tests/test_hyde.py
pytest tests/test_crag.py
pytest tests/test_colbert.py
pytest tests/test_noderag.py
pytest tests/test_graphrag.py
```

### Running All Unit Tests
```bash
pytest tests/
```

### Running Integration Tests (Requires Data Loaded)
The project includes tests that run against a live IRIS database with loaded data.
Refer to specific test files or `Makefile` targets (e.g., `make test-real-pmc-1000`) for these. You may need to adapt `Makefile` commands if they use `poetry run`. A simple way is to run `pytest` with appropriate markers or paths:
```bash
# Example for tests marked as 'real_data' (ensure data is loaded)
pytest -m real_data
```

For more details on testing with 1000+ documents, see [1000_DOCUMENT_TESTING.md](1000_DOCUMENT_TESTING.md) (this document may also need updates to reflect the new setup).

## Technique Documentation

Each RAG technique has detailed implementation documentation:

- [COLBERT_IMPLEMENTATION.md](COLBERT_IMPLEMENTATION.md)
- [NODERAG_IMPLEMENTATION.md](NODERAG_IMPLEMENTATION.md)
- [GRAPHRAG_IMPLEMENTATION.md](GRAPHRAG_IMPLEMENTATION.md)
- [CONTEXT_REDUCTION_STRATEGY.md](CONTEXT_REDUCTION_STRATEGY.md)

## Project Structure

```
rag-templates/
├── basic_rag/           # Standard RAG implementation
├── colbert/             # ColBERT implementation
├── common/              # Shared utilities
├── crag/                # Corrective RAG implementation
├── data/                # Data loading and processing
├── eval/                # Evaluation and benchmarking
├── graphrag/            # GraphRAG implementation
├── hyde/                # HyDE implementation
├── noderag/             # NodeRAG implementation
└── tests/               # Test suite
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
2.  Ensure IRIS Docker container is running and the database is initialized and data loaded.
3.  Run the desired pipeline script directly:
    ```bash
    python basic_rag/pipeline.py
    python hyde/pipeline.py
    # etc.
    ```
    (Note: Some demo scripts might be separate, e.g., `demo_basic_rag.py`. Adjust the command accordingly.)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
