# RAG Techniques Benchmarking Setup

This document provides instructions for setting up the environment to run RAG techniques benchmark tests.

## Current Project Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, full benchmarking with newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, operations requiring these embeddings for benchmarking cannot be performed on newly ingested real data.

This setup guide describes the intended process, but execution of benchmarks with real embeddings is contingent on resolving this blocker. For more details, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1).

## Prerequisites

1. **InterSystems IRIS Database (Primary Setup)**
   - IRIS (version 2025.1+ recommended for vector features) running in a dedicated Docker container.
   - Start using: `docker-compose -f docker-compose.iris-only.yml up -d`
   - Accessible via network (default: `localhost:1972`).
   - Account with appropriate permissions (default: `SuperUser`/`SYS`).

2. **Python Environment**
   - Python 3.11+ installed on your host machine.
   - `uv` (Python package installer and virtual environment manager).
   - Virtual environment created and activated (e.g., `uv venv .venv; source .venv/bin/activate`).
   - All project dependencies installed via `uv pip install -r requirements.txt` or `uv pip install .` (see [`README.md`](README.md:1) for details).

3. **Environment Variables**
   - Set up environment variables for IRIS connection as per [`README.md`](README.md:1) (e.g., `IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`, `IRIS_USERNAME`, `IRIS_PASSWORD`).

## Database Setup

### Option 1: Using the Dedicated IRIS Docker Container (Primary Method)

1. **Ensure IRIS Container is Running:**
   ```bash
   docker-compose -f docker-compose.iris-only.yml up -d
   ```

2. **Initialize the Database Schema:**
   Run the database initialization script from your activated `uv` virtual environment:
   ```bash
   python run_db_init_local.py --force-recreate
   ```
   This will create the necessary tables and indexes.

### Option 2: Using a Testcontainer (Alternative for Specific Scenarios)

For isolated tests, Testcontainers can be used. Ensure Docker is running.

1. **Install testcontainers-iris package** (if not already part of project dependencies):
   ```bash
   # Ensure .venv is active
   uv pip install "testcontainers-iris>=1.2.0" "testcontainers>=3.7.0"
   ```

2. **Note on `dbname` issue (if encountered with older testcontainers-iris versions):**
   Older versions might have a `dbname` parameter issue in `_create_connection_url`. This may require manual patching of the installed package or ensuring you use a version where this is fixed.

## Loading Test Data

For the benchmarks to work properly with real data, you need to load test data into the IRIS database.
(Note: The scripts mentioned below are currently in `scripts_to_review/`. Their canonical status and final location should be confirmed as per Phase 0 of the documentation update plan.)

1. **Load PMC Data (Text Content)**:
   ```bash
   # Ensure .venv is active
   python scripts_to_review/load_pmc_data.py --limit 1000
   ```
   This will load the text content of 1000 sample PMC documents.

2. **Generate and Load Embeddings**:
   ```bash
   # Ensure .venv is active
   # THIS STEP IS CURRENTLY BLOCKED by the TO_VECTOR/ODBC issue.
   python scripts_to_review/generate_embeddings.py
   ```
   This step is intended to create and load embeddings for the documents.

## Running the Benchmark

Once the environment is set up (and assuming the embedding blocker is resolved for real-data benchmarks):

```bash
# Ensure .venv is active
python scripts/run_rag_benchmarks.py
```

The benchmark script will:
1. Connect to the IRIS database.
2. Run tests for multiple RAG techniques.
3. Generate comparison reports with visualizations in the `benchmark_results/` directory.

## Troubleshooting

1. **Connection Issues**:
   - Verify the IRIS Docker container is running (`docker ps`).
   - Check connection parameters in environment variables.
   - Test connection using `python -c "from common.iris_connector import get_iris_connection; print(get_iris_connection())"`.

2. **Missing Tables/Data**:
   - Ensure `python run_db_init_local.py --force-recreate` was run successfully.
   - Check that PMC text data was loaded.
   - Verify if embeddings were generated and loaded (currently blocked).

3. **Python Environment Issues**:
   - Ensure your `uv` virtual environment is active.
   - Confirm all dependencies are installed (`uv pip install -r requirements.txt` or `uv pip install .`).

## Notes on Real Data Testing

The benchmarking system is designed to work with real data. However, due to the current **blocker** preventing the loading of new embeddings, full benchmarks on real data requiring these embeddings cannot be completed. Mock data or pre-existing data (if available and compatible) might be used for limited testing of the benchmark framework itself.
This setup aims to adhere to the project's requirement of testing with 1000+ real PMC documents once the blocker is resolved.
