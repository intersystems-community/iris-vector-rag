# Vector Search Investigation

This directory contains scripts and tools for investigating vector search capabilities and limitations in InterSystems IRIS.

## Reproducing Vector Issues in IRIS 2025.1

The `reproduce_vector_issues.py` script provides a simple, standalone way to reproduce two key issues:

1. Parameter substitution issues with TO_VECTOR in IRIS 2025.1
2. Inability to create views, computed columns, or materialized views with TO_VECTOR for HNSW indexing

### Setup Instructions

1. Clone the repository and navigate to the project directory:
   ```
   git clone https://gitlab.iscinternal.com/tdyar/rag-templates.git
   cd rag-templates
   ```

2. Start IRIS 2025.1 using Docker:
   ```
   docker-compose -f docker-compose.iris-only.yml up -d
   ```
   This will start an IRIS 2025.1 container with the necessary configuration.

3. Create a Python virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r investigation/requirements.txt
   ```

4. Configure the connection to IRIS:
   The script uses the following default connection parameters:
   - Host: localhost
   - Port: 1972
   - Namespace: USER
   - Username: _SYSTEM
   - Password: SYS

   If you need to use different connection parameters, you can set them as environment variables:
   ```
   export IRIS_HOST=localhost
   export IRIS_PORT=1972
   export IRIS_NAMESPACE=USER
   export IRIS_USERNAME=_SYSTEM
   export IRIS_PASSWORD=SYS
   ```

5. Run the script:
   ```
   python investigation/reproduce_vector_issues.py
   ```

   The script will:
   - Print SQL statements that you can copy and paste into DBeaver or SQL Shell
   - Save the complete SQL statements to files in the `investigation/` directory
   
   Example output:
   ```
   --- SQL FOR DBEAVER/SQL SHELL ---
   -- Test 1a: Direct query with parameter markers
   -- NOTE: This is the full SQL with actual embedding values (no parameters)
   SELECT VECTOR_COSINE(
       TO_VECTOR('0.1,0.2,0.3,...', 'double', 384),
       TO_VECTOR('0.1,0.2,0.3,...', 'double', 384)
   ) AS score;
   -- Full SQL saved to investigation/sql_for_dbeaver_test1a.sql
   ```

   The SQL files contain the complete statements with actual embedding values, making it easy to run them directly in DBeaver or SQL Shell without any modifications.

6. For DBeaver users who want to use parameterized queries:
   
   We've created a special guide for DBeaver users who want to use parameterized queries:
   
   ```
   investigation/DBEAVER_PARAMETERIZED_QUERIES.md
   ```
   
   This guide provides step-by-step instructions for using DBeaver's parameterized query feature, where you can enter queries with `?` parameters and then be prompted for the values. This is particularly useful for testing the parameter substitution issues with TO_VECTOR.

### What the Script Tests

1. **Parameter Substitution Issues**:
   - Direct query with parameter markers for TO_VECTOR
   - Query with string interpolation for TO_VECTOR

2. **View Creation Issues**:
   - Creating a view with TO_VECTOR
   - Creating a table with a computed column using TO_VECTOR
   - Creating a materialized view with TO_VECTOR

### Expected Results

All tests are expected to fail with specific error messages that confirm the issues:

1. **Parameter Substitution Issues**:
   ```
   ❌ Error executing query with TO_VECTOR and parameter markers: [SQLCODE: <-1>:<Invalid SQL statement>]
   [Location: <Prepare>]
   [%msg: < ) expected, : found ^SELECT VECTOR_COSINE ( TO_VECTOR ( :%qpar(1) , :%qpar>]
   ```

2. **View Creation Issues**:
   ```
   ❌ Failed to create view with TO_VECTOR: [SQLCODE: <-1>:<Invalid SQL statement>]
   [Location: <Prepare>]
   [%msg: < ) expected, LITERAL ('double') found ^                 TO_VECTOR(embedding, 'double'>]
   ```

These failures demonstrate that the dual-table architecture with ObjectScript triggers is the only viable approach for implementing HNSW indexing in IRIS 2025.1.

### Related Documentation

For more details, see the following documentation:

- [HNSW_VIEW_TEST_RESULTS.md](../docs/HNSW_VIEW_TEST_RESULTS.md)
- [VECTOR_SEARCH_DOCUMENTATION_INDEX.md](../docs/VECTOR_SEARCH_DOCUMENTATION_INDEX.md)
- [HNSW_INDEXING_RECOMMENDATIONS.md](../docs/HNSW_INDEXING_RECOMMENDATIONS.md)
- [VECTOR_SEARCH_ALTERNATIVES.md](../docs/VECTOR_SEARCH_ALTERNATIVES.md)
- [IRIS_SQL_VECTOR_LIMITATIONS.md](../docs/IRIS_SQL_VECTOR_LIMITATIONS.md)

## Other Investigation Scripts

- `simple_vector_demo.py`: Simple demonstration of vector storage and search
- `vector_storage_poc.py`: Proof of concept for langchain-iris approach
- `vector_storage_hnsw_poc.py`: Proof of concept for HNSW indexing
- `test_dbapi_vector_params.py`: Test of parameter substitution with TO_VECTOR
- `test_view_hnsw_2025.py`: Test of view-based approach for HNSW indexing