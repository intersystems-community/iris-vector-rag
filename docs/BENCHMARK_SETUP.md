# RAG Techniques Benchmarking Setup

This document provides instructions for setting up and running the RAG techniques benchmark tests with a real IRIS database.

## Prerequisites

1. **InterSystems IRIS Database**
   - Installed and running
   - Accessible at the configured address (default: localhost:1972)
   - With an account that has appropriate permissions (default: SuperUser)

2. **Python Environment**
   - Python 3.8+ with Poetry installed
   - All dependencies installed via `poetry install`

3. **Environment Variables**
   - Set up environment variables for IRIS connection:
     ```
     export IRIS_HOST=localhost  # Default
     export IRIS_PORT=1972       # Default
     export IRIS_NAMESPACE=USER  # Default
     export IRIS_USERNAME=SuperUser  # Default
     export IRIS_PASSWORD=SYS    # Default
     ```

## Database Setup

### Option 1: Using an Existing IRIS Instance

1. **Resolve Password Change Required Issue**
   
   If you encounter a "Password change required" error:
   
   ```
   # Connect to IRIS terminal
   iris terminal

   # Log in and change password (if required)
   USER>set $namespace="%SYS"
   %SYS>do ##class(Security.Users).UnExpireUserPasswords("*")
   %SYS>do ##class(Security.Users).ChangePassword("SuperUser","NewPassword")
   
   # Then update your IRIS_PASSWORD environment variable:
   export IRIS_PASSWORD=NewPassword
   ```

2. **Initialize the Database Schema**
   
   Run the database initialization script:
   
   ```
   python -c "from common.db_init import initialize_database; from common.iris_connector import get_iris_connection; initialize_database(get_iris_connection())"
   ```
   
   This will create the necessary tables and indexes for the benchmarking.

### Option 2: Using a Testcontainer (Docker required)

For Docker users, the benchmark can use a testcontainer that creates an ephemeral IRIS instance:

1. **Install Docker** if not already installed

2. **Install testcontainers-iris package**:
   ```
   pip install testcontainers-iris
   ```

3. **Fix for dbname issue**:
   If you encounter a `dbname` error with testcontainers, you need to edit the file:
   
   ```
   # Find the location of the installed package
   python -c "import testcontainers.iris; print(testcontainers.iris.__file__)"
   
   # Edit the file and remove or fix the dbname parameter in the _create_connection_url method
   ```

## Loading Test Data

For the benchmarks to work properly, you need to load test data into the IRIS database.

1. **Load PMC Data**:
   ```
   python load_pmc_data.py --limit 1000
   ```
   
   This will load 1000 sample PMC documents for testing.

2. **Generate Embeddings**:
   ```
   python generate_embeddings.py
   ```
   
   This will create embeddings for the loaded documents.

## Running the Benchmark

Once the environment is set up:

```
python run_benchmark_demo.py
```

The benchmark will:
1. Connect to the IRIS database
2. Run tests for multiple RAG techniques
3. Generate comparison reports with visualizations

## Troubleshooting

1. **Connection Issues**:
   - Verify that IRIS is running: `ps aux | grep iris`
   - Check connection parameters in environment variables
   - Try connecting with other tools to confirm access

2. **Missing Tables/Data**:
   - Run the db_init.py script to create schema
   - Check that PMC data was loaded successfully
   - Verify embeddings were generated

3. **Python Environment Issues**:
   - Ensure all dependencies are installed: `poetry install`
   - Check for version conflicts: `poetry show -o`

## Notes on Real Data Testing

The benchmarking system is designed to work with real data only, and does not fall back to mock data. This ensures that performance metrics accurately reflect real-world usage and adhere to the project's requirement of testing with 1000+ real PMC documents.
