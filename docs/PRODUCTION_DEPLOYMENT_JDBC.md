# Production Deployment Guide - JDBC-Based RAG System

## Overview

This guide provides step-by-step instructions for deploying the JDBC-based RAG system in production. The JDBC solution addresses critical parameter binding issues with vector functions in IRIS SQL.

## Prerequisites

1. **InterSystems IRIS** 2024.1 or later with Vector Search enabled
2. **Java Runtime Environment (JRE)** 8 or later
3. **Python 3.8+** with required dependencies
4. **JDBC Driver**: `intersystems-jdbc-3.8.4.jar` (included in repository)

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   RAG Pipelines │────▶│  JDBC Connector  │────▶│   IRIS Database │
│  (7 Techniques) │     │  (Java Bridge)   │     │  (Vector Store) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Step 1: Environment Setup

### 1.1 Configure Environment Variables

Create a `.env` file in the project root:

```bash
# IRIS Connection
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS

# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here

# Java Configuration
JAVA_HOME=/path/to/java
JDBC_DRIVER_PATH=./intersystems-jdbc-3.8.4.jar
```

### 1.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Verify JDBC Driver

```bash
ls -la intersystems-jdbc-3.8.4.jar
# Should show the JDBC driver file
```

## Step 2: Database Setup

### 2.1 Create Database Schema

```bash
python common/db_init_with_indexes.py
```

This creates:
- `RAG.SourceDocuments` - Main document storage
- `RAG.DocumentChunks` - Chunked documents for retrieval
- `RAG.Entities` - Knowledge graph entities
- `RAG.Relationships` - Entity relationships
- `RAG.ColbertTokenEmbeddings` - ColBERT token embeddings
- All necessary indexes for performance

### 2.2 Verify Schema Creation

```sql
-- Run in IRIS SQL Shell
SELECT COUNT(*) FROM %Dictionary.CompiledClass WHERE Name LIKE 'RAG.%'
-- Should return 5 or more
```

## Step 3: Data Ingestion

### 3.1 Prepare Data

Place your PMC XML files in the data directory:
```
data/
├── PMC000001/
│   └── PMC000001.xml
├── PMC000002/
│   └── PMC000002.xml
└── ...
```

### 3.2 Run Ingestion

```bash
# For production ingestion with chunking
python data/loader_optimized_performance.py --chunk-size 512 --batch-size 100
```

### 3.3 Verify Ingestion

```sql
-- Check document count
SELECT COUNT(*) FROM RAG.SourceDocuments;

-- Check chunk count
SELECT COUNT(*) FROM RAG.DocumentChunks;

-- Check embeddings
SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL;
```

## Step 4: Configure RAG Pipelines

### 4.1 Update Connection Manager

Ensure all pipelines use JDBC by updating imports:

```python
# In each pipeline file
from common.iris_connector_jdbc import get_iris_connection
```

### 4.2 Pipeline Configuration

Each pipeline can be configured with specific parameters:

```python
# Basic RAG
pipeline = BasicRAGPipeline(
    iris_connector=get_iris_connection(),
    embedding_func=get_embedding_func(),
    llm_func=get_llm_func()
)

# HyDE
pipeline = HyDEPipeline(
    iris_connector=get_iris_connection(),
    embedding_func=get_embedding_func(),
    llm_func=get_llm_func()
)

# Continue for all 7 techniques...
```

## Step 5: Performance Optimization

### 5.1 Create Performance Indexes

```bash
python add_performance_indexes.py
```

This creates:
- B-tree indexes on frequently queried columns
- Bitmap indexes for type columns
- Composite indexes for complex queries

### 5.2 Monitor Performance

```bash
python monitor_index_performance_improvements.py
```

## Step 6: Production Deployment

### 6.1 Using Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: .
    environment:
      - IRIS_HOST=iris
      - IRIS_PORT=1972
      - IRIS_NAMESPACE=USER
    volumes:
      - ./intersystems-jdbc-3.8.4.jar:/app/intersystems-jdbc-3.8.4.jar
    depends_on:
      - iris
    ports:
      - "8000:8000"
  
  iris:
    image: intersystemsdc/iris-community:latest
    ports:
      - "1972:1972"
      - "52773:52773"
    volumes:
      - iris-data:/iris-data
```

### 6.2 Using Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-system
  template:
    metadata:
      labels:
        app: rag-system
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-system:latest
        env:
        - name: IRIS_HOST
          value: "iris-service"
        - name: JDBC_DRIVER_PATH
          value: "/app/intersystems-jdbc-3.8.4.jar"
        volumeMounts:
        - name: jdbc-driver
          mountPath: /app/intersystems-jdbc-3.8.4.jar
          subPath: intersystems-jdbc-3.8.4.jar
      volumes:
      - name: jdbc-driver
        configMap:
          name: jdbc-driver-config
```

## Step 7: API Deployment

### 7.1 Create FastAPI Application

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Enterprise RAG System")

# Import all pipelines
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
# ... import all 7 techniques

# Initialize pipelines
pipelines = {
    "basic_rag": BasicRAGPipeline(...),
    "hyde": HyDEPipeline(...),
    # ... initialize all 7
}

class QueryRequest(BaseModel):
    query: str
    technique: str = "basic_rag"
    top_k: int = 10

@app.post("/query")
async def query_rag(request: QueryRequest) -> Dict[str, Any]:
    if request.technique not in pipelines:
        raise HTTPException(status_code=400, detail=f"Unknown technique: {request.technique}")
    
    pipeline = pipelines[request.technique]
    result = pipeline.run(request.query, top_k=request.top_k)
    return result
```

### 7.2 Run API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Step 8: Monitoring and Maintenance

### 8.1 Health Checks

```python
# health_check.py
import sys
from common.iris_connector_jdbc import get_iris_connection

def check_database_health():
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False

def check_vector_search():
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP 1 
            VECTOR_COSINE(TO_VECTOR('1,2,3'), TO_VECTOR('1,2,3')) as similarity
        """)
        result = cursor.fetchone()
        cursor.close()
        return result[0] > 0.99
    except Exception as e:
        print(f"Vector search check failed: {e}")
        return False

if __name__ == "__main__":
    if not check_database_health():
        sys.exit(1)
    if not check_vector_search():
        sys.exit(1)
    print("All health checks passed!")
```

### 8.2 Performance Monitoring

```bash
# Monitor query performance
python scripts/monitor_query_performance.py

# Check index usage
python validate_index_performance.py
```

## Step 9: Troubleshooting

### Common Issues and Solutions

#### 1. JDBC Connection Errors
```
Error: java.lang.ClassNotFoundException: com.intersystems.jdbc.IRISDriver
Solution: Ensure JDBC driver is in classpath and JAVA_HOME is set
```

#### 2. Vector Parameter Binding Errors
```
Error: Argument #1 of vector function VECTOR_COSINE is not a vector
Solution: Ensure you're using JDBC connection, not ODBC
```

#### 3. Memory Issues with Large Datasets
```
Error: Java heap space
Solution: Increase JVM heap size: export _JAVA_OPTIONS="-Xmx4g"
```

#### 4. Slow Query Performance
```
Solution: Run performance optimization scripts and verify indexes
```

## Step 10: Production Checklist

- [ ] Environment variables configured
- [ ] JDBC driver accessible
- [ ] Database schema created
- [ ] Indexes created and optimized
- [ ] Data ingested successfully
- [ ] All 7 RAG techniques tested
- [ ] API endpoints working
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Security measures in place
- [ ] Load testing completed
- [ ] Documentation updated

## Conclusion

The JDBC-based RAG system provides a robust solution for enterprise deployments. The key advantages:

1. **Reliable Vector Operations**: JDBC properly handles vector parameter binding
2. **High Performance**: Optimized indexes and batch processing
3. **Scalability**: Tested with 100K+ documents
4. **Flexibility**: 7 different RAG techniques available
5. **Production Ready**: Comprehensive monitoring and health checks

For additional support, refer to the technical documentation in the `docs/` directory.