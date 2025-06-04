# Remote Server Deployment Guide

This guide will help you set up the RAG templates repository on a remote server with native VECTOR types from the start.

## Prerequisites

- Remote server with Docker and Docker Compose installed
- Git installed on the remote server
- SSH access to the remote server
- At least 16GB RAM recommended for large-scale ingestion

## Quick Setup Script

### 1. Clone and Setup Repository

```bash
# Clone the repository and checkout the vector migration branch
git clone <your-repo-url> rag-templates
cd rag-templates

# Checkout the branch with native VECTOR implementation
git checkout <vector-migration-branch-name>

# Make setup script executable
chmod +x scripts/remote_setup.sh

# Run the setup script
./scripts/remote_setup.sh
```

### 2. Manual Setup Steps

If you prefer manual setup:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
# OR if using Poetry:
poetry install

# 2. Start IRIS with native VECTOR schema
docker-compose -f docker-compose.iris-only.yml up -d

# 3. Wait for IRIS to be ready (check with docker logs)
docker logs iris_db_rag_licensed

# 4. Initialize database with native VECTOR schema
python common/db_init_with_indexes.py

# 5. Verify the schema
python scripts/verify_native_vector_schema.py
```

## Environment Configuration

### Docker Compose Configuration

The `docker-compose.iris-only.yml` file is configured for:
- IRIS with vector search capabilities
- Persistent volumes for data
- Optimized memory settings
- Port mappings: 1972 (SQL), 52773 (Management Portal)

### Database Schema

The native VECTOR schema includes:
- `RAG.SourceDocuments` with `embedding VECTOR(DOUBLE, 384)` 
- `RAG.DocumentChunks` with `embedding VECTOR(DOUBLE, 384)`
- Pre-configured HNSW indexes for optimal performance
- Proper foreign key relationships

## Data Ingestion

### Large-Scale PMC Ingestion

```bash
# Download and ingest 100K+ PMC articles
python scripts/ingest_100k_documents.py

# Monitor progress
python scripts/monitor_ingestion_progress.py

# Verify ingestion
python scripts/verify_ingestion_complete.py
```

### Custom Data Ingestion

```bash
# For custom document ingestion
python data/loader.py --source /path/to/documents --batch-size 1000

# With chunking enabled
python data/loader.py --source /path/to/documents --enable-chunking --chunk-size 512
```

## Performance Optimization

### HNSW Index Configuration

The system is pre-configured with optimized HNSW indexes:
- M=16, efConstruction=200 for balanced performance
- COSINE distance for semantic similarity
- Automatic index creation during schema initialization

### Memory and Performance Settings

```bash
# Check IRIS memory allocation
docker exec iris_db_rag_licensed iris terminal IRIS -U USER -c "write $system.SYS.GetMemoryInfo()"

# Monitor query performance
python scripts/monitor_query_performance.py
```

## RAG Pipeline Testing

### Run All RAG Techniques

```bash
# Comprehensive benchmark of all 7 RAG techniques
python eval/enterprise_rag_benchmark_final.py

# Individual technique testing
python basic_rag/pipeline_v2.py
python crag/pipeline_v2.py
python hyde/pipeline_v2.py
python noderag/pipeline_v2.py
python colbert/pipeline_optimized.py
python hybrid_ifind_rag/pipeline_v2.py
python graphrag/pipeline_v2.py
```

### Performance Validation

```bash
# Verify sub-100ms query performance
python scripts/validate_hnsw_performance.py

# Run enterprise validation suite
python scripts/enterprise_rag_validator.py
```

## Remote Access and Monitoring

### SSH Tunneling for Management Portal

```bash
# On your local machine, create SSH tunnel
ssh -L 52773:localhost:52773 user@remote-server

# Access IRIS Management Portal at http://localhost:52773/csp/sys/UtilHome.csp
```

### Remote Development

```bash
# Use VS Code Remote SSH extension
# Or use terminal-based editors like vim/nano

# For file transfer
scp local-file.py user@remote-server:/path/to/rag-templates/
rsync -av local-directory/ user@remote-server:/path/to/rag-templates/
```

## Troubleshooting

### Common Issues

1. **IRIS Container Won't Start**
   ```bash
   docker logs iris_db_rag_licensed
   # Check memory allocation and port conflicts
   ```

2. **Vector Index Performance Issues**
   ```bash
   python scripts/diagnose_hnsw_performance.py
   ```

3. **Memory Issues During Ingestion**
   ```bash
   # Reduce batch size
   export INGESTION_BATCH_SIZE=100
   python data/loader.py --batch-size 100
   ```

### Log Monitoring

```bash
# Monitor all logs
docker-compose logs -f

# Monitor specific service
docker logs -f iris_db_rag_licensed

# Application logs
tail -f logs/ingestion.log
tail -f logs/rag_performance.log
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
python scripts/backup_iris_while_running.py

# Verify backup
python scripts/verify_backup_integrity.py
```

### Configuration Backup

```bash
# Backup configuration and schema
tar -czf rag-config-backup.tar.gz docker-compose.yml common/ config/
```

## Next Steps After Setup

1. **Verify Installation**: Run the verification script
2. **Ingest Data**: Start with a small dataset to verify everything works
3. **Run Benchmarks**: Execute the enterprise benchmark suite
4. **Scale Up**: Proceed with full-scale data ingestion
5. **Monitor Performance**: Set up continuous monitoring

## Support Commands

```bash
# Quick health check
python scripts/system_health_check.py

# Performance report
python scripts/generate_performance_report.py

# Schema verification
python scripts/verify_schema_integrity.py
```

This setup provides a production-ready RAG system with native VECTOR types, optimized for performance and scalability.