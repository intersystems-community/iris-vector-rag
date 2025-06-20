# RAG Templates Deployment Guide

## 🚀 Production Deployment Guide for InterSystems IRIS RAG Templates

This guide provides comprehensive instructions for deploying the RAG Templates system in production environments, from development to enterprise scale.

## 📋 Prerequisites

### System Requirements
- **InterSystems IRIS**: 2025.1+ (Community or Enterprise Edition)
- **Python**: 3.11+ with virtual environment support
- **Memory**: Minimum 8GB RAM (16GB+ recommended for enterprise)
- **Storage**: 10GB+ free space (depends on document volume)
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Dependencies
- **Docker & Docker Compose**: For IRIS container deployment
- **Conda**: Python environment manager (recommended) or `uv`
- **Git**: For repository management
- **IRIS Python Driver**: `intersystems-irispython>=5.1.2`

## 🏗️ Deployment Architecture

### Recommended Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   RAG Service   │    │  IRIS Database  │
│     Layer       │◄──►│     Layer       │◄──►│     Layer       │
│                 │    │                 │    │                 │
│ • Web UI        │    │ • 7 RAG Tech.   │    │ • Vector Store  │
│ • REST API      │    │ • Chunking      │    │ • HNSW Indexes  │
│ • CLI Interface │    │ • Embeddings    │    │ • ObjectScript  │
│ • Monitoring    │    │ • Reconciliation│    │ • Schema Mgmt   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Installation Steps

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone repository
git clone <repository_url>
cd rag-templates

# Create and activate conda environment
conda create -n iris_vector python=3.11 -y
conda activate iris_vector

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using uv
```bash
# Clone repository
git clone <repository_url>
cd rag-templates

# Create Python virtual environment
uv venv .venv --python python3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

#### Option C: Using the provided activation script
```bash
# Use the provided environment setup
./activate_env.sh
```

### 2. Database Setup

#### Option A: Docker Deployment (Recommended for Development)
```bash
# Start IRIS container using docker-compose
docker-compose -f docker-compose.iris-only.yml up -d

# Wait for container to be ready (check health)
docker-compose -f docker-compose.iris-only.yml ps

# Verify container is running
docker ps | grep iris
```

#### Option B: Native IRIS Installation (Production)
```bash
# Install IRIS on your system
# Configure connection parameters in environment variables
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_USERNAME=SuperUser
export IRIS_PASSWORD=SYS
export IRIS_NAMESPACE=USER
```

### 3. Database Schema Initialization

```bash
# Method 1: Using Makefile (Recommended)
make setup-db

# Method 2: Direct Python execution
python common/db_init_with_indexes.py

# Method 3: Using the schema manager
python -c "
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)
schema_manager = SchemaManager(connection_manager, config_manager)

# Ensure all schemas are up to date
schema_manager.ensure_table_schema('DocumentEntities')
print('✅ Schema initialization complete')
"
```

### 4. Data Loading

```bash
# Load sample PMC data (1000+ documents)
make load-1000

# Alternative: Direct loading
python -c "
from data.loader import process_and_load_documents
result = process_and_load_documents('data/pmc_oas_downloaded', limit=1000, batch_size=50, use_mock=False)
print(f'Loaded: {result}')
"

# Verify data loading
make check-data
```

## 🎯 RAG Technique Selection

### Performance-Based Selection Guide

#### For Low-Latency Applications (< 100ms)
**Recommended**: GraphRAG or HyDE
- **GraphRAG**: 0.03s avg, 20.0 docs avg ⚡
- **HyDE**: 0.03s avg, 5.0 docs avg ⚡

```python
# GraphRAG deployment
from iris_rag.pipelines.graphrag import GraphRAGPipeline
pipeline = GraphRAGPipeline()
result = pipeline.query("your query", top_k=20)
```

#### For IRIS-Native Integration
**Recommended**: Hybrid iFind RAG
- **Performance**: 0.07s avg, 10.0 docs avg
- **Benefits**: Native IRIS vector search, ObjectScript integration

```python
# Hybrid iFind RAG deployment
from iris_rag.pipelines.hybrid_ifind import HybridiFindRAGPipeline
pipeline = HybridiFindRAGPipeline()
result = pipeline.query("your query", top_k=10)
```

#### For Balanced Performance
**Recommended**: NodeRAG or BasicRAG
- **NodeRAG**: 0.07s avg, 20.0 docs avg
- **BasicRAG**: 0.45s avg, 5.0 docs avg

#### For High-Precision Applications
**Recommended**: CRAG or OptimizedColBERT
- **CRAG**: 0.56s avg, 18.2 docs avg (self-correcting)
- **OptimizedColBERT**: 3.09s avg, 5.0 docs avg (token-level precision)

## 🔄 Configuration Management

### Environment-Specific Configuration

The system supports multiple configuration approaches:

1. **Main Configuration**: [`config/config.yaml`](../../config/config.yaml)
2. **Environment Variables**: `RAG_` prefixed variables
3. **Pipeline-Specific**: [`config/pipelines.yaml`](../../config/pipelines.yaml)
4. **Reconciliation**: [`config/colbert_reconciliation_example.yaml`](../../config/colbert_reconciliation_example.yaml)

#### Development Configuration
```yaml
# config/config.yaml
database:
  db_host: "localhost"
  db_port: 1972
  db_user: "SuperUser"
  db_password: "SYS"
  db_namespace: "USER"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

logging:
  log_level: "INFO"
```

#### Production Configuration
```bash
# Environment variables for production
export RAG_DATABASE__DB_HOST="production-host"
export RAG_DATABASE__DB_PORT=1972
export RAG_DATABASE__DB_USER="production_user"
export RAG_DATABASE__DB_PASSWORD="secure_password"
export RAG_LOGGING__LOG_LEVEL="WARNING"
```

### Configuration Validation
```bash
# Validate configuration
python -c "
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager()
print('✅ Configuration loaded successfully')
print(f'Database host: {config.get(\"database:db_host\")}')
print(f'Embedding model: {config.get(\"embedding_model:name\")}')
"
```

## 🏢 Enterprise Deployment

### Scaling Configuration

#### Small Scale (< 1,000 documents)
```python
# Configuration
CHUNK_SIZE = 512
OVERLAP = 50
BATCH_SIZE = 100
MAX_WORKERS = 4

# Recommended techniques: GraphRAG, HyDE
```

#### Medium Scale (1,000 - 10,000 documents)
```python
# Configuration
CHUNK_SIZE = 1024
OVERLAP = 100
BATCH_SIZE = 500
MAX_WORKERS = 8

# Recommended techniques: Hybrid iFind RAG, NodeRAG
```

#### Large Scale (10,000+ documents)
```python
# Configuration
CHUNK_SIZE = 2048
OVERLAP = 200
BATCH_SIZE = 1000
MAX_WORKERS = 16

# Recommended techniques: All techniques with load balancing
# Enable HNSW indexing for Enterprise Edition
```

### Enterprise Validation

```bash
# Run comprehensive validation
make validate-all

# Test all pipelines
make validate-all-pipelines

# Run enterprise-scale testing
make test-1000

# Performance benchmarking
make benchmark
```

### Automated Pipeline Setup
```bash
# Auto-setup all pipelines with validation
make auto-setup-all

# Setup specific pipeline
make auto-setup-pipeline PIPELINE=colbert

# Test with auto-healing
make test-with-auto-setup
```

## 📊 Monitoring & Performance

### Health Monitoring Setup

```bash
# Setup monitoring infrastructure
python scripts/utilities/setup_monitoring.py

# Run comprehensive health check
python -c "
from iris_rag.monitoring.health_monitor import HealthMonitor
monitor = HealthMonitor()
results = monitor.run_comprehensive_health_check()
for component, result in results.items():
    print(f'{component}: {result.status} - {result.message}')
"
```

### Performance Monitoring

```python
# Built-in performance monitoring
from common.utils import PerformanceMonitor

monitor = PerformanceMonitor()
with monitor.track("rag_query"):
    result = pipeline.query("your query")

# Get metrics
metrics = monitor.get_metrics()
print(f"Average latency: {metrics['avg_latency']:.3f}s")
print(f"Throughput: {metrics['queries_per_second']:.2f} q/s")
```

### Continuous Monitoring

```bash
# Start monitoring daemon
python scripts/monitor_performance.sh

# Log rotation
python scripts/rotate_logs.sh

# Health check scheduling (add to crontab)
*/15 * * * * cd /path/to/rag-templates && python -c "from iris_rag.monitoring.health_monitor import HealthMonitor; HealthMonitor().run_comprehensive_health_check()"
```

## 🔒 Security Considerations

### Database Security
```python
# Secure connection configuration
IRIS_CONFIG = {
    'host': os.getenv('IRIS_HOST'),
    'port': int(os.getenv('IRIS_PORT', 1972)),
    'username': os.getenv('IRIS_USERNAME'),
    'password': os.getenv('IRIS_PASSWORD'),
    'namespace': os.getenv('IRIS_NAMESPACE', 'USER'),
    'ssl': True,  # Enable SSL in production
    'ssl_verify': True
}
```

### Environment Variable Security
```bash
# Use secure environment variable management
# Never commit credentials to version control

# Example .env file (not committed)
IRIS_HOST=production-host
IRIS_USERNAME=secure_user
IRIS_PASSWORD=secure_password
IRIS_NAMESPACE=PRODUCTION

# Load with python-dotenv
python -c "
from dotenv import load_dotenv
load_dotenv()
print('✅ Environment variables loaded securely')
"
```

### API Security
- Implement authentication and authorization
- Use HTTPS for all communications
- Validate and sanitize all inputs
- Implement rate limiting
- Use the CLI interface for secure operations

## 🚀 Production Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured securely
- [ ] Database schema initialized and validated
- [ ] Sample data loaded and validated (`make check-data`)
- [ ] All pipelines auto-configured (`make auto-setup-all`)
- [ ] Performance benchmarks completed (`make benchmark`)
- [ ] Security configurations applied
- [ ] Monitoring systems configured (`python scripts/utilities/setup_monitoring.py`)
- [ ] Health checks passing (`make status`)

### Deployment
- [ ] Application deployed to production environment
- [ ] Database connections verified (`make test-dbapi`)
- [ ] All 7 RAG techniques tested (`make validate-all-pipelines`)
- [ ] Schema management system validated
- [ ] Performance monitoring active
- [ ] Health checks passing
- [ ] CLI interface accessible

### Post-Deployment
- [ ] Load testing completed (`make test-1000`)
- [ ] Performance metrics within acceptable ranges
- [ ] Error handling validated
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Monitoring dashboards configured

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check IRIS container status
docker ps | grep iris

# Test connection using Makefile
make test-dbapi

# Manual connection test
python -c "
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
print('✅ Connection successful' if conn else '❌ Connection failed')
if conn:
    conn.close()
"
```

#### Performance Issues
```bash
# Run performance diagnostics
make validate-all

# Check system status
make status

# Run health checks
python -c "
from iris_rag.monitoring.health_monitor import HealthMonitor
monitor = HealthMonitor()
results = monitor.run_comprehensive_health_check()
print(f'Overall status: {monitor.get_overall_health_status(results)}')
"
```

#### Schema Issues
```bash
# Check schema status
python -c "
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)
schema_manager = SchemaManager(connection_manager, config_manager)

status = schema_manager.get_schema_status()
for table, info in status.items():
    print(f'{table}: {info[\"status\"]}')
"

# Force schema migration if needed
python -c "
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)
schema_manager = SchemaManager(connection_manager, config_manager)

success = schema_manager.ensure_table_schema('DocumentEntities')
print(f'Schema migration: {\"✅ Success\" if success else \"❌ Failed\"}')
"
```

#### Pipeline Issues
```bash
# Validate specific pipeline
make validate-pipeline PIPELINE=basic

# Auto-fix pipeline issues
make auto-setup-pipeline PIPELINE=colbert

# Test specific pipeline
make test-pipeline PIPELINE=graphrag
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Enable HNSW indexing (Enterprise Edition)
CREATE INDEX idx_embeddings_hnsw ON RAG.SourceDocuments (embedding) 
USING HNSW WITH (m=16, ef_construction=200);

-- Optimize vector search performance
SET QUERY_TIMEOUT = 30;
SET VECTOR_SEARCH_CACHE = 1000;
```

### Application Optimization
```python
# Connection pooling
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)

# Configure connection pool
connection_manager.configure_pool(
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600
)

# Batch processing
def process_documents_batch(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        process_batch(batch)
```

### Memory Optimization
```bash
# Monitor memory usage
python -c "
from iris_rag.monitoring.health_monitor import HealthMonitor
monitor = HealthMonitor()
result = monitor.check_system_resources()
print(f'Memory usage: {result.metrics.get(\"memory_percent\", 0):.1f}%')
"

# Optimize embedding batch sizes
export RAG_PIPELINES__BASIC__EMBEDDING_BATCH_SIZE=16
export RAG_COLBERT__REMEDIATION__EMBEDDING_GENERATION_BATCH_SIZE=16
```

## 🔄 Maintenance

### Regular Maintenance Tasks
```bash
# Daily health checks
make status

# Weekly performance validation
make validate-all

# Monthly comprehensive testing
make test-1000

# Quarterly scale testing (if applicable)
make benchmark
```

### Automated Maintenance
```bash
# Setup cron jobs for automated maintenance

# Daily health check (6 AM)
0 6 * * * cd /path/to/rag-templates && make status >> logs/daily_health.log 2>&1

# Weekly validation (Sunday 2 AM)
0 2 * * 0 cd /path/to/rag-templates && make validate-all >> logs/weekly_validation.log 2>&1

# Monthly comprehensive test (1st of month, 3 AM)
0 3 1 * * cd /path/to/rag-templates && make test-1000 >> logs/monthly_test.log 2>&1
```

### Backup and Recovery
```bash
# Database backup (IRIS-specific)
iris backup /path/to/backup/

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/ *.yml *.json

# Application backup
tar -czf app_backup_$(date +%Y%m%d).tar.gz iris_rag/ common/ scripts/

# Recovery testing
make validate-all
```

### Log Management
```bash
# Setup log rotation
python scripts/utilities/setup_monitoring.py

# Manual log rotation
find logs/ -name "*.log" -size +100M -exec gzip {} \;
find logs/ -name "*.log.gz" -mtime +30 -delete

# Log analysis
tail -f logs/system.log
grep ERROR logs/performance/*.log
```

## 🛠️ CLI Interface

### Installation and Usage
```bash
# Method 1: Python module (Recommended)
python -m iris_rag.cli --help
python -m iris_rag.cli status --pipeline colbert

# Method 2: Standalone script
./ragctl --help
./ragctl run --pipeline colbert --force

# Method 3: Through Makefile
make validate-pipeline PIPELINE=basic
```

### Common CLI Operations
```bash
# Check system status
./ragctl status

# Run reconciliation
./ragctl run --pipeline colbert

# Dry-run analysis
./ragctl run --pipeline basic --dry-run

# Continuous monitoring
./ragctl daemon --pipeline colbert --interval 3600
```

## 📞 Support and Resources

### Documentation
- **Main Documentation**: [`docs/INDEX.md`](../INDEX.md)
- **Configuration Guide**: [`docs/CONFIGURATION.md`](../CONFIGURATION.md)
- **CLI Usage**: [`docs/CLI_RECONCILIATION_USAGE.md`](../CLI_RECONCILIATION_USAGE.md)
- **Technical Details**: [`docs/IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)

### Performance Benchmarks
- **Enterprise Validation**: [`ENTERPRISE_VALIDATION_COMPLETE.md`](../../ENTERPRISE_VALIDATION_COMPLETE.md)
- **Chunking Performance**: [`ENHANCED_CHUNKING_IMPLEMENTATION_COMPLETE.md`](../../ENHANCED_CHUNKING_IMPLEMENTATION_COMPLETE.md)
- **Hybrid iFind RAG**: [`HYBRID_IFIND_RAG_IMPLEMENTATION_COMPLETE.md`](../../HYBRID_IFIND_RAG_IMPLEMENTATION_COMPLETE.md)

### Deployment Scripts
- **Automated Deployment**: [`scripts/utilities/deploy_rag_system.py`](../../scripts/utilities/deploy_rag_system.py)
- **Monitoring Setup**: [`scripts/utilities/setup_monitoring.py`](../../scripts/utilities/setup_monitoring.py)
- **Health Monitoring**: [`iris_rag/monitoring/health_monitor.py`](../../iris_rag/monitoring/health_monitor.py)

### Contact Information
- **Technical Issues**: Check documentation and run diagnostic scripts
- **Performance Questions**: Review benchmark results and optimization guides
- **Enterprise Support**: Consult enterprise validation reports
- **Configuration Issues**: Refer to [`docs/CONFIGURATION.md`](../CONFIGURATION.md)

## 🎯 Next Steps

### Immediate Actions
1. **Deploy development environment** using Docker setup
2. **Run validation scripts** to ensure all techniques work (`make validate-all`)
3. **Load sample data** and test performance (`make load-1000`)
4. **Configure monitoring** and health checks (`python scripts/utilities/setup_monitoring.py`)

### Production Readiness
1. **Scale testing** with enterprise validation scripts (`make test-1000`)
2. **Security hardening** with production configurations
3. **Performance optimization** based on benchmark results
4. **Team training** on deployment and maintenance procedures
5. **CLI interface setup** for operational management

### Future Enhancements
1. **LLM Integration**: Connect to production language models
2. **API Development**: RESTful service endpoints
3. **UI Development**: User interface for RAG interactions
4. **Advanced Monitoring**: Real-time performance dashboards
5. **Automated Scaling**: Dynamic resource allocation

## 🔄 Rollback Procedures

### Emergency Rollback
```bash
# Stop current deployment
docker-compose down

# Restore from backup
tar -xzf app_backup_YYYYMMDD.tar.gz
tar -xzf config_backup_YYYYMMDD.tar.gz

# Restore database (IRIS-specific)
iris restore /path/to/backup/

# Restart with previous configuration
docker-compose up -d

# Validate rollback
make validate-all
```

### Gradual Rollback
```bash
# Disable problematic pipelines
export RAG_PIPELINES__PROBLEMATIC_PIPELINE__ENABLED=false

# Restart with reduced functionality
make auto-setup-all

# Monitor and validate
make status
```

This deployment guide provides a comprehensive foundation for successfully deploying the RAG Templates system in production environments, from small-scale development to enterprise-grade deployments with proper monitoring, security, and maintenance procedures.