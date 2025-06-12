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
- **uv**: Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Git**: For repository management
- **ODBC Drivers**: InterSystems IRIS ODBC drivers

## 🏗️ Deployment Architecture

### Recommended Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   RAG Service   │    │  IRIS Database  │
│     Layer       │◄──►│     Layer       │◄──►│     Layer       │
│                 │    │                 │    │                 │
│ • Web UI        │    │ • 7 RAG Tech.   │    │ • Vector Store  │
│ • REST API      │    │ • Chunking      │    │ • HNSW Indexes  │
│ • Load Balancer │    │ • Embeddings    │    │ • ObjectScript  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Installation Steps

### 1. Environment Setup

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

### 2. Database Setup

#### Option A: Docker Deployment (Recommended for Development)
```bash
# Start IRIS container
docker-compose -f docker-compose.iris-only.yml up -d

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
# Initialize core schema
python common/db_init.py --force-recreate

# Setup enhanced chunking schema
python -c "
from chunking.enhanced_chunking_service import EnhancedChunkingService
service = EnhancedChunkingService()
service.setup_database_schema()
"

# Setup Hybrid iFind RAG schema (attempts ObjectScript deployment)
python scripts/setup_hybrid_ifind_rag.py
# NOTE: The above script attempts to deploy and compile the necessary ObjectScript class
# for iFind. Due to potential environment-specific issues with IRIS class resolution
# via scripted 'docker exec', this step may not fully succeed in making the iFind
# index operational. If iFind functionality is still missing after this step (e.g.,
# errors about "Index TEXTCONTENTFTI not found"), refer to
# 'docs/IFIND_IMPLEMENTATION_NOTES.md' for manual troubleshooting and deployment
# steps that must be performed directly within an IRIS environment (Studio/Terminal).
```

### 4. Data Loading

```bash
# Load sample PMC data (1000+ documents)
python scripts/download_pmc_data.py --limit 1100 --load-colbert

# Verify data loading
python scripts/verify_real_data_testing.py
```

## 🎯 RAG Technique Selection

### Performance-Based Selection Guide

#### For Low-Latency Applications (< 100ms)
**Recommended**: GraphRAG or HyDE
- **GraphRAG**: 0.03s avg, 20.0 docs avg ⚡
- **HyDE**: 0.03s avg, 5.0 docs avg ⚡

```python
# GraphRAG deployment
from graphrag.pipeline import GraphRAGPipeline
pipeline = GraphRAGPipeline()
result = pipeline.query("your query", top_k=20)
```

#### For IRIS-Native Integration
**Recommended**: Hybrid iFind RAG
- **Performance**: 0.07s avg, 10.0 docs avg
- **Benefits**: Native IRIS vector search, ObjectScript integration

```python
# Hybrid iFind RAG deployment
from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline
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

## 🔄 Enhanced Chunking Deployment

### Chunking Strategy Selection

```python
from chunking.enhanced_chunking_service import EnhancedChunkingService

# Initialize service
chunking_service = EnhancedChunkingService()

# Strategy selection based on use case
strategies = {
    'biomedical': 'semantic',      # Best for medical literature
    'general': 'adaptive',         # Auto-selects best strategy
    'performance': 'recursive',    # Fastest processing
    'comprehensive': 'hybrid'      # Multi-strategy approach
}

# Process documents
chunks = chunking_service.chunk_document(
    text=document_text,
    strategy=strategies['biomedical'],
    chunk_size=512,
    overlap=50
)
```

### Performance Optimization
- **Processing Rate**: 1,633-3,858 documents/second
- **Token Accuracy**: 95%+ for biomedical text
- **Quality Score**: 0.77 for semantic strategies

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

### Enterprise Validation Script

```bash
# Run enterprise validation
python scripts/enterprise_validation_with_fixed_colbert.py \
    --num-queries 50 \
    --min-docs 1000 \
    --output-dir enterprise_results

# Scale testing
python scripts/enterprise_scale_50k_validation.py \
    --document-count 50000 \
    --techniques basic_rag hyde hybrid_ifind_rag
```

## 📊 Monitoring & Performance

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

### Health Checks

```python
# System health check
def health_check():
    checks = {
        'database': check_iris_connection(),
        'embeddings': check_embedding_service(),
        'vector_search': check_vector_operations(),
        'chunking': check_chunking_service()
    }
    return all(checks.values()), checks

# Run health check
is_healthy, status = health_check()
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

### API Security
- Implement authentication and authorization
- Use HTTPS for all communications
- Validate and sanitize all inputs
- Implement rate limiting

## 🚀 Production Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Database schema initialized
- [ ] Sample data loaded and validated
- [ ] Performance benchmarks completed
- [ ] Security configurations applied
- [ ] Monitoring systems configured

### Deployment
- [ ] Application deployed to production environment
- [ ] Database connections verified
- [ ] All 7 RAG techniques tested
- [ ] Enhanced chunking system validated
- [ ] Performance monitoring active
- [ ] Health checks passing

### Post-Deployment
- [ ] Load testing completed
- [ ] Performance metrics within acceptable ranges
- [ ] Error handling validated
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated
- [ ] Team training completed

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check IRIS container status
docker ps | grep iris

# Check connection
python -c "
from common.iris_connector import IRISConnector
conn = IRISConnector()
print('Connection successful' if conn.test_connection() else 'Connection failed')
"
```

#### Performance Issues
```bash
# Run performance diagnostics
python scripts/enterprise_validation_with_fixed_colbert.py --fast

# Check HNSW indexes
python scripts/verify_hnsw_indexes.py
```

#### Chunking Issues
```bash
# Test chunking service
python scripts/test_enhanced_chunking_simple.py

# Validate chunking performance
python scripts/enhanced_chunking_validation.py
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Enable HNSW indexing (Enterprise Edition)
CREATE INDEX idx_embeddings_hnsw ON SourceDocuments (embedding) 
USING HNSW WITH (m=16, ef_construction=200);

-- Optimize vector search performance
SET QUERY_TIMEOUT = 30;
SET VECTOR_SEARCH_CACHE = 1000;
```

### Application Optimization
```python
# Connection pooling
from common.iris_connector import IRISConnector

# Configure connection pool
connector = IRISConnector(
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

## 🔄 Maintenance

### Regular Maintenance Tasks
```bash
# Weekly performance check
python scripts/enterprise_validation_with_fixed_colbert.py --weekly-check

# Monthly full validation
python scripts/comprehensive_5000_doc_benchmark.py

# Quarterly scale testing
python scripts/enterprise_scale_50k_validation.py
```

### Backup and Recovery
```bash
# Database backup
iris backup /path/to/backup/

# Configuration backup
tar -czf config_backup.tar.gz config/ common/ *.yml *.json

# Recovery testing
python scripts/verify_real_data_testing.py --recovery-test
```

## 📞 Support and Resources

### Documentation
- **Main Documentation**: [`docs/INDEX.md`](docs/INDEX.md)
- **Technical Details**: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
- **Management Summary**: [`docs/MANAGEMENT_SUMMARY.md`](docs/MANAGEMENT_SUMMARY.md)

### Performance Benchmarks
- **Enterprise Validation**: [`ENTERPRISE_VALIDATION_COMPLETE.md`](ENTERPRISE_VALIDATION_COMPLETE.md)
- **Chunking Performance**: [`ENHANCED_CHUNKING_IMPLEMENTATION_COMPLETE.md`](ENHANCED_CHUNKING_IMPLEMENTATION_COMPLETE.md)
- **Hybrid iFind RAG**: [`HYBRID_IFIND_RAG_IMPLEMENTATION_COMPLETE.md`](HYBRID_IFIND_RAG_IMPLEMENTATION_COMPLETE.md)

### Contact Information
- **Technical Issues**: Check documentation and run diagnostic scripts
- **Performance Questions**: Review benchmark results and optimization guides
- **Enterprise Support**: Consult enterprise validation reports

## 🎯 Next Steps

### Immediate Actions
1. **Deploy development environment** using Docker setup
2. **Run validation scripts** to ensure all techniques work
3. **Load sample data** and test performance
4. **Configure monitoring** and health checks

### Production Readiness
1. **Scale testing** with enterprise validation scripts
2. **Security hardening** with production configurations
3. **Performance optimization** based on benchmark results
4. **Team training** on deployment and maintenance procedures

### Future Enhancements
1. **LLM Integration**: Connect to production language models
2. **API Development**: RESTful service endpoints
3. **UI Development**: User interface for RAG interactions
4. **Advanced Monitoring**: Real-time performance dashboards

This deployment guide provides a comprehensive foundation for successfully deploying the RAG Templates system in production environments, from small-scale development to enterprise-grade deployments.