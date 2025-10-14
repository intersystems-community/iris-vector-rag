# RAG Templates Framework - Complete Docker Deployment Guide

This directory contains a complete one-click deployment solution for the RAG Templates Framework, featuring all demos and services in a production-ready configuration.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- 8GB+ RAM (16GB recommended for full stack)
- 20GB+ free disk space

### One-Click Deployment

```bash
# 1. Clone and setup
git clone <repository-url>
cd rag-templates
make setup

# 2. Configure environment (edit with your API keys)
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations

# 3. Start the complete stack
make docker-up-dev        # Development mode with Jupyter
# OR
make docker-up-prod       # Production mode with Nginx
# OR  
make docker-quick         # Just core services

# 4. Initialize with sample data
make docker-init-data

# 5. Access the applications
open http://localhost:8501  # Streamlit RAG Demo
open http://localhost:8000/docs  # API Documentation
```

Your RAG framework is now running! 🎉

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Service Descriptions](#service-descriptions)
- [Deployment Profiles](#deployment-profiles)
- [Configuration Guide](#configuration-guide)
- [Network Architecture](#network-architecture)
- [Storage and Persistence](#storage-and-persistence)
- [Environment Variables](#environment-variables)
- [Management Commands](#management-commands)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Data Management](#data-management)
- [Security Considerations](#security-considerations)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)

## 🏗️ Architecture Overview

The RAG Templates Framework uses a microservices architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │────│  Streamlit App  │────│   Jupyter Lab   │
│   (Production)  │    │     (Demo)      │    │ (Development)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │    RAG API      │
                    │   (FastAPI)     │
                    └─────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  IRIS Database  │ │  Redis Cache    │ │   Monitoring    │
    │  (Vector Store) │ │   (Sessions)    │ │  (Prometheus)   │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Network Segmentation

- **Frontend Network**: Public-facing services (Nginx, Streamlit, Jupyter)
- **Backend Network**: Internal API communications
- **Data Network**: Database and cache layer (isolated)

## 🔧 Service Descriptions

### Core Services

#### IRIS Database (`iris_db`)
- **Purpose**: Vector database for embeddings and document storage
- **Image**: `intersystemsdc/iris-community:latest` (configurable)
- **Ports**: `1972` (SuperServer), `52773` (Management Portal)
- **Features**: Vector search, SQL interface, built-in analytics

#### Redis Cache (`redis`)
- **Purpose**: High-performance caching and session storage
- **Image**: `redis:7-alpine`
- **Port**: `6379`
- **Features**: Persistence, memory optimization, clustering support

#### RAG API (`rag_api`)
- **Purpose**: RESTful API for all RAG pipeline operations
- **Build**: Custom FastAPI application
- **Port**: `8000`
- **Features**: Multiple RAG pipelines, async processing, OpenAPI docs

#### Streamlit App (`streamlit_app`)
- **Purpose**: Interactive demo interface for RAG capabilities
- **Build**: Custom Streamlit application  
- **Port**: `8501`
- **Features**: Real-time queries, pipeline comparison, visualization

### Optional Services

#### Jupyter Notebook (`jupyter`)
- **Purpose**: Interactive development environment
- **Build**: Jupyter with RAG dependencies pre-installed
- **Port**: `8888`
- **Profiles**: `dev`, `with-data`

#### Nginx Proxy (`nginx`)
- **Purpose**: Reverse proxy, load balancing, SSL termination
- **Build**: Custom Nginx with optimized configuration
- **Ports**: `80`, `443`
- **Profiles**: `prod`

#### Health Monitoring (`monitoring`)
- **Purpose**: System health monitoring and alerting
- **Build**: Custom monitoring service
- **Port**: `9090`
- **Profiles**: `prod`, `monitoring`

#### Data Loader (`data_loader`)
- **Purpose**: Initialize database with sample data
- **Build**: Custom Python service
- **Profile**: `with-data`
- **Lifecycle**: Runs once then exits

## 🎯 Deployment Profiles

Docker Compose profiles allow you to start different combinations of services:

### Core Profile (`core`)
**Best for**: Basic RAG functionality testing
```bash
make docker-up  # or docker-compose --profile core up -d
```
**Services**: IRIS, Redis, RAG API, Streamlit
**Resources**: ~4GB RAM, 2 CPU cores

### Development Profile (`dev`)
**Best for**: Active development and experimentation
```bash
make docker-up-dev  # or docker-compose --profile dev up -d
```
**Services**: Core + Jupyter Notebook
**Resources**: ~6GB RAM, 4 CPU cores

### Production Profile (`prod`)
**Best for**: Production deployments
```bash
make docker-up-prod  # or docker-compose --profile prod up -d
```
**Services**: Core + Nginx + Monitoring
**Resources**: ~8GB RAM, 4 CPU cores

### With Data Profile (`with-data`)
**Best for**: Demo with pre-loaded sample data
```bash
make docker-up-data  # or docker-compose --profile with-data up -d
```
**Services**: Core + Data Loader
**Resources**: ~4GB RAM + additional storage

### Combined Profiles
```bash
# Development with monitoring
docker-compose --profile dev --profile monitoring up -d

# Production with data loading
docker-compose --profile prod --profile with-data up -d
```

## ⚙️ Configuration Guide

### Environment File (.env)

The `.env` file controls all configuration. Key sections:

#### Required Configuration
```bash
# AI API Keys (at least OpenAI is required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional

# Database Selection
IRIS_DOCKER_IMAGE=intersystemsdc/iris-community:latest  # Community
# IRIS_DOCKER_IMAGE=containers.intersystems.com/intersystems/iris-arm64:2025.1  # Enterprise

# Security (change defaults!)
REDIS_PASSWORD=change_me_to_secure_password
JUPYTER_TOKEN=change_me_to_secure_token
JWT_SECRET=change_me_to_long_random_string
```

#### Port Configuration
```bash
# Service Ports (change if conflicts exist)
RAG_API_PORT=8000
STREAMLIT_PORT=8501
JUPYTER_PORT=8888
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443
```

#### Performance Tuning
```bash
# API Service
API_WORKERS=4                    # Number of FastAPI workers
MAX_MEMORY_USAGE=2g             # Container memory limit

# Redis Configuration  
REDIS_MAXMEMORY=512mb           # Redis memory limit
REDIS_POOL_SIZE=10              # Connection pool size

# Database
DB_POOL_SIZE=10                 # IRIS connection pool
```

### Docker Compose Overrides

For custom configurations, create `docker-compose.override.yml`:

```yaml
# Example: Custom resource limits
services:
  rag_api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## 🌐 Network Architecture

### Network Isolation

The deployment uses three isolated networks:

#### Frontend Network (`rag_frontend`)
- **Subnet**: `172.20.0.0/16`
- **Purpose**: External access to user interfaces
- **Services**: Nginx, Streamlit, Jupyter

#### Backend Network (`rag_backend`)  
- **Subnet**: `172.21.0.0/16`
- **Purpose**: Internal API communication
- **Isolation**: Internal only (no external access)
- **Services**: RAG API, Streamlit, Monitoring

#### Data Network (`rag_data`)
- **Subnet**: `172.22.0.0/16`
- **Purpose**: Database layer communication
- **Isolation**: Internal only (maximum security)
- **Services**: IRIS, Redis, RAG API

### Security Benefits

1. **Defense in Depth**: Multiple network layers
2. **Principle of Least Privilege**: Services only access required networks
3. **Attack Surface Reduction**: Databases not directly accessible
4. **Traffic Segmentation**: Clear separation of concerns

## 💾 Storage and Persistence

### Named Volumes

All persistent data uses Docker named volumes:

```bash
# View all volumes
docker volume ls | grep rag

# Backup a volume
docker run --rm -v rag_iris_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/iris_backup.tar.gz -C /data .

# Restore a volume  
docker run --rm -v rag_iris_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/iris_backup.tar.gz -C /data
```

### Volume Details

| Volume | Purpose | Typical Size | Backup Priority |
|--------|---------|--------------|----------------|
| `rag_iris_data` | Database files, vectors | 1-50GB | **High** |
| `rag_redis_data` | Cache, sessions | 100MB-1GB | Medium |
| `rag_jupyter_data` | Notebooks, experiments | 100MB-5GB | Medium |
| `rag_nginx_logs` | Access logs | 10MB-1GB | Low |
| `rag_monitoring_data` | Metrics, alerts | 100MB-2GB | Low |

### Backup Strategy

```bash
# Automated backup
make docker-backup

# Manual selective backup
docker-compose exec iris_db iris backup

# Volume snapshot (if using Docker with snapshots)
docker run --rm -v rag_iris_data:/data -v /backup:/backup \
  alpine cp -a /data/. /backup/iris-$(date +%Y%m%d)/
```

## 📊 Management Commands

### Makefile Targets

The included Makefile provides convenient management commands:

#### Basic Operations
```bash
make help                    # Show all available commands
make setup                   # Initial environment setup
make docker-up              # Start core services
make docker-down            # Stop all services
make docker-restart         # Restart all services
```

#### Environment-Specific
```bash
make docker-up-dev          # Development environment
make docker-up-prod         # Production environment  
make docker-up-data         # With sample data
make docker-quick           # Fast core startup
```

#### Monitoring and Debugging
```bash
make docker-ps              # Show container status
make docker-logs            # View all logs
make docker-logs-api        # View specific service logs
make docker-health          # Run health checks
make docker-stats           # Show resource usage
```

#### Shell Access
```bash
make docker-shell           # Shell into API container
make docker-shell-iris      # Shell into specific container
make docker-iris-shell      # IRIS database shell
make docker-redis-shell     # Redis CLI
```

#### Data Management
```bash
make docker-init-data       # Load sample data
make docker-init-data-force # Force reload data
make docker-backup          # Create backup
```

#### Cleanup
```bash
make docker-clean           # Clean containers/networks
make docker-clean-all       # Clean everything (DESTRUCTIVE)
make docker-reset           # Reset to clean state
```

### Direct Docker Compose

You can also use Docker Compose directly:

```bash
# Start specific profile
docker-compose -f docker-compose.full.yml --profile dev up -d

# Scale a service
docker-compose -f docker-compose.full.yml up -d --scale rag_api=3

# View logs with timestamps
docker-compose -f docker-compose.full.yml logs -f -t rag_api

# Execute commands in containers
docker-compose -f docker-compose.full.yml exec rag_api python -c "import sys; print(sys.version)"
```

## 🔍 Monitoring and Health Checks

### Built-in Health Checks

Each service includes health check endpoints:

```bash
# API Health
curl http://localhost:8000/health

# Streamlit Health  
curl http://localhost:8501/_stcore/health

# IRIS Database
curl http://localhost:52773/csp/sys/UtilHome.csp
```

### Comprehensive Health Monitoring

```bash
# Run health check script
./scripts/docker/health-check.sh

# Continuous monitoring
./scripts/docker/health-check.sh --continuous

# JSON output for integration
./scripts/docker/health-check.sh --json
```

### Health Check Script Features

- **Service Status**: Container running state and health
- **HTTP Endpoints**: Response time and status codes
- **Database Connectivity**: Connection and query testing
- **System Resources**: Memory, disk, CPU usage
- **Redis Connectivity**: Cache availability and performance

### Monitoring Dashboard

When using the `prod` profile, access the monitoring dashboard at:
- **URL**: http://localhost:9090
- **Features**: Service metrics, alerts, performance graphs
- **Integration**: Prometheus-compatible metrics

### Log Aggregation

```bash
# View all logs
make docker-logs

# Follow specific service
make docker-logs-api

# Search logs
docker-compose logs rag_api 2>&1 | grep "ERROR"

# Log rotation (production)
docker-compose exec nginx logrotate /etc/logrotate.d/nginx
```

## 📚 Data Management

### Sample Data Loading

The framework includes scripts for loading sample datasets:

```bash
# Load small sample (10 docs) - fastest
./scripts/docker/init-data.sh --size small

# Load medium sample (50 docs) - good for demos
./scripts/docker/init-data.sh --size medium  

# Load large sample (full dataset) - comprehensive testing
./scripts/docker/init-data.sh --size large

# Force reload existing data
./scripts/docker/init-data.sh --force
```

### Data Sources

#### Small Dataset (`data/sample_10_docs/`)
- **Size**: ~10 medical documents
- **Type**: PMC (PubMed Central) articles
- **Use Case**: Quick testing, development
- **Load Time**: ~30 seconds

#### Medium Dataset (`data/downloaded_pmc_docs/`)
- **Size**: ~50 medical documents  
- **Type**: Recent PMC articles
- **Use Case**: Demo presentations, moderate testing
- **Load Time**: ~5 minutes

#### Large Dataset (Full PMC sample)
- **Size**: 500+ documents
- **Type**: Comprehensive medical literature
- **Use Case**: Performance testing, production simulation
- **Load Time**: ~30 minutes

### Custom Data Loading

```python
# Load your own documents via API
import requests

def load_document(title, content, metadata=None):
    response = requests.post(
        "http://localhost:8000/api/v1/documents",
        json={
            "title": title,
            "content": content,
            "metadata": metadata or {}
        }
    )
    return response.json()

# Load from file
with open("my_document.txt", "r") as f:
    content = f.read()
    result = load_document("My Document", content)
```

### Database Schema

The RAG system creates these tables automatically:

```sql
-- Document storage
CREATE TABLE rag_documents (
    id INTEGER IDENTITY PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE,
    title VARCHAR(500),
    content TEXT,
    embedding VECTOR(DOUBLE, 1536),
    metadata JSON,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk storage for retrieval
CREATE TABLE rag_chunks (
    id INTEGER IDENTITY PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_id VARCHAR(255) UNIQUE,
    content TEXT,
    embedding VECTOR(DOUBLE, 1536),
    chunk_index INTEGER,
    metadata JSON,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query history
CREATE TABLE rag_queries (
    id INTEGER IDENTITY PRIMARY KEY,
    query_id VARCHAR(255) UNIQUE,
    query_text TEXT,
    query_embedding VECTOR(DOUBLE, 1536),
    response TEXT,
    retrieved_docs JSON,
    metadata JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Vector Indexes

```sql
-- Optimized vector search indexes
CREATE INDEX idx_docs_embedding ON rag_documents USING VECTOR(embedding);
CREATE INDEX idx_chunks_embedding ON rag_chunks USING VECTOR(embedding);
CREATE INDEX idx_queries_embedding ON rag_queries USING VECTOR(embedding);
```

## 🔒 Security Considerations

### Default Security Measures

1. **Network Isolation**: Multi-tier network architecture
2. **Non-Root Containers**: All services run as non-privileged users
3. **Secret Management**: Environment-based configuration
4. **Health Checks**: Automated failure detection
5. **Resource Limits**: Memory and CPU constraints

### Production Security Checklist

#### Secrets and Credentials
- [ ] Change all default passwords (`REDIS_PASSWORD`, `JUPYTER_TOKEN`)
- [ ] Use strong, unique API keys
- [ ] Rotate credentials regularly
- [ ] Consider using Docker secrets for sensitive data

#### Network Security
- [ ] Configure firewall rules for production
- [ ] Use HTTPS in production (SSL certificates)
- [ ] Restrict access to management interfaces
- [ ] Consider VPN for development access

#### Container Security
- [ ] Scan images for vulnerabilities
- [ ] Keep base images updated
- [ ] Use specific image tags (not `latest`)
- [ ] Regular security updates

#### Access Control
```bash
# Limit container capabilities
docker-compose.yml:
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - SETGID
    - SETUID
```

#### Monitoring and Auditing
- [ ] Enable audit logging
- [ ] Monitor for suspicious activities  
- [ ] Set up alerts for failures
- [ ] Regular backup verification

### SSL/TLS Configuration

For production HTTPS, update nginx configuration:

```nginx
# /docker/config/nginx/default.conf
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/your-cert.crt;
    ssl_certificate_key /etc/nginx/ssl/your-key.key;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

## ⚡ Performance Tuning

### Resource Allocation

#### Memory Optimization
```bash
# .env configuration
API_WORKERS=4                    # Based on available CPU cores
REDIS_MAXMEMORY=1g              # Adjust based on cache needs
MAX_MEMORY_USAGE=4g             # Per-container limit

# Docker Compose resource limits
services:
  rag_api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

#### CPU Optimization
```bash
# API Workers = CPU cores for CPU-bound tasks
API_WORKERS=${CPU_CORES}

# Redis Threading
REDIS_IO_THREADS=4              # For multi-core systems

# IRIS Configuration
IRIS_SHARED_MEMORY=1G           # Increase for large datasets
```

### Database Performance

#### IRIS Optimization
```bash
# Connect to IRIS and optimize
make docker-iris-shell

# Increase shared memory
SET ^%SYS("EXTMEM")=1000000

# Optimize vector search
SET ^%SYS("VECTORSEARCH","CACHE")=500000
```

#### Redis Performance
```bash
# Memory optimization in redis.conf
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence tuning
save 900 1
save 300 10
save 60 10000
```

### Application Performance

#### FastAPI Optimization
```python
# In API service configuration
import uvicorn

uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    workers=4,                   # Multi-process
    worker_class="uvicorn.workers.UvicornWorker",
    max_requests=1000,          # Worker recycling
    max_requests_jitter=50,
    keepalive_timeout=65
)
```

#### Caching Strategy
```python
# Redis caching for expensive operations
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Monitoring Performance

```bash
# Container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/api/v1/query

# Database performance
make docker-iris-shell
# SQL> SELECT TOP 10 * FROM %SYS.ProcessQuery ORDER BY TimeExecuted DESC
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Service Won't Start

**Problem**: Container exits immediately
```bash
# Check logs for error details
make docker-logs-<service>

# Common causes and solutions:
# 1. Port conflict
netstat -tlnp | grep :8000  # Check if port is in use
# Solution: Change port in .env

# 2. Missing environment variables
docker-compose config  # Validate configuration
# Solution: Check .env file

# 3. Permission issues
ls -la logs/  # Check directory permissions
# Solution: chmod 755 logs/
```

#### Database Connection Errors

**Problem**: Cannot connect to IRIS
```bash
# Check IRIS container status
make docker-logs-iris

# Test database connectivity
docker-compose exec iris_db iris session iris -U%SYS

# Common solutions:
# 1. Wait for IRIS to fully start (can take 60-120 seconds)
# 2. Check if running community vs enterprise image
# 3. Verify network connectivity
docker-compose exec rag_api ping iris_db
```

#### Out of Memory Errors

**Problem**: Containers being killed (OOMKilled)
```bash
# Check container resource usage
docker stats

# Solutions:
# 1. Increase Docker memory limit
# 2. Reduce service resource usage in .env
API_WORKERS=2  # Reduce from 4
REDIS_MAXMEMORY=256mb  # Reduce from 512mb

# 3. Close unnecessary applications
```

#### Slow Performance

**Problem**: RAG queries taking too long
```bash
# Check system resources
make docker-stats

# Profile API performance
curl -w "@curl-format.txt" http://localhost:8000/health

# Solutions:
# 1. Increase cache memory
REDIS_MAXMEMORY=1g

# 2. Optimize database
make docker-iris-shell
# Run: KILL ^%SYS.Task  // Clear background tasks

# 3. Reduce batch size
DATA_BATCH_SIZE=50  # From 100
```

#### SSL/Certificate Issues

**Problem**: HTTPS not working in production
```bash
# Check certificate files
ls -la docker/ssl/

# Verify certificate validity
openssl x509 -in docker/ssl/nginx-selfsigned.crt -text -noout

# Test SSL configuration
openssl s_client -connect localhost:443 -servername localhost

# Solutions:
# 1. Regenerate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/nginx-selfsigned.key \
  -out docker/ssl/nginx-selfsigned.crt

# 2. Use Let's Encrypt for production
# 3. Check nginx configuration
```

### Health Check Debugging

```bash
# Run comprehensive health check
./scripts/docker/health-check.sh --verbose

# Check specific endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8501/_stcore/health

# Check database connectivity
docker-compose exec iris_db iris session iris -U%SYS
# Run: write $SYSTEM.SQL.Execute("SELECT 1").%Display()

# Check Redis
docker-compose exec redis redis-cli ping
```

### Log Analysis

```bash
# API errors
make docker-logs-api | grep ERROR

# Database issues
make docker-logs-iris | grep -i error

# Performance issues
make docker-logs | grep -i "timeout\|slow\|performance"

# Security issues
make docker-logs | grep -i "unauthorized\|forbidden\|attack"
```

### Recovery Procedures

#### Corrupted Database
```bash
# Stop services
make docker-down

# Restore from backup
docker run --rm -v rag_iris_data:/data -v $(pwd)/backups:/backup alpine \
  tar xzf /backup/iris_backup_latest.tar.gz -C /data

# Restart services
make docker-up
```

#### Reset Everything
```bash
# Nuclear option - destroys all data
make docker-clean-all

# Start fresh
make setup
make docker-up-dev
make docker-init-data
```

## 🔧 Development Workflow

### Local Development Setup

```bash
# 1. Initial setup
git clone <repository>
cd rag-templates
make setup

# 2. Start development environment
make docker-up-dev

# 3. Access development tools
open http://localhost:8888  # Jupyter Lab
open http://localhost:8501  # Streamlit App
open http://localhost:8000/docs  # API Docs
```

### Code Development Cycle

1. **Edit Code**: Modify files in your IDE
2. **Test Changes**: Use Jupyter for quick tests
3. **Rebuild Service**: `make docker-build` (if needed)
4. **Restart Service**: `make docker-restart-api`
5. **Validate**: Check health and run tests

### Debugging Applications

```bash
# Enter API container for debugging
make docker-shell-api

# Check Python environment
python -c "import sys; print(sys.path)"
pip list | grep -i rag

# Debug FastAPI
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn main:app

# Debug with IPython
python -c "import IPython; IPython.embed()"
```

### Hot Reloading

For active development, enable hot reloading:

```bash
# Enable development mode in .env
FASTAPI_ENV=development
API_DEBUG=true
API_RELOAD=true

# Restart API service
make docker-restart-api
```

Changes to Python files will automatically reload the API service.

### Running Tests

```bash
# Basic functionality tests
make docker-test

# Specific API tests
docker-compose exec rag_api python -m pytest tests/

# Integration tests
./scripts/docker/health-check.sh --verbose

# Load testing
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "pipeline": "basic"}'
```

## 🚀 Production Deployment

### Pre-Deployment Checklist

#### Security
- [ ] All default passwords changed
- [ ] API keys configured and secured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Access controls implemented

#### Performance
- [ ] Resource limits set appropriately
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Log rotation configured
- [ ] Performance testing completed

#### Configuration
- [ ] Environment variables reviewed
- [ ] Production profile tested
- [ ] Health checks validated
- [ ] Networking configured
- [ ] DNS/domains configured

### Production Deployment Steps

1. **Prepare Environment**
```bash
# Clone to production server
git clone <repository> /opt/rag-templates
cd /opt/rag-templates

# Setup production environment
cp .env.example .env
# Edit .env with production values

# Set production mode
echo "ENVIRONMENT=production" >> .env
echo "DEBUG=false" >> .env
```

2. **Deploy Services**
```bash
# Start production stack
make docker-up-prod

# Wait for health
make docker-wait

# Initialize with data (optional)
make docker-init-data
```

3. **Configure Reverse Proxy**
```nginx
# /etc/nginx/sites-available/rag-templates
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/private.key;
    
    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

4. **Setup Monitoring**
```bash
# Configure external monitoring
# Point your monitoring system to:
# - http://your-domain:9090 (Prometheus metrics)
# - Health check endpoint: http://your-domain/health

# Setup log aggregation
# Configure your log aggregation system to collect from:
# - /var/lib/docker/containers/*/

# Setup alerting
# Configure alerts for:
# - Service health failures
# - Resource utilization
# - Error rates
```

5. **Backup Configuration**
```bash
# Setup automated backups
cat > /etc/cron.d/rag-backup << EOF
0 2 * * * root cd /opt/rag-templates && make docker-backup
0 3 * * 0 root cd /opt/rag-templates && make docker-clean
EOF

# Test backup/restore
make docker-backup
# Verify backup files in backups/
```

### Production Monitoring

#### Health Monitoring
```bash
# Continuous health monitoring
./scripts/docker/health-check.sh --continuous --json | \
  jq -r 'select(.overall_status != "healthy") | @base64' | \
  base64 --decode | \
  logger -t rag-health

# Integration with monitoring systems
curl -s http://localhost:9090/metrics  # Prometheus metrics
./scripts/docker/health-check.sh --json  # Health status JSON
```

#### Performance Monitoring
```bash
# Resource usage tracking
docker stats --format "json" | \
  jq '{time: now, containers: [.]}' | \
  logger -t rag-stats

# API performance monitoring
curl -w "@curl-format.txt" -s -o /dev/null \
  http://localhost:8000/api/v1/query \
  -d '{"query": "health check", "pipeline": "basic"}'
```

#### Log Management
```bash
# Configure log rotation
cat > /etc/logrotate.d/rag-templates << EOF
/opt/rag-templates/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# Centralized logging
journalctl -f -u docker | grep rag-templates
```

### Scaling Considerations

#### Horizontal Scaling
```yaml
# docker-compose.scale.yml
services:
  rag_api:
    deploy:
      replicas: 3
      
  nginx:
    volumes:
      - ./nginx-upstream.conf:/etc/nginx/conf.d/upstream.conf
```

#### Database Scaling
```bash
# IRIS clustering for high availability
# Configure in IRIS Management Portal:
# - Mirror configuration
# - ECP connections
# - Load balancing
```

#### Cache Scaling
```yaml
# Redis Cluster configuration
services:
  redis:
    deploy:
      replicas: 3
    command: redis-server --cluster-enabled yes
```

---

## 📝 Additional Resources

### Documentation Links
- [IRIS Vector Search Documentation](https://docs.intersystems.com/iris/latest/csp/docbook/DocBook.UI.Page.cls)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Support and Community
- **Issues**: Report bugs and feature requests in the GitHub repository
- **Discussions**: Join community discussions for questions and sharing
- **Documentation**: Contribute to documentation improvements

### Contributing
- Fork the repository
- Create feature branches
- Submit pull requests
- Follow the contributing guidelines

---

*This documentation is maintained as part of the RAG Templates Framework. For the latest updates, please check the repository.*