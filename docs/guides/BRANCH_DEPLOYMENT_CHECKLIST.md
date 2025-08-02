# Branch Deployment Checklist

## Overview

This checklist ensures safe and reliable deployment of branches in the RAG Templates project. It covers pre-deployment verification, deployment execution, and post-deployment validation steps.

## Pre-Deployment Verification

### 1. Local Branch Status
```bash
# Check current branch
git branch --show-current

# Verify all changes are committed
git status

# Check recent commits
git log --oneline -10

# Verify no uncommitted changes
git diff --exit-code
git diff --cached --exit-code
```

### 2. Code Quality Checks
```bash
# Run linting
make lint

# Run code formatting check
make format

# Run unit tests
make test-unit

# Run integration tests
make test-integration
```

### 3. Configuration Validation
```bash
# Validate configuration files
./ragctl config --validate

# Check for required configuration files
ls config/config.yaml
ls config/default.yaml
ls config/pipelines.yaml

# Verify environment variables are set
echo "IRIS_HOST: ${IRIS_HOST:-localhost}"
echo "IRIS_PORT: ${IRIS_PORT:-1972}"
echo "IRIS_NAMESPACE: ${IRIS_NAMESPACE:-USER}"
```

### 4. Dependency Verification
```bash
# Check Python environment
python --version
pip list | grep -E "(iris|sentence|transformers)"

# Verify Docker setup
docker --version
docker-compose --version
docker info

# Check system resources
free -h
df -h
```

### 5. Push Branch to Remote Repository
```bash
# Push current branch to remote
git push origin $(git branch --show-current)

# Verify branch is available remotely
git ls-remote --heads origin | grep $(git branch --show-current)
```

## Deployment Execution

### 1. Environment Setup
```bash
# Set deployment environment variables
export DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-staging}
export DEPLOYMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create deployment log directory
mkdir -p logs/deployment_${DEPLOYMENT_TIMESTAMP}
```

### 2. Database Preparation
```bash
# Backup current database state (if applicable)
python scripts/utilities/backup_iris_while_running.py

# Test database connectivity
make test-dbapi

# Verify database schema
python -c "
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments')
print(f'Documents: {cursor.fetchone()[0]}')
cursor.close()
conn.close()
"
```

### 3. Docker Container Management
```bash
# Check current container status
docker-compose ps

# Pull latest images if needed
docker-compose pull

# Restart containers with new configuration
docker-compose down
docker-compose up -d

# Wait for containers to be healthy
timeout 300 bash -c 'until docker-compose ps | grep -q "healthy"; do sleep 5; done'
```

### 4. Application Deployment
```bash
# Install/update dependencies
make install

# Initialize database schema
make setup-db

# Run pipeline validation
make validate-all-pipelines

# Auto-setup missing components
make auto-setup-all
```

## Post-Deployment Verification

### 1. System Health Checks
```bash
# Run comprehensive health check
python iris_rag/monitoring/health_monitor.py

# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"

# Verify Docker containers
docker-compose ps
docker-compose logs --tail=50
```

### 2. Database Validation
```bash
# Test database connectivity
make test-dbapi

# Verify data integrity
python -c "
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()

# Check table counts
tables = ['RAG.SourceDocuments', 'RAG.DocumentChunks', 'RAG.DocumentTokenEmbeddings']
for table in tables:
    try:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        count = cursor.fetchone()[0]
        print(f'{table}: {count:,} rows')
    except Exception as e:
        print(f'{table}: ERROR - {e}')

cursor.close()
conn.close()
"

# Test vector operations
python scripts/utilities/test_correct_vector_syntax_fixed.py
```

### 3. Pipeline Functionality Tests
```bash
# Test basic pipeline
./ragctl run --pipeline basic --query "What is machine learning?" --dry-run

# Test all pipeline types
for pipeline in basic colbert crag hyde graphrag noderag hybrid_ifind; do
    echo "Testing $pipeline pipeline..."
    ./ragctl validate --pipeline $pipeline || echo "⚠️ $pipeline validation failed"
done

# Run comprehensive end-to-end test
make test-1000
```

### 4. Performance Baseline
```bash
# Run performance benchmarks
python scripts/utilities/enhanced_benchmark_runner.py

# Monitor system performance
python iris_rag/monitoring/performance_monitor.py --duration 300

# Check memory usage patterns
python -c "
import time
import psutil
for i in range(5):
    mem = psutil.virtual_memory()
    print(f'Memory usage: {mem.percent}% ({mem.used/1024/1024/1024:.1f}GB used)')
    time.sleep(10)
"
```

### 5. Configuration Verification
```bash
# Verify configuration loading
./ragctl config --show

# Test reconciliation framework
python -c "
from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.reconciliation import ReconciliationController

config = ConfigurationManager()
controller = ReconciliationController(config)
status = controller.get_system_status()
print(f'Reconciliation status: {status}')
"

# Validate environment-specific settings
python -c "
import os
print('Environment variables:')
for key, value in os.environ.items():
    if key.startswith('RAG_') or key.startswith('IRIS_'):
        print(f'  {key}={value}')
"
```

## Rollback Procedures

### 1. Emergency Rollback
```bash
# Stop current deployment
docker-compose down

# Restore previous container state
docker-compose up -d

# Restore database backup (if needed)
# python scripts/utilities/restore_iris_backup.py --backup-file <backup_file>

# Verify rollback success
make test-dbapi
./ragctl validate
```

### 2. Gradual Rollback
```bash
# Disable new features
export RAG_FEATURE_FLAGS_NEW_FEATURES=false

# Restart with previous configuration
docker-compose restart

# Monitor system stability
python iris_rag/monitoring/health_monitor.py --continuous --duration 600
```

## Common Issues and Solutions

### Issue: "Docker containers not starting"
**Diagnosis:**
```bash
docker-compose logs
docker system df
docker system prune -f
```
**Solution:**
```bash
# Check system resources
free -h
df -h

# Clean up Docker resources
docker system prune -f
docker volume prune -f

# Restart Docker daemon (if needed)
sudo systemctl restart docker
```

### Issue: "Database connection failed"
**Diagnosis:**
```bash
# Check IRIS container status
docker-compose ps iris_db

# Check IRIS logs
docker-compose logs iris_db

# Test network connectivity
telnet localhost 1972
```
**Solution:**
```bash
# Restart IRIS container
docker-compose restart iris_db

# Wait for health check
timeout 300 bash -c 'until docker-compose ps iris_db | grep -q "healthy"; do sleep 5; done'

# Verify connection
make test-dbapi
```

### Issue: "Pipeline validation failed"
**Diagnosis:**
```bash
# Check specific pipeline status
./ragctl validate --pipeline <pipeline_name> --verbose

# Check embedding table status
python scripts/utilities/validation/embedding_validation_system.py
```
**Solution:**
```bash
# Auto-fix pipeline issues
make auto-setup-pipeline PIPELINE=<pipeline_name>

# Regenerate embeddings if needed
python scripts/utilities/populate_token_embeddings.py

# Verify fix
./ragctl validate --pipeline <pipeline_name>
```

### Issue: "Performance degradation"
**Diagnosis:**
```bash
# Monitor system resources
python iris_rag/monitoring/performance_monitor.py --duration 300

# Check database performance
python scripts/utilities/investigate_vector_indexing_reality.py

# Analyze query performance
python scripts/utilities/test_current_performance_with_workaround.py
```
**Solution:**
```bash
# Optimize database indexes
python scripts/utilities/setup_colbert_hnsw_optimization.py

# Clear caches
python -c "
from common.llm_cache_manager import get_global_cache_manager
cache = get_global_cache_manager()
if cache:
    cache.clear()
    print('Cache cleared')
"

# Restart services
docker-compose restart
```

## Success Criteria

### Deployment Success Indicators
- ✅ All Docker containers running and healthy
- ✅ Database connectivity established
- ✅ All pipeline types validate successfully
- ✅ System health checks pass
- ✅ Performance metrics within acceptable ranges
- ✅ No critical errors in logs
- ✅ Configuration loaded correctly
- ✅ Reconciliation framework operational

### Performance Benchmarks
- ✅ Query response time < 5 seconds for basic operations
- ✅ Memory usage < 80% of available RAM
- ✅ CPU usage < 70% under normal load
- ✅ Database operations complete without timeouts
- ✅ Vector search performance within expected ranges

### Data Integrity Checks
- ✅ Document count matches expected values
- ✅ Embedding tables populated correctly
- ✅ Vector operations function properly
- ✅ No data corruption detected
- ✅ Backup and restore procedures tested

## Post-Deployment Actions

### 1. Documentation Updates
```bash
# Update deployment log
echo "Deployment completed: $(date)" >> logs/deployment_${DEPLOYMENT_TIMESTAMP}/deployment.log

# Document configuration changes
git log --oneline --since="1 day ago" > logs/deployment_${DEPLOYMENT_TIMESTAMP}/changes.log

# Update system documentation
# (Manual step: Update relevant documentation files)
```

### 2. Monitoring Setup
```bash
# Enable continuous monitoring
python iris_rag/monitoring/health_monitor.py --continuous &

# Set up alerting (if configured)
python iris_rag/monitoring/metrics_collector.py --start-collection

# Schedule regular health checks
# (Add to cron or monitoring system)
```

### 3. Team Notification
```bash
# Generate deployment report
python -c "
import json
from datetime import datetime

report = {
    'deployment_time': datetime.now().isoformat(),
    'environment': '${DEPLOYMENT_ENV}',
    'branch': '$(git branch --show-current)',
    'commit': '$(git rev-parse HEAD)',
    'status': 'SUCCESS'
}

with open('logs/deployment_${DEPLOYMENT_TIMESTAMP}/report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Deployment report generated')
"

# Send notifications (implement as needed)
# slack/email/webhook notifications
```

## Next Steps After Successful Deployment

1. **Monitor System Performance**
   - Watch system metrics for 24-48 hours
   - Review logs for any unusual patterns
   - Validate user-facing functionality

2. **Gradual Traffic Increase**
   - Start with limited user access
   - Gradually increase load
   - Monitor performance under increased usage

3. **Data Validation**
   - Verify data integrity over time
   - Check for any data drift or corruption
   - Validate embedding quality

4. **Performance Optimization**
   - Analyze performance metrics
   - Optimize based on real usage patterns
   - Tune configuration parameters

5. **Documentation and Training**
   - Update operational documentation
   - Train team on new features/changes
   - Document lessons learned

This comprehensive checklist ensures reliable and safe branch deployments while maintaining system integrity and performance.