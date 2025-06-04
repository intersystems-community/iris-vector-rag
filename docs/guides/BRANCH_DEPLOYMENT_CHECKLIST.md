# Branch Deployment Checklist for Remote Server

## Pre-Deployment Verification

### 1. Local Branch Status
```bash
# Check current branch
git branch --show-current

# Verify all changes are committed
git status

# Check recent commits include vector migration work
git log --oneline -10

# Verify key files are present
ls scripts/remote_setup.sh
ls scripts/verify_native_vector_schema.py
ls scripts/system_health_check.py
ls VECTOR_MIGRATION_COMPLETE_SUMMARY.md
```

### 2. Push Branch to Remote Repository
```bash
# Push current branch to remote
git push origin <current-branch-name>

# Verify branch is available remotely
git ls-remote --heads origin
```

## Remote Server Deployment

### 1. Clone and Checkout Correct Branch
```bash
# On remote server
git clone <your-repo-url> rag-templates
cd rag-templates

# List available branches
git branch -r

# Checkout the vector migration branch
git checkout <vector-migration-branch-name>

# Verify you're on the correct branch
git branch --show-current
git log --oneline -5
```

### 2. Verify Branch Contents
```bash
# Check that all new files are present
ls scripts/remote_setup.sh
ls scripts/verify_native_vector_schema.py
ls scripts/system_health_check.py
ls VECTOR_MIGRATION_COMPLETE_SUMMARY.md
ls REMOTE_DEPLOYMENT_GUIDE.md

# Verify setup script is executable
ls -la scripts/remote_setup.sh
```

### 3. Run Deployment
```bash
# Make setup script executable (if needed)
chmod +x scripts/remote_setup.sh

# Run the setup
./scripts/remote_setup.sh

# The script will automatically detect the branch and proceed
```

## Post-Deployment Verification

### 1. System Health Check
```bash
python3 scripts/system_health_check.py
```

### 2. Schema Verification
```bash
python3 scripts/verify_native_vector_schema.py
```

### 3. Performance Baseline
```bash
python3 scripts/create_performance_baseline.py
```

## Common Issues and Solutions

### Issue: "Branch not found"
**Solution**: Ensure the branch is pushed to the remote repository
```bash
# On local machine
git push origin <branch-name>
```

### Issue: "Files not found after checkout"
**Solution**: Verify you're on the correct branch
```bash
git branch --show-current
git pull origin <branch-name>
```

### Issue: "Permission denied on setup script"
**Solution**: Make script executable
```bash
chmod +x scripts/remote_setup.sh
```

### Issue: "Docker containers not starting"
**Solution**: Check system resources and Docker status
```bash
docker --version
docker-compose --version
free -h
df -h
```

## Branch-Specific Files Created

The following files are specific to the vector migration branch:

### Core Deployment
- `scripts/remote_setup.sh` - Automated setup script
- `REMOTE_DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `VECTOR_MIGRATION_COMPLETE_SUMMARY.md` - Migration summary
- `BRANCH_DEPLOYMENT_CHECKLIST.md` - This checklist

### Verification and Monitoring
- `scripts/verify_native_vector_schema.py` - Schema verification
- `scripts/system_health_check.py` - System health monitoring
- `scripts/create_performance_baseline.py` - Performance baseline
- `scripts/setup_monitoring.py` - Monitoring setup

### Migration Research (Preserved)
- `scripts/migrate_sourcedocuments_native_vector.py` - Original migration script
- `objectscript/RAG.VectorMigration.cls` - ObjectScript utilities
- Various testing and debugging scripts

## Success Criteria

✅ Branch successfully deployed to remote server
✅ All new files present and accessible
✅ Setup script runs without errors
✅ System health check passes
✅ Native VECTOR schema verified
✅ Performance baseline created
✅ Ready for data ingestion and benchmarking

## Next Steps After Successful Deployment

1. **Start with small-scale testing**
2. **Verify performance with sample data**
3. **Scale up to full dataset**
4. **Run comprehensive benchmarks**
5. **Monitor system performance**

This checklist ensures that the branch-specific deployment is successful and all components are properly configured for the native VECTOR implementation.