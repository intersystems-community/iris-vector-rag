# Docker Restart with 2TB Limit - 100K Ingestion Resume Guide

## Current Status Summary

✅ **Checkpoint Status Confirmed:**
- **Current Progress:** 11,829 documents successfully ingested (11.8% complete)
- **Target:** 100,000 documents
- **Remaining:** 88,171 documents to process
- **Files Processed:** 14,000 XML files processed successfully
- **No Failed Files:** 0 errors in processing
- **Schema:** RAG (standard schema)
- **Last Checkpoint:** 2025-05-26 16:25:05

✅ **Space Analysis:**
- **Current Docker Usage:** ~49.7GB total (Images: 24.1GB, Containers: 16.3GB, Volumes: 9.3GB)
- **Estimated Data Processed:** ~5.8GB
- **Estimated Remaining Data:** ~43.1GB
- **Total Space Needed:** ~48.8GB for documents + Docker overhead = **~100GB minimum**

## Docker Restart Instructions

### Step 1: Stop Current Docker Services

```bash
# Stop the current ingestion (if still running)
docker-compose down

# Optional: Clean up unused Docker resources to free space
docker system prune -f
docker volume prune -f
```

### Step 2: Configure Docker with 2TB Limit

#### For Docker Desktop (macOS/Windows):

1. **Open Docker Desktop Settings:**
   - Click Docker Desktop icon → Settings/Preferences
   - Go to "Resources" → "Advanced"

2. **Increase Storage Limits:**
   - **Disk Image Size:** Set to `2000 GB` (2TB)
   - **Memory:** Ensure at least `8 GB` allocated
   - **CPUs:** Ensure at least `4 CPUs` allocated

3. **Apply Changes:**
   - Click "Apply & Restart"
   - Wait for Docker to restart completely

#### For Docker Engine (Linux):

1. **Edit Docker Daemon Configuration:**
```bash
sudo nano /etc/docker/daemon.json
```

2. **Add Storage Configuration:**
```json
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.size=2T"
  ],
  "data-root": "/var/lib/docker",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

3. **Restart Docker Service:**
```bash
sudo systemctl restart docker
sudo systemctl status docker  # Verify it's running
```

### Step 3: Verify Docker Configuration

```bash
# Check Docker info for storage driver and space
docker info | grep -E "(Storage Driver|Data Space)"

# Verify available space
df -h /var/lib/docker  # Linux
# or check Docker Desktop settings for macOS/Windows

# Check system disk space
df -h
```

### Step 4: Restart RAG Services

```bash
# Navigate to project directory
cd /Users/tdyar/ws/rag-templates

# Start IRIS database
docker-compose up -d

# Wait for IRIS to be ready (check logs)
docker-compose logs -f iris

# Verify database connection
python3 -c "
from common.iris_connector import get_iris_connection
conn = get_iris_connection()
if conn:
    print('✅ Database connection successful')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments')
    count = cursor.fetchone()[0]
    print(f'📊 Current document count: {count:,}')
    cursor.close()
    conn.close()
else:
    print('❌ Database connection failed')
"
```

## Resume 100K Ingestion

### Resume Command

```bash
# Resume from checkpoint with optimized settings
python3 scripts/ingest_100k_documents.py \
    --resume-from-checkpoint \
    --batch-size 500 \
    --target-docs 100000

# Alternative: Monitor with logs
nohup python3 scripts/ingest_100k_documents.py \
    --resume-from-checkpoint \
    --batch-size 500 \
    --target-docs 100000 > ingest_resume.log 2>&1 &

# Monitor progress
tail -f ingest_resume.log
```

### Monitoring Commands

```bash
# Check checkpoint status
python3 check_checkpoint.py

# Monitor Docker resource usage
docker stats

# Monitor disk space
watch -n 30 'df -h | grep -E "(Filesystem|/var/lib/docker|/$)"'

# Monitor ingestion logs
tail -f ingest_100k_documents.log
```

## Expected Performance

Based on current progress:
- **Processing Rate:** ~3 documents/second average
- **Estimated Completion Time:** ~8-10 hours for remaining 88,171 documents
- **Peak Memory Usage:** ~8-12GB during processing
- **Storage Growth:** ~43GB additional space needed

## Troubleshooting

### If Docker Fails to Start:
```bash
# Check Docker daemon logs
journalctl -u docker.service -f  # Linux
# or check Docker Desktop logs in GUI

# Reset Docker if needed (CAUTION: This removes all containers/images)
docker system prune -a --volumes
```

### If Space Issues Persist:
```bash
# Clean up old Docker data
docker system df
docker system prune -a --volumes

# Check for large files
du -sh /var/lib/docker/*  # Linux
```

### If Ingestion Fails to Resume:
```bash
# Verify checkpoint file exists
ls -la ingestion_checkpoint.pkl

# Check database connectivity
python3 -c "from common.iris_connector import get_iris_connection; print('✅' if get_iris_connection() else '❌')"

# Restart with fresh checkpoint if needed (CAUTION: Loses progress)
# python3 scripts/ingest_100k_documents.py --target-docs 100000 --batch-size 500
```

## Success Verification

After Docker restart and before resuming:

1. ✅ Docker has 2TB storage limit configured
2. ✅ IRIS database is running and accessible
3. ✅ Current document count matches checkpoint (11,829)
4. ✅ Sufficient disk space available (>100GB free)
5. ✅ Checkpoint file exists and is readable

## Post-Completion Actions

Once 100K ingestion completes:

```bash
# Verify final count
python3 check_checkpoint.py

# Generate completion report
python3 -c "
from common.iris_connector import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments')
total = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
with_embeddings = cursor.fetchone()[0]
print(f'🎯 Final Results:')
print(f'Total Documents: {total:,}')
print(f'With Embeddings: {with_embeddings:,}')
print(f'Success Rate: {(with_embeddings/total)*100:.1f}%')
cursor.close()
conn.close()
"

# Clean up checkpoint file
mv ingestion_checkpoint.pkl ingestion_checkpoint_100k_complete.pkl
```

---

**Ready to proceed with Docker restart and ingestion resume!**