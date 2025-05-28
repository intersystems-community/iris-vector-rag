# 100K Document Ingestion - Monitoring Guide

## 🚀 INGESTION STATUS: RUNNING SUCCESSFULLY

**Started:** May 27, 2025 at 6:14 PM  
**Process ID:** 22285  
**Configuration:** Conservative optimized (8 docs/batch, 150 tokens/batch)  
**Target:** 100,000 PMC documents from `data/pmc_100k_downloaded/`

## 📊 Current Progress

- **Documents Processed:** 2,680+ (as of 6:15 PM)
- **Processing Rate:** ~17.8 documents/second
- **Batch Size:** 8 documents (conservative for stability)
- **Token Batch Size:** 150 tokens (conservative for performance)
- **Status:** File processing complete, database loading in progress

## 🔍 Monitoring Commands

### Quick Status Check
```bash
python monitor_100k_ingestion.py
```

### Continuous Monitoring (updates every 30 seconds)
```bash
python monitor_100k_ingestion.py --continuous
```

### Check Process Status
```bash
ps aux | grep run_conservative_ingestion | grep -v grep
```

### View Live Logs
```bash
tail -f logs/conservative_ingestion_20250527_181422.log
```

### Check Database Counts
```bash
python -c "
from common.iris_connector import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments')
doc_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings')
token_count = cursor.fetchone()[0]
print(f'Documents: {doc_count:,}')
print(f'Token embeddings: {token_count:,}')
conn.close()
"
```

## ⏱️ Expected Timeline

**Conservative Estimates:**
- **Processing Rate:** ~17.8 docs/sec
- **Total Documents:** ~100,000
- **Estimated Time:** ~5,600 seconds (~1.5 hours)
- **Expected Completion:** ~7:45 PM (May 27, 2025)

**Note:** This is a conservative estimate. Actual time may vary based on document complexity and system performance.

## 🎯 Performance Optimizations Applied

1. **Conservative Batch Sizes:**
   - Document batches: 8 (prevents memory issues)
   - Token batches: 150 (prevents performance degradation)

2. **Checkpointing System:**
   - Progress saved every batch
   - Resumable if interrupted
   - Checkpoint file: `data/conservative_checkpoint.json`

3. **Connection Management:**
   - Frequent connection refresh
   - Proper connection cleanup
   - Error handling and recovery

4. **Memory Management:**
   - Garbage collection between batches
   - Minimal memory footprint
   - Resource cleanup

## 🚨 What to Watch For

### Normal Behavior
- CPU usage: 50-70%
- Memory usage: <1%
- Processing rate: 15-20 docs/sec
- Regular checkpoint updates

### Warning Signs
- CPU usage drops to <10% (may indicate stalling)
- No checkpoint updates for >5 minutes
- Error messages in logs
- Process disappears from `ps` output

## 🛠️ Troubleshooting

### If Process Stops
1. Check logs: `tail -20 logs/conservative_ingestion_20250527_181422.log`
2. Check checkpoint: `cat data/conservative_checkpoint.json`
3. Restart: `python run_conservative_ingestion.py &`

### If Performance Degrades
1. Monitor system resources: `top` or `htop`
2. Check database connections
3. Review error logs for issues

### If Database Issues
1. Check IRIS container: `docker ps`
2. Test connection: `python -c "from common.iris_connector import get_iris_connection; print(get_iris_connection())"`
3. Review database logs

## 📈 Success Metrics

**Target Goals:**
- ✅ 100,000 documents loaded
- ✅ Document embeddings generated for all docs
- ✅ Token embeddings generated for all docs
- ✅ No corruption or LIST ERROR issues
- ✅ Stable performance throughout

**Current Achievement:**
- ✅ Process running stably
- ✅ Conservative optimizations working
- ✅ Checkpointing system functional
- ✅ Both embedding types being generated
- 🔄 In progress: Document loading phase

## 🎉 Next Steps

Once ingestion completes (~7:45 PM):
1. Verify final document count: ~100,000
2. Validate embedding integrity
3. Run benchmark tests
4. Compare performance metrics
5. Document lessons learned

## 📞 Support Commands

**Emergency Stop (if needed):**
```bash
kill 22285
```

**Restart from Checkpoint:**
```bash
python run_conservative_ingestion.py &
```

**Clean Restart (if corruption detected):**
```bash
# Only use if absolutely necessary
python simple_fresh_start.py
python run_conservative_ingestion.py &
```

---

**Status:** ✅ RUNNING SUCCESSFULLY  
**Last Updated:** May 27, 2025 6:15 PM  
**Next Check:** Monitor continuously or check in 30 minutes