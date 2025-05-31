# Manual iFind Import Instructions

## Current Situation

HybridIFindRAG is **NOT working** because the %iFind.Index.Basic doesn't exist. We cannot import the ObjectScript class programmatically via Python/JDBC.

## Files Ready for Import

1. **ObjectScript Class**: `objectscript/RAG.SourceDocumentsWithIFind.cls`
   - Already copied to container: `/tmp/RAG.SourceDocumentsWithIFind.cls`

## Manual Import Steps

### Option 1: Direct Terminal Access

1. Access the IRIS container:
   ```bash
   docker exec -it iris_db_rag_licensed bash
   ```

2. Start IRIS terminal:
   ```bash
   iris terminal IRIS -U USER
   ```

3. Import the class:
   ```objectscript
   do $system.OBJ.Load("/tmp/RAG.SourceDocumentsWithIFind.cls", "ck")
   ```

4. Verify import:
   ```objectscript
   write ##class(%Dictionary.ClassDefinition).%ExistsId("RAG.SourceDocumentsWithIFind")
   ```

5. Exit:
   ```objectscript
   halt
   ```

### Option 2: Using IRIS Management Portal

1. Access IRIS Management Portal at http://localhost:52773/csp/sys/UtilHome.csp
2. Navigate to System Explorer > Classes
3. Switch to USER namespace
4. Import the class file

### Option 3: Using VS Code with InterSystems Extension

1. Install InterSystems ObjectScript extension
2. Connect to IRIS instance
3. Open `objectscript/RAG.SourceDocumentsWithIFind.cls`
4. Right-click and select "Import and Compile"

## Verification After Import

Run this Python script to verify:

```python
from common.iris_connector import get_iris_connection

conn = get_iris_connection()
cursor = conn.cursor()

# Test iFind search
cursor.execute("""
    SELECT TOP 5 doc_id, title
    FROM RAG.SourceDocumentsIFind
    WHERE %ID %FIND search_index(TextContentFTI, 'diabetes')
""")

results = cursor.fetchall()
print(f"iFind search returned {len(results)} results")
for doc_id, title in results:
    print(f"  - {doc_id}: {title[:60]}...")

cursor.close()
conn.close()
```

## Expected Result

After successful import:
- The TextContentFTI index will be created automatically
- %FIND search_index() queries will work
- HybridIFindRAG will have full-text search capability

## Current Status

- ❌ ObjectScript class NOT imported
- ❌ %iFind.Index.Basic NOT created
- ❌ HybridIFindRAG full-text search NOT working
- ⚠️ Pipeline falls back to vector/graph search only

## Important Note

**Without manual import of the ObjectScript class, HybridIFindRAG cannot perform its primary function of full-text search.**