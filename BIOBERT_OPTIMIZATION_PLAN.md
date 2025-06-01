# BioBERT Optimization Plan for GraphRAG

## Current State

- **Current embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Entity count**: 114,893 entities in Entities_V2
- **Performance**: ~156ms average query time with HNSW index

## BioBERT Advantages for Medical Data

1. **Domain-specific training**: BioBERT is pre-trained on PubMed abstracts and PMC full-text articles
2. **Better medical entity understanding**: Improved representation of medical terms, diseases, treatments
3. **Higher quality similarity matching**: More accurate semantic similarity for medical concepts

## Implementation Plan

### Option 1: Full Re-embedding (Recommended for best quality)
1. Use BioBERT model (e.g., `dmis-lab/biobert-v1.1` or `allenai/scibert_scivocab_uncased`)
2. Re-generate embeddings for all 114,893 entities
3. Create new Entities_V3 table with BioBERT embeddings
4. Update GraphRAG to use BioBERT for query embeddings

### Option 2: Hybrid Approach (Faster deployment)
1. Keep existing embeddings for now
2. Use BioBERT only for new entities going forward
3. Gradually migrate existing entities

## Technical Considerations

1. **Dimension change**: BioBERT typically uses 768 dimensions (vs current 384)
   - Need to update table schema: `VECTOR(DOUBLE, 768)`
   - HNSW index will need recreation

2. **Performance impact**: 
   - Larger embeddings = slightly more storage
   - HNSW performance should remain excellent
   - Query embedding generation might be slightly slower

3. **Code changes needed**:
   ```python
   # In common/embedding_utils.py
   def get_embedding_model(model_name='dmis-lab/biobert-v1.1'):
       # Load BioBERT model
   ```

## Expected Benefits

1. **Better entity matching**: Medical terms will have more accurate semantic relationships
2. **Improved retrieval**: Queries about diseases/treatments will find more relevant entities
3. **Higher quality answers**: Better context leads to better LLM responses

## Migration Steps

1. Test BioBERT embeddings on a sample of entities
2. Measure quality improvement (relevance scores)
3. If significant improvement, proceed with full migration
4. Create Entities_V3 with 768-dim VECTOR column
5. Re-embed all entities with BioBERT
6. Create HNSW index on Entities_V3
7. Update GraphRAGPipeline to use Entities_V3

## Estimated Timeline

- Testing: 1-2 days
- Full re-embedding: 2-4 hours (depending on GPU availability)
- Migration and testing: 1 day
- Total: 3-5 days

## Decision Point

Given that we already have 114x performance improvement with current embeddings, BioBERT optimization can be considered a **Phase 2 enhancement** focused on quality rather than performance.

**Recommendation**: Proceed with current all-MiniLM-L6-v2 embeddings for now, plan BioBERT migration after validating the current system in production.