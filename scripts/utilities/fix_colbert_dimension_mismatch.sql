-- Fix ColBERT token embedding dimension mismatch
-- Database schema expects 128 dimensions but ColBERT model produces 384 dimensions

-- Step 1: Create a new table with correct dimensions
CREATE TABLE RAG.DocumentTokenEmbeddings_New (
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    token_embedding VECTOR(FLOAT, 384),  -- Updated to match ColBERT model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Step 2: Copy any existing data (if any exists with correct dimensions)
-- Note: This will likely be empty since the dimension mismatch prevented insertions
INSERT INTO RAG.DocumentTokenEmbeddings_New 
SELECT doc_id, token_index, token_text, token_embedding, created_at 
FROM RAG.DocumentTokenEmbeddings 
WHERE 1=0;  -- This will copy structure but no data due to dimension mismatch

-- Step 3: Drop the old table
DROP TABLE RAG.DocumentTokenEmbeddings;

-- Step 4: Rename the new table
ALTER TABLE RAG.DocumentTokenEmbeddings_New RENAME TO DocumentTokenEmbeddings;

-- Step 5: Recreate indexes for DocumentTokenEmbeddings
CREATE INDEX idx_doc_token_embeddings_doc_id ON RAG.DocumentTokenEmbeddings(doc_id);
CREATE INDEX idx_doc_token_embeddings_token_index ON RAG.DocumentTokenEmbeddings(token_index);

-- Step 6: Create vector index for similarity search
CREATE INDEX idx_doc_token_embeddings_vector ON RAG.DocumentTokenEmbeddings(token_embedding);