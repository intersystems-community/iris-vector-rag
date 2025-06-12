-- Migration script to add bad_document flag to SourceDocuments table
-- This flag indicates documents that failed ingestion or have unusable content

-- Add the bad_document flag column
ALTER TABLE RAG.SourceDocuments 
ADD COLUMN bad_document BOOLEAN DEFAULT FALSE;

-- Create an index for efficient querying
CREATE INDEX idx_sourcedocuments_bad_document ON RAG.SourceDocuments(bad_document);

-- Update existing documents with missing/empty text_content to be marked as bad
UPDATE RAG.SourceDocuments 
SET bad_document = TRUE 
WHERE text_content IS NULL OR TRIM(text_content) = '';

-- Add a comment to document the purpose
COMMENT ON COLUMN RAG.SourceDocuments.bad_document IS 'Flag indicating documents with unusable content that failed ingestion or processing';